"""M6.1c: deterministic, resumable official-split extraction.

Run from the film-graspnet repository with the filmgrasp39 environment.
The plan is frozen before any member bytes are read. No pilot files are used.
"""

import argparse
import hashlib
import io
import json
import os
import pickle
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from huggingface_hub import HfFileSystem
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_subset import INSTRUCTION_ZIP, POSITIVE_ZIP, load_split_ids
from filmgrasp.data.annotations import parse_sample_id, validate_positive_grasps
from filmgrasp.data.remote_zip import open_split_image_zip

NAMES = ("train_expanded", "val_expanded", "test_seen", "test_unseen")
OFFICIAL = ("train_seen", "val_seen", "test_seen", "test_unseen")
TARGETS = {"train_expanded": 1200, "val_expanded": 160}
CAPS = {"train_expanded": 2, "val_expanded": 2}
SEED = 42


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def atomic_json(path, data):
    atomic_bytes(path, (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode())


def valid_image(path):
    try:
        with Image.open(path) as im:
            im.load()
            return im.size == (416, 416) and im.mode == "RGB"
    except (OSError, ValueError):
        return False


def valid_label(path):
    try:
        return validate_positive_grasps(path.read_bytes())
    except Exception:
        return None


def split_sets(root):
    return {name: load_split_ids(root / (name + ".json")) for name in OFFICIAL}


def make_plan(archive, splits):
    scene_groups = {name: defaultdict(lambda: defaultdict(list)) for name in NAMES}
    group_split = {}
    for out, official in zip(NAMES, OFFICIAL):
        for group in splits[official]:
            group_split[group] = out
    invalid_names = []
    for member in archive.namelist():
        if not member.endswith(".pkl"):
            continue
        parsed = parse_sample_id(member)
        if parsed is None or len(parsed["scene_id"]) != 64:
            invalid_names.append(member)
            continue
        group = parsed["sample_id"].rsplit("_", 1)[0]
        out = group_split.get(group)
        if out:
            scene_groups[out][parsed["scene_id"]][group].append(member)
    if invalid_names:
        raise ValueError("Invalid instruction member names: " + repr(invalid_names[:3]))

    test_scenes = set(scene_groups["test_seen"]) | set(scene_groups["test_unseen"])
    # Official membership is authoritative even when an instruction is absent.
    test_scenes |= {g.rsplit("_", 1)[0] for k in ("test_seen", "test_unseen") for g in splits[k]}
    excluded = {
        "train_seen_test_scene_overlap": sorted({g.rsplit("_", 1)[0] for g in splits["train_seen"]} & test_scenes),
        "val_seen_test_scene_overlap": sorted({g.rsplit("_", 1)[0] for g in splits["val_seen"]} & test_scenes),
        "test_seen_test_unseen_scene_overlap": sorted({g.rsplit("_", 1)[0] for g in splits["test_seen"]} & {g.rsplit("_", 1)[0] for g in splits["test_unseen"]}),
    }
    chosen = {}
    for out in NAMES:
        pool = scene_groups[out]
        if out in TARGETS:
            eligible = sorted(set(pool) - test_scenes)
            if len(eligible) < TARGETS[out]:
                raise ValueError("{} has only {} source scenes for target {}".format(out, len(eligible), TARGETS[out]))
            rng = random.Random(SEED + (0 if out == "train_expanded" else 1))
            multi = [s for s in eligible if len(pool[s]) > 1]
            single = [s for s in eligible if len(pool[s]) == 1]
            rng.shuffle(multi)
            rng.shuffle(single)
            n = TARGETS[out]
            # Favor multi-object scenes, capped so single-object diversity remains.
            multi_count = min(len(multi), round(n * 0.35))
            selected = multi[:multi_count] + single[:n - multi_count]
            if len(selected) < n:
                selected += multi[multi_count:multi_count + n - len(selected)]
            selected = sorted(selected)
        else:
            selected = sorted(pool)
        chosen[out] = []
        for scene in selected:
            groups = pool[scene]
            if out in CAPS:
                # One candidate per object first; fill remaining slots with parts.
                ordered = [sorted(members)[0] for _, members in sorted(groups.items())]
                ordered += [m for _, members in sorted(groups.items()) for m in sorted(members)[1:]]
                ordered = ordered[:CAPS[out]]
            else:
                ordered = sorted(m for members in groups.values() for m in members)
            chosen[out].extend(ordered)
    availability = {}
    for out, official in zip(NAMES, OFFICIAL):
        found = {parse_sample_id(m)["sample_id"].rsplit("_", 1)[0] for groups in scene_groups[out].values() for members in groups.values() for m in members}
        availability[out] = {"official_groups": len(splits[official]), "instruction_groups": len(found), "missing_instruction_groups": sorted(splits[official] - found), "source_scenes": len(scene_groups[out]), "selected_scenes": len({parse_sample_id(m)["scene_id"] for m in chosen[out]}), "planned_samples": len(chosen[out])}
    return {"seed": SEED, "policy": "35% cap on multi-object scenes, seeded shuffle within strata; two samples per train/val scene, distinct objects first; all instruction members for test groups", "members": chosen, "excluded_scenes": excluded, "availability": availability}


def manifest_for(out, plan):
    official = dict(zip(NAMES, OFFICIAL))[out]
    return {"metadata": {"seed": SEED, "split_file": "splits/grasp-anything-pp/{}.json".format(official), "selection_policy": plan["policy"], "planned_samples": len(plan["members"][out]), "instruction_source": "Grasp-Anything++", "image_source": "Grasp-Anything"}, "samples": []}


def extract(fs, data_root, plan, only=None, limit_scenes=None):
    names = [only] if only else NAMES
    state = {}
    for out in names:
        path = data_root / out / "manifest.json"
        state[out] = json.loads(path.read_text()) if path.exists() else manifest_for(out, plan)
    failures = defaultdict(list)
    with fs.open(INSTRUCTION_ZIP, "rb") as source, ZipFile(source) as instructions:
        with fs.open(POSITIVE_ZIP, "rb") as source_labels, ZipFile(source_labels) as labels:
            with open_split_image_zip(fs) as images:
                for out in names:
                    root = data_root / out
                    manifest = state[out]
                    existing = {r["sample_id"]: r for r in manifest["samples"]}
                    members = plan["members"][out]
                    if limit_scenes is not None:
                        first_scenes = set(sorted({parse_sample_id(m)["scene_id"] for m in members})[:limit_scenes])
                        members = [m for m in members if parse_sample_id(m)["scene_id"] in first_scenes]
                    for i, member in enumerate(members, 1):
                        parsed = parse_sample_id(member)
                        sid, scene = parsed["sample_id"], parsed["scene_id"]
                        ip = root / "images" / (scene + ".jpg")
                        gp = root / "grasps" / (sid + ".pt")
                        try:
                            if not valid_image(ip):
                                raw = images.read("image/{}.jpg".format(scene))
                                with Image.open(io.BytesIO(raw)) as im:
                                    im.load()
                                    if im.size != (416, 416) or im.mode != "RGB":
                                        raise ValueError("image is not RGB 416x416")
                                atomic_bytes(ip, raw)
                            count = valid_label(gp)
                            if count is None:
                                raw = labels.read("grasp_label_positive/{}.pt".format(sid))
                                count = validate_positive_grasps(raw)
                                atomic_bytes(gp, raw)
                            if sid in existing and existing[sid].get("grasp_count") == count:
                                continue
                            prompt = pickle.loads(instructions.read(member))
                            if not isinstance(prompt, str) or not prompt.strip():
                                raise ValueError("missing or empty instruction")
                            record = {**parsed, "prompt": prompt.strip(), "image_path": "images/{}.jpg".format(scene), "grasp_path": "grasps/{}.pt".format(sid), "image_size": [416, 416], "grasp_count": count}
                            existing[sid] = record
                            manifest["samples"] = [existing[k] for k in sorted(existing)]
                            manifest["metadata"].update(actual_scenes=len({r["scene_id"] for r in manifest["samples"]}), actual_samples=len(existing), matched_group_ids=len({k.rsplit("_", 1)[0] for k in existing}))
                            atomic_json(root / "manifest.json", manifest)
                        except Exception as exc:
                            failures[out].append({"sample_id": sid, "error": "{}: {}".format(type(exc).__name__, exc)})
                        if i % 25 == 0 or i == len(members):
                            print("{}: {}/{} planned, {} complete, {} failures".format(out, i, len(members), len(existing), len(failures[out])), flush=True)
                    state[out] = manifest
    return failures


def extract_parallel(data_root, plan, workers, only=None, limit_scenes=None):
    """Each worker owns seekable ZIP handles; one task owns one scene's files."""
    names = [only] if only else NAMES
    local = threading.local()
    stacks = []
    stacks_lock = threading.Lock()
    failures = defaultdict(list)

    def archives():
        if not hasattr(local, "archives"):
            stack = ExitStack()
            fs = HfFileSystem()
            instructions = stack.enter_context(ZipFile(stack.enter_context(fs.open(INSTRUCTION_ZIP, "rb"))))
            labels = stack.enter_context(ZipFile(stack.enter_context(fs.open(POSITIVE_ZIP, "rb"))))
            images = stack.enter_context(open_split_image_zip(fs))
            local.archives = (instructions, labels, images)
            with stacks_lock:
                stacks.append(stack)
        return local.archives

    def scene_task(out, scene, members, existing):
        root = data_root / out
        instructions, labels, images = archives()
        ip = root / "images" / (scene + ".jpg")
        records, errors = [], []
        try:
            if not valid_image(ip):
                raw = images.read("image/{}.jpg".format(scene))
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
                    if im.size != (416, 416) or im.mode != "RGB":
                        raise ValueError("image is not RGB 416x416")
                atomic_bytes(ip, raw)
        except Exception as exc:
            return records, [{"sample_id": parse_sample_id(m)["sample_id"], "error": "image {}: {}".format(type(exc).__name__, exc)} for m in members]
        for member in members:
            parsed = parse_sample_id(member)
            sid = parsed["sample_id"]
            gp = root / "grasps" / (sid + ".pt")
            try:
                count = valid_label(gp)
                if count is None:
                    raw = labels.read("grasp_label_positive/{}.pt".format(sid))
                    count = validate_positive_grasps(raw)
                    atomic_bytes(gp, raw)
                if sid in existing and existing[sid].get("grasp_count") == count:
                    records.append(existing[sid])
                    continue
                prompt = pickle.loads(instructions.read(member))
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("missing or empty instruction")
                records.append({**parsed, "prompt": prompt.strip(), "image_path": "images/{}.jpg".format(scene), "grasp_path": "grasps/{}.pt".format(sid), "image_size": [416, 416], "grasp_count": count})
            except Exception as exc:
                errors.append({"sample_id": sid, "error": "{}: {}".format(type(exc).__name__, exc)})
        return records, errors

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for out in names:
                root = data_root / out
                path = root / "manifest.json"
                manifest = json.loads(path.read_text()) if path.exists() else manifest_for(out, plan)
                existing = {r["sample_id"]: r for r in manifest["samples"]}
                by_scene = defaultdict(list)
                for member in plan["members"][out]:
                    by_scene[parse_sample_id(member)["scene_id"]].append(member)
                selected = sorted(by_scene)
                if limit_scenes is not None:
                    selected = selected[:limit_scenes]
                futures = {executor.submit(scene_task, out, scene, by_scene[scene], existing): scene for scene in selected}
                for i, future in enumerate(as_completed(futures), 1):
                    scene = futures[future]
                    try:
                        records, errors = future.result()
                    except Exception as exc:
                        records, errors = [], [{"sample_id": parse_sample_id(m)["sample_id"], "error": "worker {}: {}".format(type(exc).__name__, exc)} for m in by_scene[scene]]
                    failures[out].extend(errors)
                    changed = False
                    for record in records:
                        sid = record["sample_id"]
                        if sid not in existing:
                            existing[sid] = record
                            changed = True
                    if changed:
                        manifest["samples"] = [existing[k] for k in sorted(existing)]
                        manifest["metadata"].update(actual_scenes=len({r["scene_id"] for r in manifest["samples"]}), actual_samples=len(existing), matched_group_ids=len({k.rsplit("_", 1)[0] for k in existing}))
                        atomic_json(path, manifest)
                    if i % 25 == 0 or i == len(selected):
                        print("{}: {}/{} scenes, {} samples, {} failures".format(out, i, len(selected), len(existing), len(failures[out])), flush=True)
    finally:
        for stack in stacks:
            stack.close()
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=ROOT.parent / "storage/data")
    ap.add_argument("--run-root", type=Path, default=ROOT / "runs/dataset_preparation/m6_1c")
    ap.add_argument("--limit-scenes", type=int, help="Small stage only; do not reuse its plan for full extraction")
    ap.add_argument("--only", choices=NAMES)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="Independent remote ZIP readers (1-8)")
    args = ap.parse_args()
    if args.limit_scenes is not None and args.limit_scenes <= 0:
        ap.error("--limit-scenes must be positive")
    if not 1 <= args.workers <= 8:
        ap.error("--workers must be in [1,8]")
    args.run_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.run_root / "plan.json"
    fs = HfFileSystem()
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        splits = split_sets(ROOT / "splits/grasp-anything-pp")
        with fs.open(INSTRUCTION_ZIP, "rb") as source, ZipFile(source) as archive:
            plan = make_plan(archive, splits)
        atomic_json(plan_path, plan)
    print("Plan:", plan_path, flush=True)
    print(json.dumps({k: {n: (len(v) if isinstance(v, list) else v) for n, v in a.items()} for k, a in plan["availability"].items()}, indent=2), flush=True)
    if args.plan_only:
        return
    failures = (extract_parallel(args.data_root, plan, args.workers, args.only, args.limit_scenes)
                if args.workers > 1 else extract(fs, args.data_root, plan, args.only, args.limit_scenes))
    atomic_json(args.run_root / "extraction_failures.json", failures)
    print("Failures:", {k: len(v) for k, v in failures.items()}, flush=True)


if __name__ == "__main__":
    main()
