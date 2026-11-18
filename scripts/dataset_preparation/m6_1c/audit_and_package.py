"""Audit the four M6.1c datasets and package only a complete, valid extraction."""

import argparse
import hashlib
import json
import re
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_pilot_dataset import inspect_annotation, inspect_item
from filmgrasp.data.dataset import GraspAnythingPPDataset
from filmgrasp.evaluation.grasp_metrics import annotations_to_rectangles
from filmgrasp.training.train import _collate

NAMES = ("train_expanded", "val_expanded", "test_seen", "test_unseen")
OFFICIAL = ("train_seen", "val_seen", "test_seen", "test_unseen")
SAMPLE = re.compile(r"^([0-9a-f]{64})_([0-9]+)_([0-9]+)$")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(data_root, split_root, plan):
    official = {name: set(json.loads((split_root / (name + ".json")).read_text())) for name in OFFICIAL}
    issues = []
    stats = {}
    identities = {}

    def issue(severity, split, sid, name, detail):
        issues.append({"severity": severity, "dataset": split, "sample_id": sid, "check": name, "detail": str(detail)})

    for out, split in zip(NAMES, OFFICIAL):
        base = data_root / out
        path = base / "manifest.json"
        if not path.is_file():
            issue("critical", out, "", "manifest_missing", path)
            stats[out] = {"scenes": 0, "groups": 0, "samples": 0, "unique_prompts": 0}
            identities[out] = {"scenes": set(), "groups": set(), "samples": set()}
            continue
        manifest = json.loads(path.read_text())
        records = manifest.get("samples", [])
        scenes, groups, samples, prompts = set(), set(), set(), []
        scene_objects = defaultdict(set)
        image_hashes = defaultdict(set)
        counts = Counter()
        try:
            dataset = GraspAnythingPPDataset(base, include_annotations=True)
        except Exception as exc:
            issue("critical", out, "", "dataset_init", repr(exc))
            dataset = None
        for i, record in enumerate(records):
            sid = record.get("sample_id", "")
            m = SAMPLE.fullmatch(sid)
            if not m:
                issue("critical", out, sid, "sample_format", "expected scene_object_part")
                continue
            scene, obj, part = m.groups()
            group = sid.rsplit("_", 1)[0]
            scenes.add(scene); groups.add(group)
            scene_objects[scene].add(obj)
            counts[scene] += 1
            if sid in samples:
                issue("critical", out, sid, "duplicate_sample", "duplicate manifest ID")
            samples.add(sid)
            if group not in official[split]:
                issue("critical", out, sid, "official_membership", group)
            for key, expected in (("scene_id", scene), ("object_id", obj), ("part_id", part)):
                if record.get(key) != expected:
                    issue("critical", out, sid, "identity_" + key, record.get(key))
            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                issue("critical", out, sid, "prompt", "empty")
            else:
                prompts.append(prompt)
            for key, dirname, filename in (("image_path", "images", scene + ".jpg"), ("grasp_path", "grasps", sid + ".pt")):
                if record.get(key) != dirname + "/" + filename:
                    issue("critical", out, sid, key, record.get(key))
            ip, gp = base / "images" / (scene + ".jpg"), base / "grasps" / (sid + ".pt")
            if not ip.is_file() or not gp.is_file():
                issue("critical", out, sid, "missing_file", "image or grasp annotation")
                continue
            try:
                with Image.open(ip) as im:
                    im.load()
                    if im.mode != "RGB" or im.size != (416, 416):
                        raise ValueError("expected RGB 416x416")
                image_hashes[sha256(ip)].add(scene)
            except Exception as exc:
                issue("critical", out, sid, "image", repr(exc))
            try:
                a = inspect_annotation(gp)
                if len(a) != record.get("grasp_count"):
                    issue("critical", out, sid, "grasp_count", len(a))
                for rect in annotations_to_rectangles(a):
                    if not np.isfinite(rect.points).all():
                        issue("critical", out, sid, "geometry", "nonfinite rectangle")
                        break
            except Exception as exc:
                issue("critical", out, sid, "annotation", repr(exc))
            if dataset is not None:
                row = {"sample_id": sid, "scene_id": scene, "prompt": prompt}
                try:
                    item = dataset[i]
                    inspect_item(item, row, issue, out)
                    if not torch.equal(item["annotations"], torch.load(gp, map_location="cpu")):
                        issue("critical", out, sid, "original_rectangles", "loader changed annotations")
                except Exception as exc:
                    issue("critical", out, sid, "dataset_item", repr(exc))
        if dataset is not None:
            for batch_size in (1, 4):
                try:
                    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate, num_workers=0)
                    batch = next(iter(loader))
                    n = len(batch["sample_id"])
                    if batch["image"].shape != (n, 3, 416, 416):
                        raise ValueError("bad image batch")
                    for target in ("pos", "cos", "sin", "width"):
                        value = batch["targets"][target]
                        if value.shape != (n, 1, 416, 416) or not torch.isfinite(value).all():
                            raise ValueError("bad " + target + " target batch")
                except Exception as exc:
                    issue("critical", out, "", "dataloader_batch_{}".format(batch_size), repr(exc))
        duplicate_images = sum(len(v) > 1 for v in image_hashes.values())
        if duplicate_images:
            issue("warning", out, "", "duplicate_image_hash", duplicate_images)
        planned = len(plan["members"][out])
        if len(samples) != planned:
            issue("critical", out, "", "extraction_incomplete", "{} of {} planned".format(len(samples), planned))
        if out.startswith("test"):
            available = plan["availability"][out]["instruction_groups"]
            if len(groups) < available:
                issue("critical", out, "", "available_test_groups_missing", "{} of {}".format(len(groups), available))
            if available < len(official[split]):
                issue("warning", out, "", "official_groups_without_instructions", len(official[split]) - available)
        identities[out] = {"scenes": scenes, "groups": groups, "samples": samples}
        stats[out] = {"scenes": len(scenes), "groups": len(groups), "samples": len(samples), "unique_prompts": len(set(prompts)), "multi_object_scenes": sum(len(v) > 1 for v in scene_objects.values()), "multi_object_percent": 100 * sum(len(v) > 1 for v in scene_objects.values()) / len(scenes) if scenes else 0, "mean_samples_per_scene": len(samples) / len(scenes) if scenes else 0, "mean_groups_per_scene": len(groups) / len(scenes) if scenes else 0, "distinct_object_ids": len({sid.rsplit("_", 2)[1] for sid in samples}), "distinct_part_ids": len({sid.rsplit("_", 1)[1] for sid in samples}), "duplicate_image_hashes": duplicate_images, "repeated_prompt_strings": sum(v > 1 for v in Counter(prompts).values())}
        print(out, stats[out], flush=True)
    for i, left in enumerate(NAMES):
        for right in NAMES[i + 1:]:
            for field in ("scenes", "groups", "samples"):
                overlap = identities[left][field] & identities[right][field]
                if overlap:
                    issue("critical", "global", "", "{}_overlap".format(field), "{} vs {}: {}".format(left, right, len(overlap)))
    return stats, issues


def package(data_root, transfer_root, stats, issues, plan):
    if any(x["severity"] == "critical" for x in issues):
        return {}
    transfer_root.mkdir(parents=True, exist_ok=True)
    result = {}
    for out in NAMES:
        destination = transfer_root / (out + ".tar.gz")
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(data_root / out, arcname=out)
        with tarfile.open(destination, "r:gz") as archive:
            names = set(archive.getnames())
            if out + "/manifest.json" not in names or not any(n.startswith(out + "/images/") for n in names) or not any(n.startswith(out + "/grasps/") for n in names):
                raise RuntimeError("missing required files in " + str(destination))
            with tempfile.TemporaryDirectory(dir=transfer_root) as temporary:
                archive.extractall(path=temporary)
                test = json.loads((Path(temporary) / out / "manifest.json").read_text())
                if len(test["samples"]) != stats[out]["samples"]:
                    raise RuntimeError("archive manifest mismatch")
                for record in test["samples"]:
                    for key in ("image_path", "grasp_path"):
                        if not (Path(temporary) / out / record[key]).is_file():
                            raise RuntimeError("archive member missing after extraction: " + record[key])
        result[out] = {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256(destination)}
    (transfer_root / "SHA256SUMS.txt").write_text("".join("{}  {}.tar.gz\n".format(result[n]["sha256"], n) for n in NAMES))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=ROOT.parent / "storage/data")
    ap.add_argument("--run-root", type=Path, default=ROOT / "runs/dataset_preparation/m6_1c")
    ap.add_argument("--transfer-root", type=Path, default=ROOT.parent / "storage/colab_transfer")
    ap.add_argument("--package", action="store_true")
    args = ap.parse_args()
    plan = json.loads((args.run_root / "plan.json").read_text())
    stats, issues = audit(args.data_root, ROOT / "splits/grasp-anything-pp", plan)
    summary = {"status": "PASSED" if not any(i["severity"] == "critical" for i in issues) else "PARTIAL", "statistics": stats, "issue_counts": dict(Counter(i["severity"] for i in issues)), "issues": issues}
    if args.package:
        summary["packages"] = package(args.data_root, args.transfer_root, stats, issues, plan)
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(summary["status"], summary["issue_counts"], flush=True)


if __name__ == "__main__":
    main()
