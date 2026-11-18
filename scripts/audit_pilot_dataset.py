"""Offline integrity and quality audit for Grasp-Anything++ pilot datasets."""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filmgrasp.data.dataset import GraspAnythingPPDataset
from filmgrasp.evaluation.grasp_metrics import annotations_to_rectangles
from filmgrasp.training.train import _collate


SAMPLE_RE = re.compile(r"^([0-9a-f]{64})_([0-9]+)_([0-9]+)$")
GROUP_RE = re.compile(r"^[0-9a-f]{64}_[0-9]+$")
TARGETS = ("pos", "cos", "sin", "width")
EXPECTED = {"train": (200, 222, 381), "validation": (40, 48, 76)}


def digest(path):
    """Return SHA-256 of a local file without loading it entirely into memory."""
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def distribution(values):
    """Summarize a numeric sequence."""
    if not values:
        return {"count": 0}
    a = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "max": float(a.max())}


def load_split(path, issue):
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list) or not values:
            raise ValueError("expected a nonempty JSON list")
        if any(not isinstance(x, str) or not GROUP_RE.fullmatch(x) for x in values):
            raise ValueError("invalid group ID")
        if len(values) != len(set(values)):
            raise ValueError("duplicate group ID")
        return set(values)
    except (OSError, ValueError) as exc:
        issue("critical", "global", "", "split_file", "{}: {}".format(path, exc))
        return set()


def safe_path(root, value, suffix, issue, split, sample_id, name):
    if not isinstance(value, str) or not value:
        issue("critical", split, sample_id, name, "missing or invalid path")
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != suffix:
        issue("critical", split, sample_id, name, "unsafe path or wrong suffix: {}".format(value))
        return None
    resolved = (root / relative).resolve()
    if root.resolve() not in resolved.parents or not resolved.is_file():
        issue("critical", split, sample_id, name, "missing file or path escapes dataset: {}".format(value))
        return None
    return resolved


def inspect_image(path):
    with Image.open(path) as image:
        fmt, size = image.format, image.size
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return {"format": fmt, "size": list(size), "mean": float(rgb.mean()),
            "min": int(rgb.min()), "max": int(rgb.max()),
            "constant": bool(np.all(rgb == rgb.flat[0])), "sha256": digest(path)}


def inspect_annotation(path):
    # Only files in the trusted, already extracted local dataset are deserialized.
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, (torch.Tensor, np.ndarray)):
        raise ValueError("annotation is not a numeric tensor or ndarray")
    if not np.issubdtype(np.asarray(obj).dtype, np.number):
        raise ValueError("annotation dtype is not numeric")
    a = np.asarray(obj)
    if a.ndim != 2 or a.shape[1] != 6 or len(a) == 0:
        raise ValueError("expected nonempty [N,6], got {}".format(a.shape))
    if not np.isfinite(a).all():
        raise ValueError("NaN or Inf annotation value")
    if np.any((a[:, 1:3] < 0) | (a[:, 1:3] >= 416)):
        raise ValueError("grasp center outside [0,416)")
    if np.any(a[:, 3:5] <= 0):
        raise ValueError("nonpositive grasp length or width")
    return a


def inspect_item(item, row, issue, split):
    sample_id = row["sample_id"]
    image = item["image"]
    if image.shape != (3, 416, 416) or image.dtype != torch.float32 or not torch.isfinite(image).all():
        issue("critical", split, sample_id, "dataset_image", "invalid image tensor shape, dtype or values")
    elif image.min() < 0 or image.max() > 1:
        issue("critical", split, sample_id, "dataset_image", "RGB tensor outside [0,1]")
    if item["sample_id"] != sample_id or item["scene_id"] != row["scene_id"] or item["prompt"] != row["prompt"]:
        issue("critical", split, sample_id, "dataset_identity", "Dataset item disagrees with manifest")
    targets = item["targets"]
    for name in TARGETS:
        t = targets.get(name)
        if t is None or t.shape != (1, 416, 416) or t.dtype != torch.float32 or not torch.isfinite(t).all():
            issue("critical", split, sample_id, "target_" + name, "invalid target shape, dtype or values")
            continue
        row[name + "_min"] = float(t.min())
        row[name + "_max"] = float(t.max())
        if name in ("pos", "width") and (t.min() < 0 or t.max() > 1):
            issue("critical", split, sample_id, "target_" + name, "target outside [0,1]")
        if name in ("cos", "sin") and (t.min() < -1.00001 or t.max() > 1.00001):
            issue("critical", split, sample_id, "target_" + name, "angle target outside [-1,1]")
    if all(name + "_min" in row for name in TARGETS):
        pos = targets["pos"]
        count = int((pos > 0).sum())
        row["positive_pixels"] = count
        row["positive_pixel_ratio"] = count / (416 * 416)
        if not torch.all((pos == 0) | (pos == 1)):
            issue("critical", split, sample_id, "target_pos", "position map is not binary")
        if count == 0:
            issue("critical", split, sample_id, "empty_target", "valid annotation produced empty target")
        elif count < 10:
            issue("warning", split, sample_id, "small_target", "fewer than 10 positive pixels")
        foreground = pos > 0
        norm = targets["cos"][foreground] ** 2 + targets["sin"][foreground] ** 2
        if norm.numel() and torch.max(torch.abs(norm - 1)) > 1e-4:
            issue("critical", split, sample_id, "target_angle", "foreground cos/sin norm differs from one")
        if torch.any(targets["width"][~foreground] != 0):
            issue("critical", split, sample_id, "target_width", "background width is nonzero")


def audit(args):
    anomalies = []

    def issue(severity, split, sample_id, name, description):
        anomalies.append({"severity": severity, "split": split, "sample_id": sample_id,
                          "check_name": name, "description": description})

    split_paths = {name: args.split_root / (name + ".json") for name in
                   ("train_seen", "val_seen", "test_seen", "test_unseen")}
    split_ids = {name: load_split(path, issue) for name, path in split_paths.items()}
    if split_ids["train_seen"] & split_ids["val_seen"]:
        issue("critical", "global", "", "official_split_overlap", "official train and validation groups overlap")
    if (split_ids["train_seen"] | split_ids["val_seen"]) & (split_ids["test_seen"] | split_ids["test_unseen"]):
        issue("critical", "global", "", "official_test_overlap", "official training/validation groups intersect test groups")

    rows, stats, provenance, inventories = [], {}, {}, {}
    image_cache = {}
    all_ids = set()
    sets = {}
    for split, root in (("train", args.train_root), ("validation", args.val_root)):
        print("Auditing {}: {}".format(split, root), flush=True)
        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = manifest["samples"]
            if not isinstance(records, list) or not records:
                raise ValueError("samples is empty or not a list")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issue("critical", split, "", "manifest", str(exc))
            continue
        meta = manifest.get("metadata", {})
        if not isinstance(meta, dict):
            issue("critical", split, "", "metadata", "metadata is not an object")
            meta = {}
        provenance[split] = {"metadata": meta, "manifest_sha256": digest(manifest_path),
                             "split_sha256": digest(split_paths["train_seen" if split == "train" else "val_seen"])
                             if split_paths["train_seen" if split == "train" else "val_seen"].is_file() else None}
        if meta.get("split_file") != "splits/grasp-anything-pp/{}.json".format("train_seen" if split == "train" else "val_seen"):
            issue("critical", split, "", "metadata_split", "metadata split_file disagrees with expected split")
        for key in ("seed", "requested_scenes", "max_prompts_per_scene"):
            if type(meta.get(key)) is not int or meta[key] <= 0:
                issue("critical", split, "", "metadata_" + key, "missing or invalid extraction parameter")
        scenes, groups, prompts, counts, lengths, widths, angles, pixels = set(), set(), [], [], [], [], [], []
        scene_objects, group_parts, group_prompts, scene_counts, group_counts = defaultdict(set), defaultdict(set), defaultdict(set), Counter(), Counter()
        ids = set()
        file_hashes = {}
        dataset = None
        try:
            dataset = GraspAnythingPPDataset(root, include_annotations=True)
            if len(dataset) != len(records):
                issue("critical", split, "", "dataset_length", "dataset length differs from manifest")
        except Exception as exc:
            issue("critical", split, "", "dataset_init", repr(exc))
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                issue("critical", split, "", "manifest_record", "record {} is not an object".format(index))
                record = {}
            sid = record.get("sample_id", "")
            sid = sid if isinstance(sid, str) else ""
            row = {"split": split, "sample_id": sid, "scene_id": record.get("scene_id", ""),
                   "group_id": sid.rsplit("_", 1)[0] if "_" in sid else "",
                   "prompt": record.get("prompt", ""), "grasp_count": record.get("grasp_count", ""),
                   "image_valid": False, "annotation_valid": False, "targets_valid": False,
                   "positive_pixel_ratio": ""}
            rows.append(row)
            match = SAMPLE_RE.fullmatch(sid)
            if not match:
                issue("critical", split, sid, "sample_id", "invalid sample ID format")
            else:
                scene, obj, part = match.groups()
                for field, expected in (("scene_id", scene), ("object_id", obj), ("part_id", part)):
                    if record.get(field) != expected:
                        issue("critical", split, sid, field, "field disagrees with sample ID")
                scenes.add(scene)
                groups.add(row["group_id"])
                scene_objects[scene].add(obj)
                group_parts[row["group_id"]].add(part)
                scene_counts[scene] += 1
                group_counts[row["group_id"]] += 1
                allowed = split_ids["train_seen" if split == "train" else "val_seen"]
                if row["group_id"] not in allowed:
                    issue("critical", split, sid, "split_membership", "group absent from official split")
                if row["group_id"] in split_ids["test_seen"] | split_ids["test_unseen"]:
                    issue("critical", split, sid, "test_leakage", "group belongs to official test set")
            if sid in ids or sid in all_ids:
                issue("critical", split, sid, "duplicate_sample", "duplicate sample ID")
            ids.add(sid)
            all_ids.add(sid)
            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                issue("critical", split, sid, "prompt", "empty or non-string prompt")
            else:
                try:
                    prompt.encode("utf-8", errors="strict")
                    if any(ord(c) < 32 and c not in "\t\n\r" for c in prompt):
                        issue("warning", split, sid, "prompt_control", "prompt contains control character")
                    prompts.append(prompt)
                    group_prompts[row["group_id"]].add(prompt)
                except UnicodeError:
                    issue("critical", split, sid, "prompt_unicode", "prompt cannot encode to UTF-8")
            if record.get("image_size") != [416, 416]:
                issue("critical", split, sid, "manifest_image_size", "image_size is not [416,416]")
            ip = safe_path(root, record.get("image_path"), ".jpg", issue, split, sid, "image_path")
            gp = safe_path(root, record.get("grasp_path"), ".pt", issue, split, sid, "grasp_path")
            if ip is not None:
                if match and ip.stem != match.group(1):
                    issue("critical", split, sid, "image_filename", "image filename disagrees with scene ID")
                if ip not in image_cache:
                    try:
                        image_cache[ip] = inspect_image(ip)
                    except Exception as exc:
                        image_cache[ip] = {"error": repr(exc)}
                info = image_cache[ip]
                if "error" in info:
                    issue("critical", split, sid, "image_read", info["error"])
                else:
                    file_hashes[str(ip.relative_to(root.resolve()))] = info["sha256"]
                    row["image_valid"] = info["format"] in ("JPEG", "PNG") and info["size"] == [416, 416]
                    if not row["image_valid"]:
                        issue("critical", split, sid, "image_format_size", "format={} size={}".format(info["format"], info["size"]))
                    if info["constant"]:
                        issue("warning", split, sid, "constant_image", "RGB image is constant valued")
            if gp is not None:
                if match and gp.stem != sid:
                    issue("critical", split, sid, "grasp_filename", "annotation filename disagrees with sample ID")
                try:
                    a = inspect_annotation(gp)
                    file_hashes[str(gp.relative_to(root.resolve()))] = digest(gp)
                    row["annotation_valid"] = True
                    counts.append(len(a))
                    lengths.extend(a[:, 3].tolist())
                    widths.extend(a[:, 4].tolist())
                    angles.extend(a[:, 5].tolist())
                    if type(record.get("grasp_count")) is not int or record["grasp_count"] != len(a):
                        issue("critical", split, sid, "grasp_count", "manifest count disagrees with annotation")
                    if np.any((a[:, 5] < 0) | (a[:, 5] > 180)):
                        issue("warning", split, sid, "angle_range", "angle falls outside observed Grasp-Anything [0,180] degree convention")
                    rectangles = annotations_to_rectangles(a)
                    for rect in rectangles:
                        if not np.isfinite(rect.points).all():
                            issue("critical", split, sid, "geometry", "nonfinite rectangle coordinates")
                            break
                    if any(np.any((r.points < 0) | (r.points >= 416)) for r in rectangles):
                        issue("info", split, sid, "rectangle_border", "at least one rectangle extends outside image")
                except Exception as exc:
                    issue("critical", split, sid, "annotation", repr(exc))
            if dataset is not None:
                before = len(anomalies)
                try:
                    item = dataset[index]
                    inspect_item(item, row, issue, split)
                    if row["annotation_valid"] and (not torch.equal(item["annotations"], torch.load(gp, map_location="cpu"))):
                        issue("critical", split, sid, "dataset_annotations", "Dataset annotations differ from file")
                    row["targets_valid"] = not any(x["severity"] == "critical" for x in anomalies[before:])
                    if row.get("positive_pixel_ratio") != "":
                        pixels.append(row["positive_pixel_ratio"])
                except Exception as exc:
                    issue("critical", split, sid, "dataset_item", repr(exc))
            if (index + 1) % 50 == 0 or index + 1 == len(records):
                print("  {}/{} samples".format(index + 1, len(records)), flush=True)
        sets[split] = {"scenes": scenes, "groups": groups, "ids": ids}
        expected = EXPECTED[split]
        if (len(scenes), len(groups), len(records)) != expected:
            issue("critical", split, "", "expected_size", "actual scenes/groups/samples={}/{}/{} expected={}".format(len(scenes), len(groups), len(records), expected))
        for key, actual in (("actual_scenes", len(scenes)), ("actual_samples", len(records)), ("matched_group_ids", len(groups))):
            if meta.get(key) != actual:
                issue("critical", split, "", "metadata_" + key, "metadata={} actual={}".format(meta.get(key), actual))
        if type(meta.get("requested_scenes")) is int and len(scenes) != meta["requested_scenes"]:
            issue("critical", split, "", "metadata_requested_scenes", "selected scene count differs from request")
        if type(meta.get("max_prompts_per_scene")) is int and any(n > meta["max_prompts_per_scene"] for n in scene_counts.values()):
            issue("critical", split, "", "metadata_max_prompts_per_scene", "scene exceeds extraction cap")
        inventory_text = "\n".join("{} {}".format(p, h) for p, h in sorted(file_hashes.items()))
        inventories[split] = {"file_count": len(file_hashes), "sha256": hashlib.sha256(inventory_text.encode()).hexdigest()}
        stats[split] = {"unique_scenes": len(scenes), "unique_object_groups": len(groups), "total_samples": len(records),
                        "unique_prompts": len(set(prompts)), "prompt_word_count": distribution([len(p.split()) for p in prompts]),
                        "repeated_prompt_strings": sum(n > 1 for n in Counter(prompts).values()),
                        "mean_samples_per_scene": len(records) / len(scenes) if scenes else None,
                        "samples_per_scene": distribution(list(scene_counts.values())),
                        "prompts_per_group": distribution(list(group_counts.values())),
                        "scenes_with_multiple_objects": sum(len(v) > 1 for v in scene_objects.values()),
                        "scenes_with_multiple_objects_percent": 100 * sum(len(v) > 1 for v in scene_objects.values()) / len(scenes) if scenes else None,
                        "groups_with_multiple_parts": sum(len(v) > 1 for v in group_parts.values()),
                        "groups_with_distinct_prompts": sum(len(v) > 1 for v in group_prompts.values()),
                        "total_grasp_rectangles": sum(counts), "grasps_per_sample": distribution(counts),
                        "samples_with_one_grasp": sum(n == 1 for n in counts),
                        "grasp_length": distribution(lengths), "grasp_width": distribution(widths),
                        "angle_degrees": distribution(angles), "positive_pixel_ratio": distribution(pixels),
                        "empty_target_maps": sum(x["check_name"] == "empty_target" and x["split"] == split for x in anomalies),
                        "unique_images": len({p for p in image_cache if root.resolve() in p.parents})}
        split_images = [info for path, info in image_cache.items() if root.resolve() in path.parents and "error" not in info]
        stats[split]["image_size_distribution"] = dict(Counter("{}x{}".format(*info["size"]) for info in split_images))
        stats[split]["image_rgb_mean_intensity"] = distribution([info["mean"] for info in split_images])
        stats[split]["shared_image_references"] = len(records) - len(split_images)
        if dataset is not None:
            for batch_size in (1, 2):
                try:
                    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate)
                    count = 0
                    for batch in loader:
                        b = len(batch["sample_id"])
                        if batch["image"].shape != (b, 3, 416, 416) or len(batch["prompt"]) != b or len(batch["annotations"]) != b:
                            raise ValueError("invalid batch image/prompt/annotation structure")
                        if not torch.isfinite(batch["image"]).all():
                            raise ValueError("nonfinite batch image")
                        for name in TARGETS:
                            t = batch["targets"][name]
                            if t.shape != (b, 1, 416, 416) or not torch.isfinite(t).all():
                                raise ValueError("invalid batch target " + name)
                        for a in batch["annotations"]:
                            if a.ndim != 2 or a.shape[1] != 6 or len(a) == 0:
                                raise ValueError("invalid variable-length annotations")
                        count += b
                    if count != len(records):
                        raise ValueError("iterated {} of {}".format(count, len(records)))
                    stats[split]["dataloader_batch_{}".format(batch_size)] = "passed"
                except Exception as exc:
                    issue("critical", split, "", "dataloader_batch_{}".format(batch_size), repr(exc))
                    stats[split]["dataloader_batch_{}".format(batch_size)] = "failed"
    if "train" in sets and "validation" in sets:
        for key in ("scenes", "groups", "ids"):
            overlap = sets["train"][key] & sets["validation"][key]
            if overlap:
                issue("critical", "global", "", "train_val_" + key + "_overlap", "{} overlaps; examples: {}".format(len(overlap), sorted(overlap)[:3]))
    hashes = defaultdict(set)
    for path, info in image_cache.items():
        if "sha256" in info:
            hashes[info["sha256"]].add(path.stem)
    content_duplicates = [sorted(v) for v in hashes.values() if len(v) > 1]
    if content_duplicates:
        issue("warning", "global", "", "duplicate_image_content", "{} content hashes occur under different scene IDs".format(len(content_duplicates)))
    for row in rows:
        related = [a for a in anomalies if a["split"] == row["split"] and a["sample_id"] == row["sample_id"]]
        row["warnings"] = "; ".join(a["check_name"] for a in related if a["severity"] == "warning")
        row["errors"] = "; ".join(a["check_name"] for a in related if a["severity"] == "critical")
    for split in stats:
        stats[split]["invalid_records"] = sum(bool(r["errors"]) for r in rows if r["split"] == split)
    counts = Counter(a["severity"] for a in anomalies)
    checks = {name: "failed" if any(a["severity"] == "critical" and a["check_name"] in names for a in anomalies) else "passed"
              for name, names in {
                  "official_split_integrity": {"split_file", "official_split_overlap", "official_test_overlap", "split_membership", "test_leakage", "train_val_scenes_overlap", "train_val_groups_overlap", "train_val_ids_overlap"},
                  "manifest_integrity": {"manifest", "manifest_record", "sample_id", "scene_id", "object_id", "part_id", "duplicate_sample", "image_path", "grasp_path", "image_filename", "grasp_filename", "manifest_image_size", "grasp_count", "expected_size", "metadata_actual_scenes", "metadata_actual_samples", "metadata_matched_group_ids", "metadata_split"},
                  "rgb_image_validation": {"image_read", "image_format_size", "dataset_image"},
                  "annotation_validation": {"annotation", "geometry", "dataset_annotations"},
                  "dense_target_validation": {"dataset_item", "empty_target", "target_pos", "target_cos", "target_sin", "target_width", "target_angle"},
                  "dataloader_compatibility": {"dataloader_batch_1", "dataloader_batch_2", "dataset_init", "dataset_length"}}.items()}
    summary = {"audit_date_utc": datetime.now(timezone.utc).isoformat(),
               "dataset_paths": {"train": str(args.train_root.resolve()), "validation": str(args.val_root.resolve())},
               "samples_checked": {s: sum(r["split"] == s for r in rows) for s in ("train", "validation")},
               "images_checked": len(image_cache), "checks": checks,
               "critical_errors": counts["critical"], "warnings": counts["warning"],
               "informational_findings": counts["info"], "status": "FAILED" if counts["critical"] else "PASSED"}
    stats["duplicate_image_content_scene_groups"] = content_duplicates
    stats["provenance"] = provenance
    stats["inventory_hashes"] = inventories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "dataset_statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    for filename, data, fields in (("sample_audit.csv", rows, ["split", "sample_id", "scene_id", "group_id", "prompt", "grasp_count", "image_valid", "annotation_valid", "targets_valid", "positive_pixel_ratio", "positive_pixels", "pos_min", "pos_max", "cos_min", "cos_max", "sin_min", "sin_max", "width_min", "width_max", "warnings", "errors"]),
                                   ("anomalies.csv", anomalies, ["severity", "split", "sample_id", "check_name", "description"])):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
    print("Train samples checked: {}/381".format(summary["samples_checked"]["train"]))
    print("Validation samples checked: {}/76".format(summary["samples_checked"]["validation"]))
    for name, state in checks.items():
        print("{}: {}".format(name, state.upper()))
    print("Critical errors: {}  Warnings: {}".format(counts["critical"], counts["warning"]))
    print("M6.1B FULL PILOT DATASET AUDIT {}".format(summary["status"]))
    return summary, stats, anomalies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=Path("../storage/data/train_pilot"))
    parser.add_argument("--val-root", type=Path, default=Path("../storage/data/val_pilot"))
    parser.add_argument("--split-root", type=Path, default=Path("splits/grasp-anything-pp"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dataset_audits/m6_1b"))
    summary, _, _ = audit(parser.parse_args())
    return 1 if summary["status"] == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
