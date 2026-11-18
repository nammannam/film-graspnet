"""Verify two extracted Grasp-Anything++ debug datasets against official splits."""

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from filmgrasp.data.dataset import GraspAnythingPPDataset


ID_PATTERN = re.compile(r"^([0-9a-f]{64})_([0-9]+)_([0-9]+)$")
GROUP_PATTERN = re.compile(r"^[0-9a-f]{64}_[0-9]+$")
OFFICIAL_TEST_SHA256 = {
    "test_seen": "da23aea67c5d12691a8be3a8472bafde8773eee5d9ac38503bde7cb1d0437507",
    "test_unseen": "ef5d445f34e50d1df25758f897eebca740cca40287a527ff19779aa4ae287540",
}


def load_ids(path):
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssertionError(f"Cannot read split {path}: {error}") from error
    assert isinstance(values, list) and values, f"Invalid split list: {path}"
    assert all(isinstance(x, str) and GROUP_PATTERN.fullmatch(x) for x in values), path
    assert len(values) == len(set(values)), f"Duplicate split IDs: {path}"
    return set(values)


def inspect_dataset(root, allowed, name):
    path = root / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssertionError(f"Cannot read manifest {path}: {error}") from error
    assert isinstance(manifest, dict) and isinstance(manifest.get("samples"), list), path
    records = manifest["samples"]
    assert records, f"Empty manifest: {path}"
    ids, scenes, groups = set(), set(), set()
    for record in records:
        sample_id = record["sample_id"]
        match = ID_PATTERN.fullmatch(sample_id)
        assert match, f"Invalid sample ID: {sample_id}"
        scene_id = match.group(1)
        group_id = sample_id.rsplit("_", 1)[0]
        assert group_id in allowed, f"{name} split mismatch: {sample_id}"
        assert record["scene_id"] == scene_id, f"Scene mismatch: {sample_id}"
        assert record["object_id"] == match.group(2), f"Object mismatch: {sample_id}"
        assert record["part_id"] == match.group(3), f"Part mismatch: {sample_id}"
        assert sample_id not in ids, f"Duplicate sample ID: {sample_id}"
        assert isinstance(record["prompt"], str) and record["prompt"].strip(), sample_id
        for key, suffix in (("image_path", ".jpg"), ("grasp_path", ".pt")):
            relative = Path(record[key])
            assert not relative.is_absolute() and ".." not in relative.parts, sample_id
            assert relative.suffix == suffix and (root / relative).is_file(), f"Missing {key}: {sample_id}"
        assert Path(record["image_path"]).stem == scene_id, sample_id
        assert Path(record["grasp_path"]).stem == sample_id, sample_id
        ids.add(sample_id)
        scenes.add(scene_id)
        groups.add(group_id)
    assert len(scenes) == manifest["metadata"]["actual_scenes"], name
    assert len(records) == manifest["metadata"]["actual_samples"], name
    dataset = GraspAnythingPPDataset(root)
    assert len(dataset) == len(records), name
    loader = DataLoader(dataset, batch_size=min(2, len(dataset)), shuffle=False)
    loaded = 0
    for batch in loader:
        batch_size = len(batch["prompt"])
        batch_ids = list(batch["sample_id"])
        assert tuple(batch["image"].shape) == (batch_size, 3, 416, 416), batch_ids
        assert torch.isfinite(batch["image"]).all().item(), batch_ids
        assert all(isinstance(p, str) and p.strip() for p in batch["prompt"]), batch_ids
        for key in ("pos", "cos", "sin", "width"):
            tensor = batch["targets"][key]
            assert tuple(tensor.shape) == (batch_size, 1, 416, 416), f"{batch_ids}: {key}"
            assert torch.isfinite(tensor).all().item(), f"{batch_ids}: {key}"
        loaded += batch_size
    assert loaded == len(records), name
    print(f"{name}: {len(scenes)} scenes, {len(records)} samples/prompts, {len(groups)} matched groups")
    return records, scenes, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=Path("splits/grasp-anything-pp"))
    parser.add_argument("--data-dir", type=Path, default=Path("../storage/data"))
    parser.add_argument("--skip-selection", action="store_true", help="Skip remote reproducibility check")
    args = parser.parse_args()

    splits = {name: load_ids(args.split_dir / f"{name}.json") for name in
              ("train_seen", "val_seen", "test_seen", "test_unseen")}
    assert splits["train_seen"].isdisjoint(splits["val_seen"]), "Train/val group overlap"
    tests = splits["test_seen"] | splits["test_unseen"]
    assert splits["test_seen"].isdisjoint(splits["test_unseen"]), "Official test overlap"
    assert (splits["train_seen"] | splits["val_seen"]).isdisjoint(tests), "Train/val test overlap"
    for name, expected_hash in OFFICIAL_TEST_SHA256.items():
        actual_hash = hashlib.sha256((args.split_dir / f"{name}.json").read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Official {name} JSON changed"

    train, train_scenes, train_groups = inspect_dataset(
        args.data_dir / "train_debug", splits["train_seen"], "train_debug")
    print("Train membership: PASSED")
    val, val_scenes, val_groups = inspect_dataset(
        args.data_dir / "val_debug", splits["val_seen"], "val_debug")
    print("Validation membership: PASSED")
    assert train_scenes.isdisjoint(val_scenes), f"Train/val scene overlap: {train_scenes & val_scenes}"
    print("Train/Val scene disjointness: PASSED")
    print("Manifest integrity: PASSED")
    print("Dataset loading: PASSED")
    print("DataLoader compatibility: PASSED")
    assert (train_groups | val_groups).isdisjoint(tests), "Debug sample in official test"
    print("Official split integrity: PASSED")

    if not args.skip_selection:
        from huggingface_hub import HfFileSystem
        spec = importlib.util.spec_from_file_location("prepare_subset", Path(__file__).with_name("prepare_subset.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fs = HfFileSystem()
        for name, records in (("train_seen", train), ("val_seen", val)):
            metadata = json.loads((args.data_dir / ("train_debug" if name == "train_seen" else "val_debug") / "manifest.json").read_text())["metadata"]
            selection = [module.select_instructions(fs, metadata["requested_scenes"],
                         metadata["max_prompts_per_scene"], metadata["seed"], splits[name]) for _ in range(2)]
            ids = [[record["sample_id"] for record in item] for item in selection]
            assert ids[0] == ids[1], f"Nonreproducible selection: {name}"
            assert set(ids[0]) == {record["sample_id"] for record in records}, f"Selection differs from {name} manifest"
        print("Reproducibility: PASSED")
    else:
        print("Reproducibility: SKIPPED (--skip-selection requested)")
    if not args.skip_selection:
        print("OFFICIAL SPLIT DATASET TEST PASSED")


if __name__ == "__main__":
    main()
