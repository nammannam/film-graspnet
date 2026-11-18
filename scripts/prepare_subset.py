
import argparse
import io
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from huggingface_hub import HfFileSystem

from filmgrasp.data.remote_zip import open_split_image_zip
from filmgrasp.data.annotations import (
    parse_sample_id,
    validate_positive_grasps,
)



PP_REPO = "datasets/airvlab/Grasp-Anything-pp"

INSTRUCTION_ZIP = f"{PP_REPO}/grasp_instructions.zip"
POSITIVE_ZIP = f"{PP_REPO}/grasp_label_positive.zip"


def load_split_ids(path):
    try:
        ids = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Split file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid split JSON {path}: {error}") from error
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"Split JSON must be a nonempty list: {path}")
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError(f"Split JSON must contain nonempty string IDs: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate group IDs in split JSON: {path}")
    for group_id in ids:
        scene, separator, object_id = group_id.rpartition("_")
        if not separator or len(scene) != 64 or any(c not in "0123456789abcdef" for c in scene) or not object_id.isdigit():
            raise ValueError(f"Invalid group ID in {path}: {group_id}")
    return set(ids)


def select_instructions(fs, num_scenes, prompts_per_scene, seed, allowed_ids=None):
    rng = random.Random(seed)

    try:
        remote = fs.open(INSTRUCTION_ZIP, "rb")
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError(f"Missing or inaccessible instruction archive: {INSTRUCTION_ZIP}: {error}") from error
    with remote:
        with ZipFile(remote) as archive:
            infos = archive.infolist()

            if allowed_ids is None:
                # Preserve the original fast random probe for unsplit subsets.
                count = min(len(infos), num_scenes * 40)
                indices = rng.sample(range(len(infos)), count)
                selected_scenes = set()
                for index in indices:
                    name = infos[index].filename
                    if name.endswith(".pkl"):
                        parsed = parse_sample_id(name)
                        if parsed:
                            selected_scenes.add(parsed["scene_id"])
                    if len(selected_scenes) >= num_scenes:
                        break
            else:
                # ZIP central directory names are cheap to scan; member bytes
                # are read only after split filtering and scene selection.
                eligible_scenes = set()
                for info in infos:
                    name = info.filename
                    if not name.endswith(".pkl"):
                        continue
                    parsed = parse_sample_id(name)
                    if parsed is None:
                        raise ValueError(f"Invalid annotation naming: {name}")
                    group_id = parsed["sample_id"].rsplit("_", 1)[0]
                    if group_id in allowed_ids:
                        eligible_scenes.add(parsed["scene_id"])
                if len(eligible_scenes) < num_scenes:
                    raise ValueError(
                        f"Requested {num_scenes} scenes but only "
                        f"{len(eligible_scenes)} match the split in {INSTRUCTION_ZIP}"
                    )
                selected_scenes = set(rng.sample(sorted(eligible_scenes), num_scenes))

            if not selected_scenes:
                raise RuntimeError("No scenes found")

            # Reservoir sampling: select prompts from
            # each scene without keeping all filenames.
            reservoirs = {
                scene: [] for scene in selected_scenes
            }
            seen_counts = defaultdict(int)

            for info in infos:
                name = info.filename

                if not name.endswith(".pkl"):
                    continue

                parsed = parse_sample_id(name)

                if parsed is None:
                    continue

                if allowed_ids is not None and parsed["sample_id"].rsplit("_", 1)[0] not in allowed_ids:
                    continue

                scene = parsed["scene_id"]

                if scene not in selected_scenes:
                    continue

                seen_counts[scene] += 1
                bucket = reservoirs[scene]

                if len(bucket) < prompts_per_scene:
                    bucket.append(name)
                else:
                    j = rng.randrange(seen_counts[scene])

                    if j < prompts_per_scene:
                        bucket[j] = name

            records = []

            for scene in sorted(reservoirs):
                for name in reservoirs[scene]:
                    try:
                        prompt = pickle.loads(
                            archive.read(name)
                        )

                        if not isinstance(prompt, str):
                            continue

                        prompt = prompt.strip()

                        if not prompt:
                            continue

                        parsed = parse_sample_id(name)

                        records.append({
                            **parsed,
                            "prompt": prompt,
                        })

                    except Exception as error:
                        print("Skipped instruction:", name, error)

            if not records:
                raise RuntimeError("No valid instruction candidates selected")
            return records


def read_positive_labels(fs, records, strict=False):
    valid = {}

    try:
        remote = fs.open(POSITIVE_ZIP, "rb")
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError(f"Missing or inaccessible grasp archive: {POSITIVE_ZIP}: {error}") from error
    with remote:
        with ZipFile(remote) as archive:

            for record in records:
                sample_id = record["sample_id"]

                member = (
                    f"grasp_label_positive/"
                    f"{sample_id}.pt"
                )

                try:
                    raw = archive.read(member)
                    count = validate_positive_grasps(raw)

                    valid[sample_id] = {
                        "raw": raw,
                        "grasp_count": count,
                    }

                except Exception as error:
                    if strict:
                        raise RuntimeError(f"Missing or invalid grasp annotation for {sample_id}: {error}") from error
                    print("Skipped label:", sample_id, error)

    return valid


def extract_images(fs, records, labels, output_dir, strict=False):
    images_dir = output_dir / "images"
    labels_dir = output_dir / "grasps"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)

    for record in records:
        if record["sample_id"] in labels:
            grouped[record["scene_id"]].append(record)

    manifest = []

    with open_split_image_zip(fs) as archive:

        for scene_id, samples in sorted(grouped.items()):
            member = f"image/{scene_id}.jpg"
            image_path = images_dir / f"{scene_id}.jpg"

            try:
                if not image_path.exists():
                    raw_image = archive.read(member)

                    with Image.open(
                        io.BytesIO(raw_image)
                    ) as image:
                        image.verify()

                    image_path.write_bytes(raw_image)

                with Image.open(image_path) as image:
                    width, height = image.size

                    if image.mode != "RGB":
                        raise ValueError(
                            f"Unexpected image mode: {image.mode}"
                        )

                for sample in samples:
                    sample_id = sample["sample_id"]

                    label_path = (
                        labels_dir / f"{sample_id}.pt"
                    )

                    label_path.write_bytes(
                        labels[sample_id]["raw"]
                    )

                    manifest.append({
                        **sample,
                        "image_path": str(
                            Path("images") / f"{scene_id}.jpg"
                        ),
                        "grasp_path": str(
                            Path("grasps") / f"{sample_id}.pt"
                        ),
                        "image_size": [width, height],
                        "grasp_count": labels[
                            sample_id
                        ]["grasp_count"],
                    })

                print(
                    f"Extracted scene {scene_id[:12]} "
                    f"({len(samples)} samples)"
                )

            except Exception as error:
                if strict:
                    raise RuntimeError(f"Missing or invalid RGB image for scene {scene_id}: {error}") from error
                print("Skipped image:", scene_id, error)

    return manifest


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--num-scenes", type=int, dest="scenes")
    parser.add_argument(
        "--prompts-per-scene",
        type=int,
        default=2,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../storage/data/subset_debug"),
    )
    parser.add_argument("--output-dir", type=Path, dest="output")
    parser.add_argument("--split-file", type=Path)

    args = parser.parse_args()

    if args.scenes <= 0 or args.prompts_per_scene <= 0:
        parser.error("Counts must be positive")

    fs = HfFileSystem()
    allowed_ids = load_split_ids(args.split_file) if args.split_file else None

    print("1. Selecting instructions...")

    records = select_instructions(
        fs,
        args.scenes,
        args.prompts_per_scene,
        args.seed,
        allowed_ids,
    )

    print("Selected instructions:", len(records))

    print("2. Reading positive grasp labels...")

    labels = read_positive_labels(fs, records, strict=allowed_ids is not None)

    print("Valid grasp labels:", len(labels))
    if not labels:
        raise RuntimeError("No matching positive grasp annotations found")

    print("3. Extracting RGB images...")

    manifest = extract_images(
        fs,
        records,
        labels,
        args.output,
        strict=allowed_ids is not None,
    )

    if not manifest:
        raise RuntimeError("No complete samples extracted")

    result = {
        "metadata": {
            "seed": args.seed,
            "requested_scenes": args.scenes,
            "max_prompts_per_scene": args.prompts_per_scene,
            "actual_scenes": len({
                item["scene_id"] for item in manifest
            }),
            "actual_samples": len(manifest),
            "instruction_source": "Grasp-Anything++",
            "image_source": "Grasp-Anything",
            **({"split_file": str(args.split_file), "matched_group_ids": len({item["sample_id"].rsplit("_", 1)[0] for item in manifest})} if args.split_file else {}),
        },
        "samples": manifest,
    }

    args.output.mkdir(parents=True, exist_ok=True)

    path = args.output / "manifest.json"

    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Saved manifest:", path)
    print("Total samples:", len(manifest))


if __name__ == "__main__":
    main()
