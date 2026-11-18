
"""Convert trusted LGD split files into standalone JSON."""

import json
import math
import pickle
import random
from pathlib import Path


SOURCE = Path(
    "../refer_gitrepo/LGD/split/grasp-anything++"
)

OUTPUT = Path("splits/grasp-anything-pp")
SEED = 42
VAL_RATIO = 0.10


def load_ids(relative_path):
    # Only load pickle files from the trusted LGD repo.
    with (SOURCE / relative_path).open("rb") as f:
        ids = pickle.load(f)

    if not isinstance(ids, list):
        raise TypeError("Expected a list of IDs")

    if not all(isinstance(x, str) for x in ids):
        raise TypeError("Expected string IDs")

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs detected")

    return set(ids)


def scene_id(group_id):
    return group_id.rsplit("_", 1)[0]


def save_json(name, ids):
    output_path = OUTPUT / f"{name}.json"
    output_path.write_text(
        json.dumps(sorted(ids), indent=2),
        encoding="utf-8",
    )
    print(f"{name}: {len(ids)} IDs")


def main():
    official_train = load_ids("train/seen.obj")
    test_seen = load_ids("test/seen.obj")
    test_unseen = load_ids("test/unseen.obj")

    # Official IDs must not overlap.
    assert official_train.isdisjoint(test_seen)
    assert official_train.isdisjoint(test_unseen)
    assert test_seen.isdisjoint(test_unseen)

    # Group by the underlying RGB scene.
    scenes = sorted({
        scene_id(x) for x in official_train
    })

    rng = random.Random(SEED)
    rng.shuffle(scenes)

    n_val = max(1, math.ceil(len(scenes) * VAL_RATIO))
    val_scenes = set(scenes[:n_val])

    train_ids = {
        x for x in official_train
        if scene_id(x) not in val_scenes
    }

    val_ids = official_train - train_ids

    assert train_ids
    assert val_ids
    assert train_ids.isdisjoint(val_ids)

    assert {
        scene_id(x) for x in train_ids
    }.isdisjoint({
        scene_id(x) for x in val_ids
    })

    assert train_ids | val_ids == official_train

    OUTPUT.mkdir(parents=True, exist_ok=True)

    save_json("train_seen", train_ids)
    save_json("val_seen", val_ids)
    save_json("test_seen", test_seen)
    save_json("test_unseen", test_unseen)

    metadata = {
        "source": "Fsoft-AIC/LGD",
        "source_dataset": "grasp-anything++",
        "seed": SEED,
        "validation_scene_ratio": VAL_RATIO,
        "official_train_ids": len(official_train),
        "official_test_seen_ids": len(test_seen),
        "official_test_unseen_ids": len(test_unseen),
        "note": (
            "Validation is derived from official training IDs. "
            "Official test splits are preserved."
        ),
    }

    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Scene-disjoint train/val: OK")
    print("Official test splits unchanged: OK")
    print("SPLIT EXPORT PASSED")


if __name__ == "__main__":
    main()
