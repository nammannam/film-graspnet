# Project scripts

Only reproducibility, audit, and smoke-test utilities belong here.

- `prepare_subset.py`, `prepare_expanded.py`, and
  `export_official_splits.py`: deterministic dataset/split preparation.
- `audit_and_package.py`: expanded/test dataset audit and packaging.
- `audit_pilot_dataset.py`: offline dataset integrity audit.
- `export_grconvnet3_pretrained.py` and `compare_original_backbone.py`:
  pretrained-backbone provenance and identity checks.
- `smoke_*.py`: focused configuration, CLIP/model integration, metric,
  and loss checks.

To prepare the pilot datasets from the remote Grasp-Anything collections:

```bash
python scripts/prepare_subset.py --scenes 200 --split-file splits/grasp-anything-pp/train_seen.json --output ../storage/data/train_pilot
python scripts/prepare_subset.py --scenes 40 --split-file splits/grasp-anything-pp/val_seen.json --output ../storage/data/val_pilot
```

Run these commands from the repository root after installing the project.
The expanded/test workflow uses `python scripts/prepare_expanded.py` followed
by `python scripts/audit_and_package.py`; both share the plan under
`runs/dataset_preparation/m6_1c/`. Exporting or comparing pretrained
GR-ConvNet3 weights requires a trusted Grasp-Anything checkout at
`../refer_gitrepo/Grasp-Anything/`.

Exploratory inspection and hard-coded visualization scripts are intentionally
excluded. Checkpoint visualization is provided by
`python -m filmgrasp.evaluation.evaluate_checkpoint --visualize`.
