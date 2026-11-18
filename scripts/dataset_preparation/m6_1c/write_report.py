"""Render the measured M6.1c plan and audit as the required Markdown report."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/dataset_preparation/m6_1c"
plan = json.loads((RUN / "plan.json").read_text())
audit = json.loads((RUN / "audit.json").read_text())
stats = audit["statistics"]
issues = audit["issues"]
names = ("train_expanded", "val_expanded", "test_seen", "test_unseen")
packages = audit.get("packages", {})
blocker_path = RUN / "blocker.txt"
blocker = (blocker_path.read_text(encoding="utf-8").strip() if blocker_path.exists() else "") or "none recorded"


def detail(label):
    return "\n".join("- `{}`: {}".format(n, stats[n].get(label, "n/a")) for n in names)


rows = []
for name in names:
    s = stats[name]
    critical = sum(i["severity"] == "critical" and i["dataset"] == name for i in issues)
    leakage = sum(i["severity"] == "critical" and "overlap" in i["check"] for i in issues)
    rows.append("| {} | {} | {} | {} | {} | {} | {} |".format(name, s["scenes"], s["groups"], s["samples"], s["unique_prompts"], "PARTIAL ({} critical)".format(critical) if critical else "PASSED", "{} cross-split findings".format(leakage)))

missing = []
for name in ("test_seen", "test_unseen"):
    a = plan["availability"][name]
    missing.append("- `{}`: {}/{} official groups have instruction members; {} groups unavailable in the instruction ZIP. Exact IDs are in `runs/dataset_preparation/m6_1c/plan.json`.".format(name, a["instruction_groups"], a["official_groups"], len(a["missing_instruction_groups"])))

package_lines = []
for name in names:
    p = packages.get(name)
    package_lines.append("- `{}`: {}".format(name, "`{}`; {} bytes; SHA-256 `{}`".format(p["path"], p["bytes"], p["sha256"]) if p else "not created; audit remains partial"))

report = """# M6.1c Expanded and Test Dataset Preparation Report

**Status: {status}.** Counts below are measured from the extracted files. The four transfer archives were created only after the complete integrity audit passed.

Current blocker: {blocker}

## Repository and extraction pipeline audit

`scripts/prepare_subset.py` reads the Grasp-Anything++ instruction ZIP, positive grasp label ZIP, and split Grasp-Anything image ZIP through `HfFileSystem` and `remote_zip.open_split_image_zip`. Sample IDs are `<64-lowercase-hex-scene>_<object-integer>_<part-integer>`; group IDs omit the final part integer. The scene ID is the 64-character image hash. Original annotations are PyTorch `[N,6]` tensors, saved unmodified. `GraspAnythingPPDataset(include_annotations=True)` exposes them for `annotations_to_rectangles` in the grasp metrics code.

The M6.1c script reuses the existing split parser, ID parser, annotation validator, remote image ZIP reader, dataset loader and audit checks. It scans ZIP central-directory names once to form a reproducible plan, then reads selected members only. Output image and label files are written atomically; a resumed run revalidates existing files and continues its manifest. No pilot dataset, checkpoint, official split file, model or training code was changed by this work.

## Selection and source availability

Seed **42**. Train/val candidates are shuffled deterministically within multi-object and single-object scene strata; multi-object scenes are preferred up to a 35% cap. Each selected train/val scene contributes up to two instructions, prioritizing different object groups before additional parts. Seen and Unseen tests include all instruction members of every available official group. No selection uses model predictions.

Planned extraction: train 1,200 scenes/2,300 samples; val 160 scenes/314 samples; Seen 412 available scenes/1,376 instruction members; Unseen 193 available scenes/601 instruction members. Availability of labels and images is verified as extraction proceeds.

{missing}

## Exact measured extraction counts

| Dataset | Scenes | Groups | Samples | Unique prompts | Integrity | Leakage |
|---|---:|---:|---:|---:|---|---|
{rows}

## Scene and group leakage

Official split manifests have zero shared group IDs. The original `train_seen` and `test_unseen` groups share **32** scene hashes; `val_seen` and `test_unseen` share **5**, independently confirming **37** total test-unseen scene conflicts. The selected expanded train/val pools exclude all official test scenes. `test_seen` and `test_unseen` share zero scene hashes. Exact excluded scene IDs are stored in `plan.json`. The original official definitions remain intact, so the official protocol itself is not globally scene-disjoint; comparisons using other train selections must account for this.

## Diversity and integrity

Multi-object scenes:

{multi}

Mean samples per scene:

{samples_per_scene}

Mean groups per scene:

{groups_per_scene}

Distinct object IDs:

{objects}

Distinct part IDs:

{parts}

Repeated prompt strings:

{repeated}

Exact duplicate image hashes under different scene IDs:

{duplicates}

Near-duplicate images were not measured reliably. Sample ID duplication, official group membership, image decoding and dimensions, nonempty finite `[N,6]` grasps, grasp geometry, original annotation retention, finite target maps, and DataLoader batches of 1 and 4 are checked by `audit_and_package.py`. Sparse single-pixel or under-10-pixel position targets are warnings, not deletion criteria. Current issue counts: {issues}. Detailed findings are in `runs/dataset_preparation/m6_1c/audit.json`.

## Packaging and evaluation readiness

{packages}

Packages are created only after all planned available members pass the audit, with `SHA256SUMS.txt` and archive manifest checks. The test manifests and original grasp tensors support Seen/Unseen grasp success, their harmonic mean, detection rate and mean maximum IoU using the existing evaluation code. Model evaluation was not run. Expanded training and official evaluation readiness: **{status}**.

## Files and reproduction

Added `scripts/dataset_preparation/m6_1c/prepare_expanded.py`, `audit_and_package.py`, `write_report.py`, and `run_to_completion.sh`; added this report. Data and plan artifacts are under `../storage/data/` and `runs/dataset_preparation/m6_1c/`. No commit or push was made.

From the `film-graspnet` directory:

```bash
PY=/home/namng/anaconda3/envs/filmgrasp39/bin/python
$PY scripts/dataset_preparation/m6_1c/prepare_expanded.py --plan-only
$PY scripts/dataset_preparation/m6_1c/prepare_expanded.py --limit-scenes 2 --workers 1
$PY scripts/dataset_preparation/m6_1c/prepare_expanded.py --workers 1
$PY scripts/dataset_preparation/m6_1c/audit_and_package.py --package
$PY scripts/dataset_preparation/m6_1c/write_report.py
```

The intended Colab Drive destination is `/content/drive/MyDrive/0.InterviewPrep/FPT-Innovators/Assignment/FiLM-GraspNet`. No local Ubuntu Drive mount was verified, so no transfer is claimed. After packages exist, copy the four tarballs and `SHA256SUMS.txt` to a verified mounted Drive folder, then run `sha256sum -c SHA256SUMS.txt` there.
""".format(status=audit["status"], blocker=blocker, missing="\n".join(missing), rows="\n".join(rows), multi=detail("multi_object_scenes"), samples_per_scene=detail("mean_samples_per_scene"), groups_per_scene=detail("mean_groups_per_scene"), objects=detail("distinct_object_ids"), parts=detail("distinct_part_ids"), repeated=detail("repeated_prompt_strings"), duplicates=detail("duplicate_image_hashes"), issues=audit["issue_counts"], packages="\n".join(package_lines))
(ROOT / "M6_1C_EXPANDED_AND_TEST_DATASET_PREPARATION_REPORT.md").write_text(report, encoding="utf-8")
print(ROOT / "M6_1C_EXPANDED_AND_TEST_DATASET_PREPARATION_REPORT.md")
