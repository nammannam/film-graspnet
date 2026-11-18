# M6.3b.2 — A2 Background-Weighted Geometry Loss implementation

**Date:** 19 November 2026 (Asia/Bangkok). **Verdict: CONDITIONAL PASS** for implementation readiness. No training, official evaluation, commit, merge, push, dataset edit, or checkpoint edit was performed. CUDA is unavailable locally, so A2 CUDA checkpoint loading and standardized metrics remain untested until a checkpoint exists on a CUDA host.

## Before / after audit

| Dependency | Before, verified in current source | After |
|---|---|---|
| Loss calls | `train_one_epoch()` and `validate_one_epoch()` called `grasp_loss()` with four equal, unweighted SmoothL1 means. | Default `--loss-type b0` follows the same four calls and sum. Opt-in `masked_geometry` leaves position unchanged and uses `M=(target_pos>0)` for cosine, sine, width: `mean(SmoothL1 * M) + lambda_bg * mean(SmoothL1 * (1-M))`. All means cover the full batch and map; no positive normalization. |
| CLI/config | Dropout default 0, pretrained default absent (random initialization), seed default 42. | Adds `--loss-type {b0,masked_geometry}` (default b0), `--lambda-bg` (default 0.05, finite nonnegative). Existing Dropout and pretrained defaults remain. Both loss settings persist in run config and checkpoint metadata. |
| Resume | Existing checks covered Dropout, backbone initialization, pretrained SHA, batch size, LR, seed, optimizer, CLIP, and manifest hashes. | Rejects mismatched objective or lambda, including attempted A1-to-A2 resume. Legacy absent loss fields resolve to `b0,0.05`; existing guards remain. CSV header is checked against the selected objective. |
| Logging / selection | CSV and TensorBoard recorded four component losses and total; best checkpoint selected by validation accuracy, then validation loss. | Existing columns and their per-objective meanings remain. A2 adds six positive/weighted-background geometry columns for each split and `A2/` TensorBoard scalars. Three geometry components equal their two diagnostic terms; total remains the sum of four heads. Selection rule unchanged. Objective values across B0/A1 and A2 are not directly comparable. |
| Evaluator | Rebuilds model from saved Dropout/backbone config; `validate_one_epoch()` computes B0 loss and standard grasp metrics. | Checks saved loss metadata against run config and records training objective/weight plus `evaluation_loss_type=b0`. It continues to report the existing B0 reference loss and uses unchanged decoding and grasp metrics. Official comparison must use CUDA. |
| Data/model | `targets.py` sets cosine background to 1; FiLM/GRConvNet3 architecture and preprocessing were shared. | No change. Mask comes from ground-truth position, never prediction. Random backbone mode remains available. |

The audit source `docs/M6_3B_1_LOSS_AUDIT.md` proposed A2 with Dropout off. This task explicitly requires **Dropout p=0.1** for the A2 pilot, so the run command below uses it. This combines the A2 loss with A1 Dropout and must be interpreted accordingly; it is not an isolated loss-only comparison against B0.

## Changed files

- `src/filmgrasp/training/losses.py`: validated loss settings and opt-in A2 formula; kept the B0 `F.smooth_l1_loss(prediction, target)` branch.
- `src/filmgrasp/training/train.py`: CLI/config/checkpoint/resume guards, train/validation loss routing, A2 diagnostics in CSV and TensorBoard.
- `src/filmgrasp/evaluation/evaluate_checkpoint.py`: objective provenance checks/output only; B0 reference loss and grasp decoding unchanged.
- `scripts/smoke_a2_masked_geometry.py`: synthetic objective/gradient, edge-mask, CLI, checkpoint/resume, and legacy B0/A1 tests.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/namng/anaconda3/envs/filmgrasp39/bin/python scripts/smoke_a2_masked_geometry.py`: **4 tests passed**. Covers lambda=1 B0 equivalence, lambda=0 zero background gradient, lambda=.05 exact 5% background gradient with unchanged positives, zero/one/mixed/full masks, B=1/4, cosine background=1, finite gradients, shape/invalid-lambda rejection, save/load, resume rejection for loss/Dropout/backbone/pretrained SHA, and legacy B0/A1 reconstruction.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/namng/anaconda3/envs/filmgrasp39/bin/python scripts/smoke_dropout_ablation.py`: **3 tests passed**, including Dropout active in train and inactive in eval.
- Train and evaluator `--help`: **passed**. `git diff --check`: **passed**. Syntax/import verified by both CLI invocations and test imports. Direct `py_compile` attempted but sandbox blocked its `__pycache__` write; this was an environment write restriction, not a syntax error.
- `torch.cuda.is_available()` on this host: **False**. A2 CUDA evaluator load cannot be exercised before an A2 checkpoint is produced; no official metric was computed here.

## Compatibility matrix

| Mode | Loss | Dropout | Initialization | Checkpoint/evaluator |
|---|---|---|---|---|
| B0 | default `b0`, original four SmoothL1 heads | default 0 | explicit original pretrained path for baseline; optional random mode preserved | Legacy absent loss/Dropout metadata resolves to B0; evaluator uses B0 reference loss. |
| A1 | `b0` | explicit 0.1 | explicit original pretrained path | Existing Dropout reconstruction/guards preserved; evaluator uses B0 reference loss. |
| A2 pilot | explicit `masked_geometry`, `lambda_bg=0.05` | explicit 0.1 per this task | explicit original pretrained path | Separate run, no A1 resume; objective metadata checked; evaluator uses B0 reference loss with unchanged CUDA grasp metrics. |

## Exact Kaggle A2 launch after source is pushed

Use the **same original pretrained file** with SHA-256 `2fdb9867125eb7cbe5f05b0ee1c83fad29f157e39575b69da10868faf71349a4` and the original pilot manifests (train `7038dc3929deae9a729ed75d854e241f70729f41d22fea2b67520374c96225ff`, validation `4f93a67ab3d636489f784ca6d9c42de0ef99f7de0766ac1890cb1870fb31f7cc`). On Kaggle, replace the three local paths below with the mounted paths containing these exact files; record the source commit, environment and launch command. Do not supply `--resume`.

```bash
PYTHONPATH=src python -m filmgrasp.training.train \
  --train-root ../storage/data/train_pilot \
  --val-root ../storage/data/val_pilot \
  --pretrained checkpoints/grconvnet3_pretrained.pth \
  --output-root runs --run-name filmgrasp_kaggle_a2_masked_geometry_bg005_dropout_p01_seed42_100e \
  --device cuda --seed 42 --batch-size 4 --lr 1e-4 --epochs 100 --num-workers 0 \
  --dropout-prob 0.1 --loss-type masked_geometry --lambda-bg 0.05
```

After training, use the unchanged standardized CUDA evaluator with `--device cuda`, `--batch-size 4`, both pilot manifests and the selected `best` checkpoint. Record grasp accuracy, detection and mean max IoU. The evaluator's `total` and head losses are the B0 reference objective; A2 training/validation objective and diagnostics are in its run CSV/TensorBoard.
