# FiLM GraspNet

## Install the project package

From this project directory, with the `filmgrasp39` Conda environment active:

```bash
python -m pip install -e .
```

The editable install makes `filmgrasp` importable from Python and keeps it
linked to the code in `src/filmgrasp`.

## Train with a randomly initialized GRConvNet3 backbone

Omit `--pretrained` to initialize the RGB GRConvNet3 backbone from scratch:

```bash
python -m filmgrasp.training.train \
  --train-root ../storage/data/train_pilot \
  --val-root ../storage/data/val_pilot \
  --epochs 100 --batch-size 4 \
  --run-name filmgrasp_scratch_pilot01
```

This still trains FiLM-GraspNet: FiLM and the GRConvNet3 backbone are trainable,
while the pretrained CLIP text encoder remains frozen. Pass
`--pretrained checkpoints/grconvnet3_pretrained.pth` explicitly for the existing
pretrained-backbone behavior. Use a separate run name for each initialization.
The evaluator reads the run configuration and needs no `--pretrained` argument
for a scratch-backbone checkpoint.

## Evaluate and optionally visualize a trusted checkpoint

```bash
python -m filmgrasp.evaluation.evaluate_checkpoint \
  --run-dir runs/filmgrasp_colab_pilot_bs4_100e --checkpoint best \
  --dataset validation=../storage/data/val_pilot \
  --pretrained checkpoints/grconvnet3_pretrained.pth \
  --device cpu --visualize --max-visualizations 8 \
  --visualization-selection random --seed 42 \
  --visualization-dir runs/checkpoint_visualizations/pilot_review
```

Evaluation always covers every sample. Visualization outputs go into the named
dataset subdirectory, with one PNG per selected sample, `predictions.csv`,
`contact_sheet.png`, and `visualization_metadata.json`. The destination dataset
directory must not exist. Selection may be `random`, `failures`, or `successes`;
each uses a deterministic seed. Without `--visualize`, evaluation uses its
existing path and does not import Matplotlib.


