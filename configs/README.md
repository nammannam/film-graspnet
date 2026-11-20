# Experiment configurations

This directory contains **training inputs**, not generated run metadata.
Start a run with:

```bash
python -m filmgrasp.training.train --config configs/pilot_b0.json
```

The trainer accepts JSON keys matching its command-line option names, converted
from kebab-case to snake_case (for example, `batch_size` maps to
`--batch-size`). Unknown keys and invalid value types are rejected. Relative
paths are resolved from the configuration file's directory. An explicitly
provided CLI option overrides the corresponding config value.

The three pilot configs encode the intended controlled comparison:

| Config | Dropout | Loss | Purpose |
|---|---:|---|---|
| `pilot_b0.json` | 0.0 | `b0` | Baseline |
| `pilot_a1_dropout.json` | 0.1 | `b0` | Dropout ablation |
| `pilot_a2_masked_geometry.json` | 0.1 | `masked_geometry` | Masked-geometry objective |

All three use the same local pilot manifests, pretrained backbone, optimizer,
learning rate, batch size, epoch budget, worker count, seed, and CUDA device.
Prepare `../storage/data/train_pilot/`, `../storage/data/val_pilot/`, and
`checkpoints/grconvnet3_pretrained.pth` before training. The JSON paths are
relative to `configs/`; pass `--device cpu` on a machine without CUDA.
Use a new `run_name` (or a CLI override) rather than overwriting an existing
run. Generated, environment-resolved metadata belongs under
`runs/<run_name>/config.json` and should not be copied back here.
