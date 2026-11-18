"""A2 loss, legacy compatibility and checkpoint smoke checks."""
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from filmgrasp.training.losses import TARGET_NAMES, grasp_loss
from filmgrasp.training.train import (
    _checkpoint, dropout_settings, load_checkpoint, loss_settings, parse_args,
)


class MaskedGeometryTests(unittest.TestCase):
    def maps(self, batch, mask_kind):
        pos = torch.zeros(batch, 1, 2, 3)
        if mask_kind == "one":
            pos[0, 0, 0, 0] = 1
        elif mask_kind == "mixed":
            pos[:, 0, 0, 0] = 1
            pos[:, 0, 1, 2] = 1
        elif mask_kind == "full":
            pos.fill_(1)
        targets = {"pos": pos, "cos": torch.ones_like(pos),
                   "sin": torch.zeros_like(pos), "width": torch.zeros_like(pos)}
        preds = tuple(torch.full_like(pos, 1.5, requires_grad=True) for _ in TARGET_NAMES)
        return preds, targets

    def test_formula_and_diagnostics(self):
        for batch in (1, 4):
            for kind in ("zero", "one", "mixed", "full"):
                with self.subTest(batch=batch, mask=kind):
                    preds, targets = self.maps(batch, kind)
                    legacy, b0_heads = grasp_loss(preds, targets)
                    equal, equal_heads = grasp_loss(preds, targets, "masked_geometry", 1)
                    self.assertTrue(torch.allclose(equal, legacy, atol=1e-6))
                    for head in TARGET_NAMES:
                        self.assertTrue(torch.allclose(equal_heads[head], b0_heads[head], atol=1e-6))
                    diagnostics = {}
                    total, heads = grasp_loss(preds, targets, "masked_geometry", .05, diagnostics)
                    self.assertTrue(torch.isfinite(total))
                    self.assertTrue(torch.allclose(total, sum(heads.values())))
                    for head in TARGET_NAMES[1:]:
                        expected = F.smooth_l1_loss(preds[TARGET_NAMES.index(head)], targets[head], reduction="none")
                        mask = targets["pos"] > 0
                        expected = (expected * mask).mean() + .05 * (expected * (~mask)).mean()
                        self.assertTrue(torch.allclose(heads[head], expected))
                        self.assertTrue(torch.allclose(heads[head], diagnostics[head + "_positive"] + diagnostics[head + "_background_weighted"]))
                    total.backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in preds))

    def test_gradient_ratio_and_zero_background(self):
        preds, targets = self.maps(4, "mixed")
        b0, _ = grasp_loss(preds, targets)
        b0_grads = torch.autograd.grad(b0, preds)
        b0_grad = b0_grads[1]
        a2, _ = grasp_loss(preds, targets, "masked_geometry", .05)
        a2_grads = torch.autograd.grad(a2, preds)
        a2_grad = a2_grads[1]
        zero, _ = grasp_loss(preds, targets, "masked_geometry", 0)
        zero_grad = torch.autograd.grad(zero, preds)[1]
        mask = targets["pos"] > 0
        self.assertTrue(torch.allclose(a2_grad[mask], b0_grad[mask]))
        self.assertTrue(torch.allclose(a2_grad[~mask], b0_grad[~mask] * .05))
        self.assertTrue(torch.count_nonzero(zero_grad[~mask]) == 0)
        self.assertTrue(torch.allclose(zero_grad[mask], b0_grad[mask]))
        self.assertTrue(torch.allclose(a2_grads[0], b0_grads[0]))

    def test_settings_and_shape(self):
        self.assertEqual(parse_args(["--device", "cpu"]).loss_type, "b0")
        self.assertEqual(loss_settings({}), ("b0", .05))
        for bad in ("nan", "inf", "-0.1"):
            with self.assertRaises(SystemExit):
                parse_args(["--device", "cpu", "--lambda-bg", bad])
        preds, targets = self.maps(1, "one")
        targets["cos"] = torch.zeros(1, 1, 2, 2)
        with self.assertRaisesRegex(ValueError, "cos"):
            grasp_loss(preds, targets, "masked_geometry")

    def test_checkpoint_resume_and_legacy(self):
        base = {"pretrained_sha256": "same", "backbone_init": "pretrained",
                "batch_size": 4, "lr": 1e-4, "seed": 42, "optimizer": "Adam",
                "clip_model": "ViT-B/32", "epochs": 2,
                "dropout_enabled": True, "dropout_prob": .1}
        a2 = dict(base, loss_type="masked_geometry", lambda_bg=.05)
        summary = {name: {"manifest_sha256": name} for name in ("train", "validation")}
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            checkpoint = _checkpoint(model, optimizer, 1, 0., 1., a2, summary)
            self.assertEqual((checkpoint["loss_type"], checkpoint["lambda_bg"]), ("masked_geometry", .05))
            torch.save(checkpoint, path)
            self.assertEqual(load_checkpoint(path, model, optimizer, a2, summary)["epoch"], 1)
            for changed in (dict(a2, loss_type="b0"), dict(a2, lambda_bg=.1),
                            dict(a2, dropout_prob=0., dropout_enabled=False),
                            dict(a2, backbone_init="random"),
                            dict(a2, pretrained_sha256="other")):
                with self.assertRaises(ValueError):
                    load_checkpoint(path, model, optimizer, changed, summary)
            for legacy in (dict(base), {k: v for k, v in base.items() if not k.startswith("dropout_")}):
                old = _checkpoint(model, optimizer, 1, 0., 1., legacy, summary)
                old.pop("loss_type"); old.pop("lambda_bg")
                old.pop("dropout_enabled"); old.pop("dropout_prob")
                torch.save(old, path)
                self.assertEqual(load_checkpoint(path, model, optimizer, legacy, summary)["epoch"], 1)
                with self.assertRaisesRegex(ValueError, "loss"):
                    load_checkpoint(path, model, optimizer, dict(legacy, loss_type="masked_geometry"), summary)


if __name__ == "__main__":
    unittest.main()
