import sys
import tempfile
import unittest
from pathlib import Path

import torch


PYTORCH_DIR = Path(__file__).resolve().parents[1]
if str(PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_DIR))

from velo_model.pretrained_utils import (
    resolve_pretrained_checkpoint,
    select_prefixed_substate,
    unwrap_checkpoint_state_dict,
)


class PretrainedUtilsTest(unittest.TestCase):
    def test_resolve_pretrained_checkpoint_optional_empty_returns_none(self):
        resolved = resolve_pretrained_checkpoint("", model_label="filmunet", required=False)
        self.assertIsNone(resolved)

    def test_resolve_pretrained_checkpoint_required_empty_raises(self):
        with self.assertRaises(ValueError):
            resolve_pretrained_checkpoint("", model_label="hpt_pretrained", required=True)

    def test_resolve_pretrained_checkpoint_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            resolve_pretrained_checkpoint("/tmp/definitely_missing_checkpoint.pth", model_label="hpt_pretrained", required=True)

    def test_resolve_pretrained_checkpoint_existing_path_returns_path(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as fp:
            resolved = resolve_pretrained_checkpoint(fp.name, model_label="hpt_pretrained", required=True)
        self.assertEqual(resolved, Path(fp.name).resolve())

    def test_unwrap_checkpoint_state_dict_strips_module_prefix(self):
        state = unwrap_checkpoint_state_dict({
            "state_dict": {
                "module.layer.weight": torch.tensor([1.0]),
            }
        })
        self.assertIn("layer.weight", state)
        self.assertNotIn("module.layer.weight", state)

    def test_select_prefixed_substate_prefers_requested_prefix(self):
        state = {
            "base_adapter.model.weight": torch.tensor([1.0]),
            "base_adapter.model.bias": torch.tensor([2.0]),
            "other.weight": torch.tensor([3.0]),
        }
        substate = select_prefixed_substate(
            state,
            model_keys=["weight", "bias"],
            prefixes=("base_adapter.model.", "model.", ""),
        )
        self.assertEqual(set(substate.keys()), {"weight", "bias"})
        self.assertEqual(float(substate["weight"].item()), 1.0)
        self.assertEqual(float(substate["bias"].item()), 2.0)


if __name__ == "__main__":
    unittest.main()
