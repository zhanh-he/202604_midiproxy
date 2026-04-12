import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTORCH_DIR = PROJECT_ROOT / "pytorch"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_DIR))

from benchmarks.model_FilmUnet import FiLMUNet


class _DummyFrontend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = torch.nn.Linear(1, 1)
        self.logmel_extractor = torch.nn.Linear(1, 1)
        self.backbone = torch.nn.Linear(1, 1)


class FiLMUNetCheckpointTest(unittest.TestCase):
    def test_prepare_state_dict_preserves_local_feature_extractors(self):
        model = object.__new__(FiLMUNet)
        torch.nn.Module.__init__(model)
        model.model = _DummyFrontend()

        local_state = model.model.state_dict()
        remote_state = {
            "spectrogram_extractor.weight": torch.full_like(local_state["spectrogram_extractor.weight"], 9.0),
            "spectrogram_extractor.bias": torch.full_like(local_state["spectrogram_extractor.bias"], 9.0),
            "logmel_extractor.weight": torch.full_like(local_state["logmel_extractor.weight"], 8.0),
            "logmel_extractor.bias": torch.full_like(local_state["logmel_extractor.bias"], 8.0),
            "backbone.weight": torch.full_like(local_state["backbone.weight"], 7.0),
            "backbone.bias": torch.full_like(local_state["backbone.bias"], 7.0),
        }

        prepared = model._prepare_state_dict(remote_state)

        self.assertTrue(torch.equal(prepared["backbone.weight"], remote_state["backbone.weight"]))
        self.assertTrue(torch.equal(prepared["backbone.bias"], remote_state["backbone.bias"]))
        self.assertTrue(torch.equal(prepared["spectrogram_extractor.weight"], local_state["spectrogram_extractor.weight"]))
        self.assertTrue(torch.equal(prepared["spectrogram_extractor.bias"], local_state["spectrogram_extractor.bias"]))
        self.assertTrue(torch.equal(prepared["logmel_extractor.weight"], local_state["logmel_extractor.weight"]))
        self.assertTrue(torch.equal(prepared["logmel_extractor.bias"], local_state["logmel_extractor.bias"]))


if __name__ == "__main__":
    unittest.main()
