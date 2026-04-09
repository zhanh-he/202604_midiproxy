from __future__ import annotations

from .ddsp_guitar_synth import DDSPGuitarSynthProxy


class DDSPGuitarProxy:
    """DDSP-Guitar Route III backend backed by DDSP-Guitar-Synth."""

    def __init__(self, cfg, device):
        self.backend = DDSPGuitarSynthProxy(cfg, device)

    def render(self, batch_data_dict, audio, vel_pred, random_state=None):
        return self.backend.render(batch_data_dict, audio, vel_pred, random_state=random_state)
