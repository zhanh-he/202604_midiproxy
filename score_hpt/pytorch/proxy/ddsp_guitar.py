from __future__ import annotations

from .ddsp_guitar_legacy import LegacyDDSPGuitarProxy
from .ddsp_guitar_synth import DDSPGuitarSynthProxy


class DDSPGuitarProxy:
    """Backend selector for guitar DiffSynth.

    Default implementation in this repo is now DDSP-Guitar-Synth.
    To switch back for ablation, set:

        proxy.ddsp_guitar.implementation=ddsp_guitar

    and point proxy.ddsp.project_root to synthesizer/ddsp-guitar.
    """

    def __init__(self, cfg, device):
        ddsp_cfg = getattr(cfg.proxy, 'ddsp_guitar', None)
        implementation = str(getattr(ddsp_cfg, 'implementation', 'ddsp_guitar_synth') or 'ddsp_guitar_synth').strip().lower()
        implementation = implementation.replace('-', '_').replace('/', '_')
        if implementation in {'ddsp_guitar', 'legacy', 'legacy_ddsp_guitar', 'ddsp_guitar_legacy'}:
            self.backend = LegacyDDSPGuitarProxy(cfg, device)
            self.implementation = 'ddsp_guitar'
        else:
            self.backend = DDSPGuitarSynthProxy(cfg, device)
            self.implementation = 'ddsp_guitar_synth'

    def render(self, batch_data_dict, audio, vel_pred, random_state=None):
        result = self.backend.render(batch_data_dict, audio, vel_pred, random_state=random_state)
        stats = dict(result.get('stats', {}))
        # Keep the marker human-readable in logs.
        import torch
        stats['guitar_backend_impl'] = torch.tensor(0.0 if self.implementation == 'ddsp_guitar' else 1.0, device=audio.device)
        result['stats'] = stats
        return result
