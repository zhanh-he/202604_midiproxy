from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import importlib
import sys
from typing import Dict, Optional

import torch

from pytorch_utils import move_data_to_device
from .common import (
    choose_crop_bounds,
    crop_audio,
    crop_roll_time_first,
    crop_roll_time_last,
    resample_roll_bct,
    resample_roll_btp,
    resample_waveform,
    resolve_backend_segment_seconds,
)


class LegacyDDSPGuitarProxy:
    """
    Frozen DiffSynth-Guitar renderer.

    Preferred synthesizer-input batch keys:
      - synth_midi_pitch:    [B, 6, T_synth] MIDI note per string
      - synth_string_index:  [B, 6, T_synth] string index 0..5 (optional)
      - synth_midi_onsets:   [B, 6, T_synth] onset mask (optional)
      - synth_midi_activity: [B, 6, T_synth] active mask (optional)

    Legacy aliases are still accepted for backward compatibility:
      - proxy_midi_pitch
      - proxy_string_index
      - proxy_midi_onsets
      - proxy_midi_activity

    Important timing rule:
    the dataset-side synthesizer-input tensors may live at their own source frame rate.
    This wrapper crops them at that source frame rate first, then resamples them to the
    DiffSynth-Guitar model frame rate before rendering.
    """

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.src_sample_rate = int(cfg.feature.sample_rate)
        self.src_frames_per_second = float(cfg.feature.frames_per_second)
        self.segment_seconds = float(cfg.feature.segment_seconds)
        self.crop_mode = str(getattr(cfg.proxy, "crop_mode", "random"))
        self.begin_note = int(cfg.feature.begin_note)

        ddsp_cfg = getattr(cfg.proxy, 'ddsp_guitar', None)
        synth_keys_cfg = getattr(ddsp_cfg, 'synth_input_keys', None)
        if synth_keys_cfg is None:
            synth_keys_cfg = getattr(ddsp_cfg, 'score_keys', None)

        self.pitch_key = str(getattr(synth_keys_cfg, "midi_pitch", "synth_midi_pitch"))
        self.string_index_key = str(getattr(synth_keys_cfg, "string_index", "synth_string_index"))
        self.onset_key = str(getattr(synth_keys_cfg, "midi_onsets", "synth_midi_onsets"))
        self.activity_key = str(getattr(synth_keys_cfg, "midi_activity", "synth_midi_activity"))

        self.legacy_pitch_key = 'proxy_midi_pitch'
        self.legacy_string_index_key = 'proxy_string_index'
        self.legacy_onset_key = 'proxy_midi_onsets'
        self.legacy_activity_key = 'proxy_midi_activity'

        self.default_synth_input_frame_rate = float(getattr(ddsp_cfg, 'source_frame_rate', self.src_frames_per_second) or self.src_frames_per_second)
        self.batch_frame_rate_key = str(getattr(ddsp_cfg, 'batch_frame_rate_key', 'synth_input_frame_rate') or 'synth_input_frame_rate')
        self.legacy_batch_frame_rate_key = 'proxy_frame_rate'

        proxy_root = self._resolve_proxy_root()
        self.model, self.preprocess_model_inputs, self.convert_dtype = self._build_model(proxy_root)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.sample_rate = int(self.model.config["model_sample_rate"])
        self.frame_rate = int(self.model.config["model_ft_frame_rate"])
        self.native_sample_rate = self.sample_rate
        self.native_frame_rate = self.frame_rate
        self.native_segment_seconds = float(self.model.config.get("n_seconds", getattr(ddsp_cfg, 'native_segment_seconds', 8.0) or 8.0))
        self.crop_seconds = resolve_backend_segment_seconds(
            cfg,
            backend_cfg=ddsp_cfg,
            backend_default=self.native_segment_seconds,
            total_segment_seconds=self.segment_seconds,
        )

    def _resolve_proxy_root(self) -> Path:
        explicit = str(getattr(getattr(self.cfg.proxy, 'ddsp', None), 'project_root', '') or '').strip()
        if explicit:
            root = Path(explicit).expanduser().resolve()
        else:
            generic = str(getattr(self.cfg.proxy, 'project_root', '') or '').strip()
            if generic:
                root = Path(generic).expanduser().resolve()
            else:
                candidates = [
                    Path(__file__).resolve().parents[3] / 'synthesizer' / 'ddsp-guitar',
                    Path(__file__).resolve().parents[3] / 'synthesizer_ddsp' / 'ddsp-guitar',
                ]
                root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if not root.exists():
            raise FileNotFoundError(f"DDSP project root not found: {root}")
        return root

    def _build_model(self, proxy_root: Path):
        if str(proxy_root) not in sys.path:
            sys.path.insert(0, str(proxy_root))
        ControlBase = importlib.import_module("train_control").ControlBase
        preprocess_model_inputs = importlib.import_module("preprocessing").preprocess_model_inputs
        convert_dtype = importlib.import_module("util").convert_dtype

        checkpoint = str(getattr(self.cfg.proxy, "checkpoint", "") or "").strip()
        if not checkpoint:
            raise ValueError("proxy.checkpoint must point to a trained DDSP-Guitar checkpoint when proxy.enabled=true")
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"DDSP-Guitar checkpoint not found: {checkpoint_path}")
        model = ControlBase.load_from_checkpoint(str(checkpoint_path))
        return model, preprocess_model_inputs, convert_dtype

    @staticmethod
    def _candidate_batch_keys(preferred_key: str, legacy_key: str) -> list[str]:
        keys = []
        for key in (preferred_key, legacy_key):
            if not key:
                continue
            key = str(key)
            if key not in keys:
                keys.append(key)
            if key.startswith('synth_'):
                alt = 'proxy_' + key[len('synth_'):]
                if alt not in keys:
                    keys.append(alt)
            if key.startswith('proxy_'):
                alt = 'synth_' + key[len('proxy_'):]
                if alt not in keys:
                    keys.append(alt)
        return keys

    def _find_batch_key(self, batch_data_dict, preferred_key: str, legacy_key: str, required: bool) -> Optional[str]:
        for key in self._candidate_batch_keys(preferred_key, legacy_key):
            if key in batch_data_dict:
                return key
        if required:
            tried = ', '.join(self._candidate_batch_keys(preferred_key, legacy_key))
            raise KeyError(
                f"DDSP-Guitar requires synthesizer-input tensor '{preferred_key}'. "
                f"Checked batch keys: {tried}. Please add this tensor in the dataset packer."
            )
        return None

    @staticmethod
    def _scalar_from_batch_value(value, default: float) -> float:
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            flat = value.detach().cpu().reshape(-1)
            if flat.numel() == 0:
                return float(default)
            return float(flat[0].item())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == 0:
                return float(default)
            return float(value[0])
        return float(value)

    def _resolve_synth_input_frame_rate(self, batch_data_dict) -> float:
        for key in self._candidate_batch_keys(self.batch_frame_rate_key, self.legacy_batch_frame_rate_key):
            if key in batch_data_dict:
                frame_rate = self._scalar_from_batch_value(batch_data_dict[key], self.default_synth_input_frame_rate)
                if frame_rate > 0:
                    return frame_rate
        return self.default_synth_input_frame_rate

    @staticmethod
    def _derive_onsets(midi_pitch: torch.Tensor) -> torch.Tensor:
        active = (midi_pitch > 0).to(midi_pitch.dtype)
        onsets = torch.zeros_like(active)
        onsets[..., 0] = active[..., 0]
        if midi_pitch.size(-1) > 1:
            changed = midi_pitch[..., 1:] != midi_pitch[..., :-1]
            onsets[..., 1:] = ((midi_pitch[..., 1:] > 0) & changed).to(midi_pitch.dtype)
        return onsets

    def _crop_batch(self, batch_data_dict, audio: torch.Tensor, vel_pred: torch.Tensor, random_state=None):
        synth_input_frame_rate = self._resolve_synth_input_frame_rate(batch_data_dict)
        if synth_input_frame_rate <= 0:
            raise ValueError(f'Invalid DDSP-Guitar synthesizer-input frame rate: {synth_input_frame_rate}')

        pitch_batch_key = self._find_batch_key(batch_data_dict, self.pitch_key, self.legacy_pitch_key, required=True)
        string_batch_key = self._find_batch_key(batch_data_dict, self.string_index_key, self.legacy_string_index_key, required=False)
        onset_batch_key = self._find_batch_key(batch_data_dict, self.onset_key, self.legacy_onset_key, required=False)
        activity_batch_key = self._find_batch_key(batch_data_dict, self.activity_key, self.legacy_activity_key, required=False)

        midi_pitch = move_data_to_device(batch_data_dict[pitch_batch_key], self.device).float()
        string_index = move_data_to_device(batch_data_dict[string_batch_key], self.device).float() if string_batch_key else None
        midi_onsets = move_data_to_device(batch_data_dict[onset_batch_key], self.device).float() if onset_batch_key else None
        midi_activity = move_data_to_device(batch_data_dict[activity_batch_key], self.device).float() if activity_batch_key else None

        start_sec, crop_sec = choose_crop_bounds(
            total_seconds=self.segment_seconds,
            crop_seconds=self.crop_seconds,
            mode=self.crop_mode,
            random_state=random_state,
        )
        audio = crop_audio(audio, self.src_sample_rate, start_sec, crop_sec)
        vel_pred = crop_roll_time_first(vel_pred, self.src_frames_per_second, start_sec, crop_sec, include_endpoint=True)
        midi_pitch = crop_roll_time_last(midi_pitch, synth_input_frame_rate, start_sec, crop_sec, include_endpoint=False)
        if string_index is not None:
            string_index = crop_roll_time_last(string_index, synth_input_frame_rate, start_sec, crop_sec, include_endpoint=False)
        if midi_onsets is not None:
            midi_onsets = crop_roll_time_last(midi_onsets, synth_input_frame_rate, start_sec, crop_sec, include_endpoint=False)
        if midi_activity is not None:
            midi_activity = crop_roll_time_last(midi_activity, synth_input_frame_rate, start_sec, crop_sec, include_endpoint=False)
        return audio, vel_pred, midi_pitch, string_index, midi_onsets, midi_activity, synth_input_frame_rate

    def _resample_synth_inputs(
        self,
        midi_pitch: torch.Tensor,
        string_index: Optional[torch.Tensor],
        midi_onsets: Optional[torch.Tensor],
        midi_activity: Optional[torch.Tensor],
        synth_input_frame_rate: float,
    ):
        if synth_input_frame_rate == self.frame_rate:
            if string_index is not None:
                string_index = string_index.long()
            return midi_pitch, string_index, midi_onsets, midi_activity

        target_frames = max(1, int(round(midi_pitch.size(-1) * float(self.frame_rate) / float(synth_input_frame_rate))))
        midi_pitch = resample_roll_bct(midi_pitch, target_frames=target_frames, mode='nearest')
        if string_index is not None:
            string_index = resample_roll_bct(string_index, target_frames=target_frames, mode='nearest').round().long()
        if midi_onsets is not None:
            midi_onsets = (resample_roll_bct(midi_onsets, target_frames=target_frames, mode='nearest') > 0.5).to(midi_pitch.dtype)
        if midi_activity is not None:
            midi_activity = (resample_roll_bct(midi_activity, target_frames=target_frames, mode='nearest') > 0.5).to(midi_pitch.dtype)
        return midi_pitch, string_index, midi_onsets, midi_activity

    def render(self, batch_data_dict, audio: torch.Tensor, vel_pred: torch.Tensor, random_state=None) -> Dict[str, torch.Tensor]:
        audio, vel_pred, midi_pitch, string_index, midi_onsets, midi_activity, synth_input_frame_rate = self._crop_batch(
            batch_data_dict,
            audio,
            vel_pred,
            random_state=random_state,
        )

        midi_pitch, string_index, midi_onsets, midi_activity = self._resample_synth_inputs(
            midi_pitch=midi_pitch,
            string_index=string_index,
            midi_onsets=midi_onsets,
            midi_activity=midi_activity,
            synth_input_frame_rate=synth_input_frame_rate,
        )

        target_frames = midi_pitch.size(-1)
        pred_proxy = resample_roll_btp(vel_pred, target_frames, mode="nearest")
        pred_proxy = pred_proxy.unsqueeze(1).expand(-1, midi_pitch.size(1), -1, -1)

        pitch_idx = torch.clamp(midi_pitch.long() - self.begin_note, min=0, max=87)
        gathered_velocity = torch.gather(pred_proxy, dim=-1, index=pitch_idx.unsqueeze(-1)).squeeze(-1)
        active_mask = (midi_pitch > 0).to(gathered_velocity.dtype)
        if midi_activity is not None:
            active_mask = midi_activity.to(gathered_velocity.dtype)
        if midi_onsets is None:
            midi_onsets = self._derive_onsets(midi_pitch)
        else:
            midi_onsets = midi_onsets.to(gathered_velocity.dtype)
        midi_pseudo_velocity = gathered_velocity * active_mask * midi_onsets

        if string_index is None:
            num_strings = midi_pitch.size(1)
            string_index = torch.arange(num_strings, device=self.device).view(1, num_strings, 1).expand_as(midi_pitch).long()

        model_inputs = {
            "midi_pitch": midi_pitch,
            "midi_pseudo_velocity": midi_pseudo_velocity,
            "string_index": string_index,
        }
        if midi_activity is not None:
            model_inputs["midi_activity"] = midi_activity
        if midi_onsets is not None:
            model_inputs["midi_onsets"] = midi_onsets

        model_inputs = self.preprocess_model_inputs(model_inputs)
        model_inputs = self.convert_dtype(model_inputs, {torch.float32: self.model.dtype, torch.float64: self.model.dtype})
        model_inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in model_inputs.items()}

        target_audio = resample_waveform(audio, self.src_sample_rate, self.sample_rate)
        rendered = self.model(model_inputs)
        proxy_audio = self.model.render(rendered, n_samples=target_audio.size(-1))["output"]
        if proxy_audio.dim() == 3 and proxy_audio.size(1) == 1:
            proxy_audio = proxy_audio.squeeze(1)

        min_len = min(proxy_audio.size(-1), target_audio.size(-1))
        proxy_audio = proxy_audio[..., :min_len]
        target_audio = target_audio[..., :min_len]

        return {
            "rendered_audio": proxy_audio,
            "reference_audio": target_audio,
            "proxy_audio": proxy_audio,
            "target_audio": target_audio,
            "sample_rate": int(self.sample_rate),
            "stats": {
                "proxy_frames": torch.tensor(float(target_frames), device=self.device),
                "proxy_polyphony": active_mask.sum(dim=1).mean().detach(),
                "synth_input_source_fps": torch.tensor(float(synth_input_frame_rate), device=self.device),
                "renderer_sample_rate": torch.tensor(float(self.sample_rate), device=self.device),
                "renderer_frame_rate": torch.tensor(float(self.frame_rate), device=self.device),
                "renderer_segment_seconds": torch.tensor(float(self.crop_seconds), device=self.device),
                "native_segment_seconds": torch.tensor(float(self.native_segment_seconds), device=self.device),
            },
        }
