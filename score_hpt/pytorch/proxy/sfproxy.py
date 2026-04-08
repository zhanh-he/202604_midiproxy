from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from pytorch_utils import move_data_to_device
from .common import (
    align_time_dim,
    choose_crop_bounds,
    crop_audio,
    crop_roll_time_first,
    resample_waveform,
    resolve_backend_segment_seconds,
    resolve_supervision_fft_size,
    resolve_supervision_frame_rate,
    resolve_supervision_hop_size,
    resolve_supervision_sample_rate,
)


class SFProxyObjective:
    """Frozen SoundFont neural proxy for note-wise weak velocity supervision."""

    enabled = True

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.random_state = __import__('numpy').random.RandomState(cfg.exp.random_seed)
        self.warmup_iterations = int(getattr(cfg.proxy, 'warmup_iterations', 0) or 0)

        self.src_sample_rate = int(cfg.feature.sample_rate)
        self.src_frames_per_second = float(cfg.feature.frames_per_second)
        self.full_segment_seconds = float(cfg.feature.segment_seconds)
        self.begin_note = int(cfg.feature.begin_note)

        self.crop_mode = str(getattr(cfg.proxy, 'crop_mode', 'random') or 'random')
        sfproxy_cfg = getattr(cfg.proxy, 'sfproxy', None)
        native_segment_seconds = float(getattr(sfproxy_cfg, 'native_segment_seconds', getattr(sfproxy_cfg, 'segment_seconds', 2.0) or 2.0) or 2.0)
        self.crop_seconds = resolve_backend_segment_seconds(
            cfg,
            backend_cfg=sfproxy_cfg,
            backend_default=native_segment_seconds,
            total_segment_seconds=self.full_segment_seconds,
        )
        self.native_segment_seconds = native_segment_seconds

        supervision_sample_rate = resolve_supervision_sample_rate(cfg)
        supervision_frame_rate = resolve_supervision_frame_rate(cfg)
        self.sample_rate = int(getattr(cfg.proxy.sfproxy, 'sample_rate', 0) or 0)
        if self.sample_rate <= 0:
            self.sample_rate = supervision_sample_rate
        self.instrument_name = str(getattr(cfg.proxy.sfproxy, 'instrument_name', '') or '').strip()
        self.feature_name = str(getattr(cfg.proxy.sfproxy, 'feature_name', 'note_dynamics') or 'note_dynamics').strip()
        self.loss_type = str(getattr(cfg.proxy.sfproxy, 'loss_type', 'smooth_l1') or 'smooth_l1').strip().lower()
        self.loss_beta = float(getattr(cfg.proxy.sfproxy, 'loss_beta', 1.0) or 1.0)
        self.use_gt_aligned_note_events = bool(getattr(cfg.proxy.sfproxy, 'use_gt_aligned_note_events', True))

        note_builder_cfg = getattr(cfg.proxy.sfproxy, 'note_builder', None)
        self.onset_threshold = float(getattr(note_builder_cfg, 'onset_threshold', 0.5) or 0.5)
        self.frame_threshold = float(getattr(note_builder_cfg, 'frame_threshold', 0.5) or 0.5)
        self.max_notes = int(getattr(note_builder_cfg, 'max_notes', 512) or 512)
        self.min_duration_frames = int(getattr(note_builder_cfg, 'min_duration_frames', 1) or 1)

        project_root = self._resolve_project_root()
        if str(project_root / 'src') not in sys.path:
            sys.path.insert(0, str(project_root / 'src'))

        from features.dynamics import DynamicsFeatureConfig, extract_note_features_padded
        from models.note_proxy_tfm import NoteProxyTransformer

        self.extract_note_features_padded = extract_note_features_padded
        feature_cfg = getattr(cfg.proxy.sfproxy, 'feature', None)
        feature_kwargs = self._to_plain_dict(feature_cfg)
        feature_kwargs.setdefault('n_fft', 0)
        feature_kwargs.setdefault('hop', 0)
        if not int(feature_kwargs.get('n_fft') or 0):
            feature_kwargs['n_fft'] = resolve_supervision_fft_size(cfg)
        if not int(feature_kwargs.get('hop') or 0):
            feature_kwargs['hop'] = resolve_supervision_hop_size(
                cfg,
                sample_rate=self.sample_rate,
                frame_rate=supervision_frame_rate,
            )
        self.feature_cfg = DynamicsFeatureConfig(**feature_kwargs)

        model_cfg = self._to_plain_dict(getattr(cfg.proxy.sfproxy, 'model', None))
        self.model = NoteProxyTransformer(cfg=model_cfg)
        self._load_checkpoint(self.model, str(getattr(cfg.proxy, 'checkpoint', '') or '').strip())
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @staticmethod
    def _to_plain_dict(value) -> Dict:
        if value is None:
            return {}
        try:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(value):
                return OmegaConf.to_container(value, resolve=True)
        except Exception:
            pass
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, '__dict__'):
            return {k: v for k, v in vars(value).items() if not k.startswith('_')}
        return dict(value)

    def _resolve_project_root(self) -> Path:
        explicit = str(getattr(getattr(self.cfg.proxy, 'sfproxy', None), 'project_root', '') or '').strip()
        if explicit:
            root = Path(explicit).expanduser().resolve()
        else:
            generic = str(getattr(self.cfg.proxy, 'project_root', '') or '').strip()
            if generic:
                root = Path(generic).expanduser().resolve()
            else:
                root = Path(__file__).resolve().parents[3] / 'synth-proxy'
        if not root.exists():
            raise FileNotFoundError(f'SFProxy project root not found: {root}')
        return root

    @staticmethod
    def _load_checkpoint(model: torch.nn.Module, checkpoint: str) -> None:
        if not checkpoint:
            raise ValueError('proxy.checkpoint must point to a trained SFProxy checkpoint when proxy.enabled=true')
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'SFProxy checkpoint not found: {checkpoint_path}')

        state = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if isinstance(state, dict):
            prefixed = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
            if prefixed:
                state = prefixed
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f'Unexpected SFProxy checkpoint keys: {unexpected}')
        allowed_missing = {'seg_head.1.weight', 'seg_head.1.bias', 'seg_head.3.weight', 'seg_head.3.bias'}
        missing = [k for k in missing if k not in allowed_missing]
        if missing:
            raise RuntimeError(f'Missing SFProxy checkpoint keys: {missing}')

    def _crop_inputs(self, batch_data_dict, audio: torch.Tensor, vel_pred: torch.Tensor):
        onset_roll = move_data_to_device(batch_data_dict['onset_roll'], self.device).float()
        frame_roll = move_data_to_device(batch_data_dict['frame_roll'], self.device).float()
        vel_pred, onset_roll, frame_roll = align_time_dim(vel_pred, onset_roll, frame_roll)

        start_sec, crop_sec = choose_crop_bounds(
            total_seconds=self.full_segment_seconds,
            crop_seconds=self.crop_seconds,
            mode=self.crop_mode,
            random_state=self.random_state,
        )
        audio = crop_audio(audio, self.src_sample_rate, start_sec, crop_sec)
        onset_roll = crop_roll_time_first(onset_roll, self.src_frames_per_second, start_sec, crop_sec, include_endpoint=True)
        frame_roll = crop_roll_time_first(frame_roll, self.src_frames_per_second, start_sec, crop_sec, include_endpoint=True)
        vel_pred = crop_roll_time_first(vel_pred, self.src_frames_per_second, start_sec, crop_sec, include_endpoint=True)
        return audio, onset_roll, frame_roll, vel_pred, float(start_sec), float(crop_sec)

    @staticmethod
    def _normalize_note_event_batch(raw_batch) -> Optional[List[List[dict]]]:
        if raw_batch is None:
            return None
        if hasattr(raw_batch, 'tolist') and not isinstance(raw_batch, list):
            try:
                raw_batch = raw_batch.tolist()
            except Exception:
                pass
        if not isinstance(raw_batch, (list, tuple)):
            return None

        normalized: List[List[dict]] = []
        for sample in raw_batch:
            if sample is None:
                normalized.append([])
            elif isinstance(sample, list):
                normalized.append(sample)
            elif isinstance(sample, tuple):
                normalized.append(list(sample))
            else:
                try:
                    normalized.append(list(sample))
                except Exception:
                    normalized.append([])
        return normalized

    def _crop_note_events(self, note_events: List[dict], start_sec: float, crop_sec: float) -> List[dict]:
        """Crop GT aligned note events to the backend window.

        Notes are kept iff their onset lies inside the crop. This matches the
        Route IV note-wise supervision target: onset flux and early harmonic
        energy are both defined relative to the note onset, so notes that began
        before the crop are intentionally excluded instead of being rebuilt from
        frame activity.
        """
        min_dur = float(self.min_duration_frames) / max(float(self.src_frames_per_second), 1e-6)
        crop_end = float(start_sec) + float(crop_sec)
        cropped: List[dict] = []

        for event in note_events or []:
            if not isinstance(event, dict):
                continue
            midi_note = int(event.get('midi_note', -1))
            onset_time = float(event.get('onset_time', 0.0))
            offset_time = float(event.get('offset_time', onset_time + min_dur))
            if onset_time < start_sec or onset_time >= crop_end:
                continue
            onset_rel = onset_time - start_sec
            offset_rel = min(float(crop_sec), max(onset_rel + min_dur, offset_time - start_sec))
            cropped.append({
                'midi_note': midi_note,
                'onset_time': onset_rel,
                'offset_time': offset_rel,
                'velocity': int(event.get('velocity', 0)),
            })

        cropped.sort(key=lambda item: (item['onset_time'], item['midi_note']))
        if self.max_notes > 0:
            cropped = cropped[: self.max_notes]
        return cropped

    def _crop_note_events_batch(self, batch_data_dict, start_sec: float, crop_sec: float) -> Optional[List[List[dict]]]:
        if not self.use_gt_aligned_note_events:
            return None
        raw_batch = batch_data_dict.get('aligned_note_events')
        note_events_batch = self._normalize_note_event_batch(raw_batch)
        if note_events_batch is None:
            return None
        return [self._crop_note_events(sample, start_sec=start_sec, crop_sec=crop_sec) for sample in note_events_batch]

    def _pack_note_tuples(
        self,
        notes: List[Tuple[float, int, int, float, float]],
        segment_seconds: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if not notes:
            empty_pitch = torch.zeros((0,), device=self.device, dtype=torch.long)
            empty_cont = torch.zeros((0, 3), device=self.device, dtype=torch.float32)
            empty_mask = torch.zeros((0,), device=self.device, dtype=torch.bool)
            return empty_pitch, empty_cont, empty_cont.clone(), empty_mask, 0

        notes.sort(key=lambda item: (item[0], item[1]))
        if self.max_notes > 0:
            notes = notes[: self.max_notes]

        pitch = torch.tensor([item[2] for item in notes], device=self.device, dtype=torch.long)
        cont_sec = torch.tensor(
            [[item[0], item[3], item[4]] for item in notes],
            device=self.device,
            dtype=torch.float32,
        )
        cont_norm = cont_sec.clone()
        denom = max(float(segment_seconds), 1e-6)
        cont_norm[:, 0] = (cont_norm[:, 0] / denom).clamp(0.0, 1.0)
        cont_norm[:, 1] = (cont_norm[:, 1] / denom).clamp(0.0, 1.0)
        cont_norm[:, 2] = cont_norm[:, 2].clamp(0.0, 1.0)
        mask = torch.ones((pitch.numel(),), device=self.device, dtype=torch.bool)
        return pitch, cont_sec, cont_norm, mask, int(pitch.numel())

    def _extract_note_list_from_events(
        self,
        note_events: List[dict],
        vel_pred: torch.Tensor,
        segment_seconds: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Pack the GT aligned note list while replacing only velocity values.

        Timing and pitch come from the aligned note list provided by the
        dataloader. The current Score-Inf VeloEst prediction is sampled from the
        predicted velocity roll at the GT onset frame of each note. In other
        words, Route IV no longer rebuilds note identities from onset/frame
        thresholds when aligned note events are available.
        """
        fps = float(self.src_frames_per_second)
        min_dur = float(self.min_duration_frames) / max(fps, 1e-6)
        vel_np = vel_pred.detach().cpu().clamp(0.0, 1.0)
        total_frames = int(vel_np.shape[0])
        total_pitches = int(vel_np.shape[1]) if vel_np.ndim >= 2 else 0

        notes: List[Tuple[float, int, int, float, float]] = []
        for event in note_events or []:
            if not isinstance(event, dict):
                continue
            midi_note = int(event.get('midi_note', -1))
            pitch_idx = midi_note - self.begin_note
            if pitch_idx < 0 or pitch_idx >= total_pitches:
                continue

            onset_sec = float(event.get('onset_time', 0.0))
            offset_sec = float(event.get('offset_time', onset_sec + min_dur))
            dur_sec = max(min_dur, float(offset_sec - onset_sec))

            if total_frames > 0:
                onset_frame = int(round(onset_sec * fps))
                onset_frame = max(0, min(onset_frame, total_frames - 1))
                velocity_01 = float(vel_np[onset_frame, pitch_idx].item())
            else:
                velocity_01 = 0.0
            notes.append((onset_sec, pitch_idx, midi_note, dur_sec, velocity_01))

        return self._pack_note_tuples(notes, segment_seconds=segment_seconds)

    def _extract_note_list(
        self,
        onset_roll: torch.Tensor,
        frame_roll: torch.Tensor,
        vel_pred: torch.Tensor,
        segment_seconds: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Fallback conversion from aligned rolls [T, 88] into padded note tokens.

        This branch is kept only for compatibility when GT aligned note events
        are unavailable. The main Route IV path should prefer the aligned note
        list above because it removes threshold-dependent note-building errors.
        """
        fps = float(self.src_frames_per_second)
        onset_np = (onset_roll.detach().cpu() > self.onset_threshold)
        frame_np = (frame_roll.detach().cpu() > self.frame_threshold)
        vel_np = vel_pred.detach().cpu().clamp(0.0, 1.0)

        notes: List[Tuple[float, int, int, float, float]] = []
        total_frames, total_pitches = onset_np.shape

        for pitch_idx in range(total_pitches):
            onset_frames = torch.nonzero(onset_np[:, pitch_idx], as_tuple=False).view(-1).tolist()
            if not onset_frames:
                continue

            active = frame_np[:, pitch_idx]
            for idx, start_frame in enumerate(onset_frames):
                next_onset = onset_frames[idx + 1] if idx + 1 < len(onset_frames) else total_frames
                end_frame = min(total_frames, max(start_frame + self.min_duration_frames, next_onset))

                cur = start_frame
                while cur < next_onset and bool(active[cur].item()):
                    cur += 1
                if cur < next_onset:
                    end_frame = max(start_frame + self.min_duration_frames, cur)
                else:
                    end_frame = max(start_frame + self.min_duration_frames, min(total_frames, next_onset))

                onset_sec = float(start_frame) / fps
                dur_sec = max(float(self.min_duration_frames) / fps, float(end_frame - start_frame) / fps)
                velocity_01 = float(vel_np[start_frame, pitch_idx].item()) if start_frame < vel_np.shape[0] else 0.0
                notes.append((onset_sec, pitch_idx, self.begin_note + pitch_idx, dur_sec, velocity_01))

        return self._pack_note_tuples(notes, segment_seconds=segment_seconds)

    def _build_note_batch(
        self,
        onset_roll: torch.Tensor,
        frame_roll: torch.Tensor,
        vel_pred: torch.Tensor,
        segment_seconds: float,
        note_events_batch: Optional[List[List[dict]]] = None,
    ):
        batch_pitch = []
        batch_cont_sec = []
        batch_cont_norm = []
        lengths = []

        batch_size = vel_pred.size(0)
        for batch_idx in range(batch_size):
            if note_events_batch is not None:
                pitch, cont_sec, cont_norm, _mask, length = self._extract_note_list_from_events(
                    note_events=note_events_batch[batch_idx],
                    vel_pred=vel_pred[batch_idx],
                    segment_seconds=segment_seconds,
                )
            else:
                pitch, cont_sec, cont_norm, _mask, length = self._extract_note_list(
                    onset_roll=onset_roll[batch_idx],
                    frame_roll=frame_roll[batch_idx],
                    vel_pred=vel_pred[batch_idx],
                    segment_seconds=segment_seconds,
                )
            batch_pitch.append(pitch)
            batch_cont_sec.append(cont_sec)
            batch_cont_norm.append(cont_norm)
            lengths.append(length)

        max_notes = max(lengths) if lengths else 0
        if max_notes == 0:
            return None

        pitch_pad = torch.zeros((batch_size, max_notes), device=self.device, dtype=torch.long)
        cont_sec_pad = torch.zeros((batch_size, max_notes, 3), device=self.device, dtype=torch.float32)
        cont_norm_pad = torch.zeros((batch_size, max_notes, 3), device=self.device, dtype=torch.float32)
        mask_pad = torch.zeros((batch_size, max_notes), device=self.device, dtype=torch.bool)

        for batch_idx in range(batch_size):
            length = lengths[batch_idx]
            if length <= 0:
                continue
            pitch_pad[batch_idx, :length] = batch_pitch[batch_idx]
            cont_sec_pad[batch_idx, :length] = batch_cont_sec[batch_idx]
            cont_norm_pad[batch_idx, :length] = batch_cont_norm[batch_idx]
            mask_pad[batch_idx, :length] = True

        return {
            'pitch': pitch_pad,
            'cont_sec': cont_sec_pad,
            'cont_norm': cont_norm_pad,
            'mask': mask_pad,
            'lengths': lengths,
        }

    @torch.no_grad()
    def _extract_target_features(self, audio: torch.Tensor, pitch: torch.Tensor, cont_sec: torch.Tensor, mask: torch.Tensor):
        target_audio = resample_waveform(audio, self.src_sample_rate, self.sample_rate)
        feats = []
        batch_size = target_audio.size(0)
        segment_seconds = float(target_audio.size(-1)) / float(max(1, self.sample_rate))
        for batch_idx in range(batch_size):
            note_feat, _ = self.extract_note_features_padded(
                audio=target_audio[batch_idx].to(torch.float32),
                pitch=pitch[batch_idx].to(torch.long),
                cont=cont_sec[batch_idx].to(torch.float32),
                mask=mask[batch_idx].to(torch.bool),
                sr=int(self.sample_rate),
                seg_len_s=float(segment_seconds),
                cfg=self.feature_cfg,
            )
            feats.append(note_feat.to(self.device))
        return torch.stack(feats, dim=0)

    def _masked_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = mask.to(pred.dtype).unsqueeze(-1)
        denom = torch.clamp(weights.sum(), min=1.0)
        loss_name = self.loss_type.lower()

        if loss_name == 'smooth_l1':
            loss = F.smooth_l1_loss(pred * weights, target * weights, reduction='sum', beta=self.loss_beta) / denom
        elif loss_name == 'l1':
            loss = torch.abs(pred - target).mul(weights).sum() / denom
        elif loss_name in {'mse', 'l2'}:
            loss = (pred - target).pow(2).mul(weights).sum() / denom
        else:
            raise ValueError(f'Unsupported proxy.sfproxy.loss_type: {self.loss_type}')

        mae = torch.abs(pred - target).mul(weights).sum() / denom
        return loss, mae

    def compute(self, batch_data_dict, audio, vel_pred, iteration: int) -> Dict[str, torch.Tensor]:
        if iteration < self.warmup_iterations:
            return {}

        audio, onset_roll, frame_roll, vel_pred, start_sec, crop_sec = self._crop_inputs(batch_data_dict, audio, vel_pred)
        segment_seconds = float(audio.size(-1)) / float(max(1, self.src_sample_rate))
        note_events_batch = self._crop_note_events_batch(batch_data_dict, start_sec=start_sec, crop_sec=crop_sec)
        note_batch = self._build_note_batch(
            onset_roll,
            frame_roll,
            vel_pred,
            segment_seconds=segment_seconds,
            note_events_batch=note_events_batch,
        )
        if note_batch is None:
            zero = vel_pred.new_tensor(0.0)
            return {
                'proxy_loss': zero,
                'note_mae': zero,
                'note_count': zero,
            }

        target_note = self._extract_target_features(
            audio=audio,
            pitch=note_batch['pitch'],
            cont_sec=note_batch['cont_sec'],
            mask=note_batch['mask'],
        )

        pred_note, _ = self.model(
            pitch=note_batch['pitch'],
            cont=note_batch['cont_norm'],
            mask=note_batch['mask'],
        )
        proxy_loss, note_mae = self._masked_loss(pred_note, target_note, note_batch['mask'])

        note_lengths = torch.tensor(note_batch['lengths'], device=self.device, dtype=torch.float32)
        stats = {
            'proxy_loss': proxy_loss,
            'note_mae': note_mae.detach(),
            'note_count': note_lengths.mean().detach() if note_lengths.numel() else vel_pred.new_tensor(0.0),
            'renderer_segment_seconds': vel_pred.new_tensor(float(self.crop_seconds)),
            'native_segment_seconds': vel_pred.new_tensor(float(self.native_segment_seconds)),
            'renderer_sample_rate': vel_pred.new_tensor(float(self.sample_rate)),
            'renderer_frame_rate': vel_pred.new_tensor(float(self.feature_cfg.hop and (self.sample_rate / float(self.feature_cfg.hop)) or 0.0)),
        }
        return stats
