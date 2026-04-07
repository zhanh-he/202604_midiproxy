from __future__ import annotations

"""Velocity recovery evaluation for sfproxy.

This script evaluates whether a trained proxy P_i(note, v) provides
useful gradients to recover note velocities from target features.

Two settings are supported:
  1) In-domain: sampling distribution matches export/training defaults.
  2) Stress: denser polyphony and tighter IOI than training.

The test is purely synthetic and in-domain with respect to the instrument
renderer. It does NOT require any real audio dataset.
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import utils.resolvers  # noqa: F401
from typing import Any, Dict, Tuple

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from sfproxy.data.datasets_online import InstrumentSpec, RenderedInstrumentDataset
from sfproxy.data.note_samplers import make_sampler
from sfproxy.features.dynamics import DynamicsFeatureConfig, extract_note_features_padded
from sfproxy.models.lit_module import masked_mae, masked_smooth_l1
from utils.logging import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)
MIDI_VELOCITY_MAX = 127.0
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True")


def _load_model_from_lightning_ckpt(model: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    """Load weights saved by Lightning where state_dict keys are prefixed by 'model.'."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model_state = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        log.warning(f"Missing keys when loading model from ckpt: {missing}")
    if unexpected:
        log.warning(f"Unexpected keys when loading model from ckpt: {unexpected}")
    return model


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inverse of sigmoid for x in (0,1)."""
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


@dataclass
class RecoveryMetrics:
    init_mae_01: float
    init_rmse_01: float
    mae_01: float
    rmse_01: float
    loss_final: float
    steps: int


def _compute_error_stats(err: torch.Tensor) -> Tuple[float, float]:
    if not err.numel():
        return 0.0, 0.0
    return float(err.abs().mean().cpu().item()), float(torch.sqrt((err * err).mean()).cpu().item())


def _scale_stats(stats: Dict[str, float], factor: float) -> Dict[str, float]:
    return {k: float(v) * float(factor) for k, v in stats.items()}


def _cfg_summary(cfg: DictConfig) -> str:
    rec = cfg.velocity_recovery
    return "\n".join(
        [
            f"ckpt={Path(str(cfg.ckpt_path)).name} | device={cfg.device} | steps={int(rec.optim.steps)} | lr={float(rec.optim.lr):.3g} | init={rec.optim.init}",
            f"instrument={rec.instrument.name} | source={Path(str(rec.instrument.path)).name} | sr={int(rec.instrument.sr)} | seg_len={float(rec.instrument.seg_len_s):.2f}s | nmax={int(rec.instrument.nmax)}",
            f"indomain: segments={int(rec.indomain.num_segments)} | polyphony<={int(rec.sampler_base.polyphony_limit)} | chord_prob={float(rec.sampler_base.chord_prob):.2f} | ioi={list(rec.sampler_base.ioi_range)}",
            f"stress: segments={int(rec.stress.num_segments)} | polyphony<={int(rec.stress.overrides.polyphony_limit)} | chord_prob={float(rec.stress.overrides.chord_prob):.2f} | ioi={list(rec.stress.overrides.ioi_range)}",
            "metrics: velocity errors reported in MIDI [0,127]; model inputs remain normalized internally",
        ]
    )


def _suite_summary_line(name: str, suite: Dict[str, Any]) -> str:
    return (
        f"{name}: mae127 {suite['init_vel_mae']['mean']:.2f} -> {suite['vel_mae']['mean']:.2f} "
        f"({suite['mae_improvement_vs_init_pct']['mean']:.1f}% better) | "
        f"rmse127={suite['vel_rmse']['mean']:.2f} | feat_mae={suite['feat_mae_at_vhat']['mean']:.3f}"
    )


def _agg(x):
    t = torch.tensor(x, dtype=torch.float32)
    return {"mean": float(t.mean().item()), "std": float(t.std(unbiased=False).item()), "median": float(t.median().item())}


def _make_dataset(instrument: InstrumentSpec, sampler, suite_cfg: DictConfig, filters: DictConfig) -> RenderedInstrumentDataset:
    return RenderedInstrumentDataset(
        instrument=instrument,
        sampler=sampler,
        dataset_size=int(suite_cfg.dataset_size),
        seed_offset=int(suite_cfg.seed_offset),
        sr=int(instrument.sr),
        seg_len_s=float(instrument.seg_len_s),
        nmax=int(instrument.nmax),
        rms_range=(float(filters.rms_min), float(filters.rms_max)),
        peak_abs_max=float(filters.peak_abs_max),
        max_tries=int(filters.max_tries),
    )


def _annotate_bars(ax, bars, values, fmt):
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), format(value, fmt), ha="center", va="bottom", fontsize=9)


def _stats(data: Dict[str, Any], suites, key):
    return [data[s][key]["mean"] for s in suites], [data[s][key]["std"] for s in suites]


def _grouped(ax, xpos, width, labels, title, ylabel, init_vals, init_err, rec_vals, rec_err, init_color, rec_color):
    bars_init = ax.bar([x - width / 2 for x in xpos], init_vals, width, yerr=init_err, color=init_color, label="Init")
    bars_rec = ax.bar([x + width / 2 for x in xpos], rec_vals, width, yerr=rec_err, color=rec_color, label="Recovered")
    _annotate_bars(ax, bars_init, init_vals, ".1f")
    _annotate_bars(ax, bars_rec, rec_vals, ".1f")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xpos, labels)


def plot_results_file(results_json: Path, out: Path | None = None) -> Path:
    results_path = Path(results_json).expanduser().resolve()
    data = json.loads(results_path.read_text())
    out_path = Path(out).expanduser().resolve() if out else results_path.with_name(f"{Path(str(data['ckpt_path'])).stem}_velocity_recovery_summary.png")
    suites, labels, xpos, width = ["indomain", "stress"], ["In-domain", "Stress"], [0, 1], 0.34
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    fig.patch.set_facecolor("#f7f4ef")
    for ax in axes.flat:
        ax.set_facecolor("#fffdf8")
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    _grouped(axes[0, 0], xpos, width, labels, "Velocity MAE", "MIDI velocity", *_stats(data, suites, "init_vel_mae"), *_stats(data, suites, "vel_mae"), "#9aa4b2", "#ff8f00")
    axes[0, 0].legend(frameon=False)
    _grouped(axes[0, 1], xpos, width, labels, "Velocity RMSE", "MIDI velocity", *_stats(data, suites, "init_vel_rmse"), *_stats(data, suites, "vel_rmse"), "#9aa4b2", "#ff8f00")
    gain_vals, gain_err = _stats(data, suites, "mae_improvement_vs_init_pct")
    bars_gain = axes[1, 0].bar(xpos, gain_vals, width=0.5, yerr=gain_err, color="#2da44e")
    _annotate_bars(axes[1, 0], bars_gain, gain_vals, ".1f")
    axes[1, 0].set_title("MAE Improvement vs Init")
    axes[1, 0].set_ylabel("Percent")
    axes[1, 0].set_xticks(xpos, labels)
    axes[1, 0].set_ylim(0.0, max(gain_vals) * 1.25)
    feat_vals, feat_err = _stats(data, suites, "feat_mae_at_vhat")
    bars_feat = axes[1, 1].bar(xpos, feat_vals, width=0.5, yerr=feat_err, color="#1f6feb")
    _annotate_bars(axes[1, 1], bars_feat, feat_vals, ".3f")
    axes[1, 1].set_title("Feature MAE at Recovered Velocity")
    axes[1, 1].set_ylabel("Feature-space MAE")
    axes[1, 1].set_xticks(xpos, labels)
    seg_counts = ", ".join(f"{label}={int(data[s]['num_segments'])}" for label, s in zip(labels, suites))
    fig.suptitle(f"SFProxy Velocity Recovery\n{Path(str(data['ckpt_path'])).name} | {seg_counts}", fontsize=14, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_cli():
    parser = argparse.ArgumentParser(description="Plot sfproxy velocity recovery metrics.")
    parser.add_argument("--plot", dest="results_json", type=Path, required=True, help="Path to velocity_recovery_results.json")
    parser.add_argument("--out", type=Path, default=None, help="Optional output image path")
    print(plot_results_file(**vars(parser.parse_args())))


def recover_velocities_one_segment(
    *,
    model: torch.nn.Module,
    pitch: torch.Tensor,
    cont_sec: torch.Tensor,
    mask: torch.Tensor,
    target_note: torch.Tensor,
    seg_len_s: float,
    steps: int,
    lr: float,
    init: str,
    device: torch.device,
) -> Tuple[torch.Tensor, RecoveryMetrics]:
    """Recover velocities by optimizing v to match target_note features.

    Args:
        pitch: (N,) int64
        cont_sec: (N,3) float32 [onset_s, dur_s, vel_01_gt]
        mask: (N,) bool
        target_note: (N,D) float32

    Returns:
        v_hat_full: (N,) float32
        metrics: RecoveryMetrics
    """

    model.eval()

    pitch_b = pitch.to(device=device, dtype=torch.long).unsqueeze(0)  # (1,N)
    cont_sec = cont_sec.to(device=device, dtype=torch.float32)  # (N,3)
    mask_b = mask.to(device=device, dtype=torch.bool).unsqueeze(0)  # (1,N)
    target_b = target_note.to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,N,D)

    nmax = cont_sec.shape[0]
    valid_idx = torch.nonzero(mask, as_tuple=False).view(-1).to(device)

    v_gt = cont_sec[:, 2].detach().clone()
    if init == "gt_noise":
        # Small perturbation around GT, useful for debugging gradients.
        v0 = (v_gt + 0.1 * torch.randn_like(v_gt)).clamp(0.0, 1.0)
    elif init == "random":
        v0 = torch.rand((nmax,), device=device)
    else:
        v0 = torch.full((nmax,), 0.5, device=device)

    # Only optimize valid positions
    logits = _inverse_sigmoid(v0[valid_idx]).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=float(lr))

    # Precompute normalized onset/dur
    cont_norm_base = cont_sec.clone()
    cont_norm_base[:, 0] = (cont_norm_base[:, 0] / float(seg_len_s)).clamp(0.0, 1.0)
    cont_norm_base[:, 1] = (cont_norm_base[:, 1] / float(seg_len_s)).clamp(0.0, 1.0)

    init_err = (v0[valid_idx] - v_gt[valid_idx]).detach()
    init_mae_01, init_rmse_01 = _compute_error_stats(init_err)

    loss_final = 0.0
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        v_valid = torch.sigmoid(logits)
        v_full = v0.clone()
        v_full[valid_idx] = v_valid

        cont_norm = cont_norm_base.clone()
        cont_norm[:, 2] = v_full

        pred_note, _ = model(pitch=pitch_b, cont=cont_norm.unsqueeze(0), mask=mask_b)
        loss = masked_smooth_l1(pred_note, target_b, mask_b)
        loss.backward()
        opt.step()
        loss_final = float(loss.detach().cpu().item())

    v_hat = v0.clone()
    v_hat[valid_idx] = torch.sigmoid(logits.detach())

    # Metrics
    err = (v_hat[valid_idx] - v_gt[valid_idx]).detach()
    mae_01, rmse_01 = _compute_error_stats(err)
    return v_hat.detach().cpu(), RecoveryMetrics(
        init_mae_01=init_mae_01,
        init_rmse_01=init_rmse_01,
        mae_01=mae_01,
        rmse_01=rmse_01,
        loss_final=loss_final,
        steps=int(steps),
    )


def _make_stress_sampler_dict(base: Dict[str, Any], stress: Dict[str, Any]) -> Dict[str, Any]:
    """Create stress sampler config by overriding a base coverage sampler."""
    out = dict(base)
    out.update(stress)
    out["type"] = out.get("type", "coverage")
    return out


@torch.no_grad()
def _extract_targets(
    audio_1d: torch.Tensor,
    pitch: torch.Tensor,
    cont_sec: torch.Tensor,
    mask: torch.Tensor,
    sr: int,
    seg_len_s: float,
    feat_cfg: DynamicsFeatureConfig,
) -> torch.Tensor:
    feats, _ = extract_note_features_padded(
        audio=audio_1d.to(torch.float32),
        pitch=pitch.to(torch.long),
        cont=cont_sec.to(torch.float32),
        mask=mask.to(torch.bool),
        sr=int(sr),
        seg_len_s=float(seg_len_s),
        cfg=feat_cfg,
    )
    return feats.cpu()


def _run_suite(
    *,
    name: str,
    dataset: RenderedInstrumentDataset,
    model: torch.nn.Module,
    feat_cfg: DynamicsFeatureConfig,
    steps: int,
    lr: float,
    init: str,
    device: torch.device,
    num_segments: int,
) -> Dict[str, Any]:
    """Run recovery evaluation for a given dataset."""

    init_maes_01 = []
    init_rmses_01 = []
    maes_01 = []
    rmses_01 = []
    losses = []
    feat_maes = []

    seg_len_s = float(dataset.seg_len_s)
    sr = int(dataset.sr)

    pbar = tqdm(range(int(num_segments)), desc=name, dynamic_ncols=True)
    for idx in pbar:
        pitch, cont_sec, mask, audio, _ = dataset[idx]
        audio_1d = audio[0]

        target_note = _extract_targets(
            audio_1d=audio_1d,
            pitch=pitch,
            cont_sec=cont_sec,
            mask=mask,
            sr=sr,
            seg_len_s=seg_len_s,
            feat_cfg=feat_cfg,
        )

        v_hat, met = recover_velocities_one_segment(
            model=model,
            pitch=pitch,
            cont_sec=cont_sec,
            mask=mask,
            target_note=target_note,
            seg_len_s=seg_len_s,
            steps=steps,
            lr=lr,
            init=init,
            device=device,
        )

        # Feature fit quality at recovered v_hat
        cont_hat = cont_sec.clone()
        cont_hat[:, 2] = v_hat
        cont_norm = cont_hat.clone()
        cont_norm[:, 0] = (cont_norm[:, 0] / seg_len_s).clamp(0.0, 1.0)
        cont_norm[:, 1] = (cont_norm[:, 1] / seg_len_s).clamp(0.0, 1.0)

        pred_note, _ = model(
            pitch=pitch.unsqueeze(0).to(device),
            cont=cont_norm.unsqueeze(0).to(device),
            mask=mask.unsqueeze(0).to(device),
        )
        feat_mae = float(masked_mae(pred_note.cpu(), target_note.unsqueeze(0), mask.unsqueeze(0)).item())

        init_maes_01.append(met.init_mae_01)
        init_rmses_01.append(met.init_rmse_01)
        maes_01.append(met.mae_01)
        rmses_01.append(met.rmse_01)
        losses.append(met.loss_final)
        feat_maes.append(feat_mae)

        running_mae_midi = MIDI_VELOCITY_MAX * (sum(maes_01) / len(maes_01))
        running_feat_mae = sum(feat_maes) / len(feat_maes)
        running_init_mae_midi = MIDI_VELOCITY_MAX * (sum(init_maes_01) / len(init_maes_01))
        running_gain_pct = 100.0 * (running_init_mae_midi - running_mae_midi) / max(running_init_mae_midi, 1e-8)
        pbar.set_postfix(
            {
                "mae127": f"{running_mae_midi:.2f}",
                "gain": f"{running_gain_pct:.1f}%",
                "feat": f"{running_feat_mae:.3f}",
            }
        )

    init_mae_01 = _agg(init_maes_01)
    init_rmse_01 = _agg(init_rmses_01)
    vel_mae_01 = _agg(maes_01)
    vel_rmse_01 = _agg(rmses_01)
    improvement_pct = [
        100.0 * (init_mae - mae) / max(init_mae, 1e-8) for init_mae, mae in zip(init_maes_01, maes_01)
    ]

    return {
        "name": name,
        "num_segments": int(num_segments),
        "velocity_unit": "midi_0_127",
        "init_vel_mae": _scale_stats(init_mae_01, MIDI_VELOCITY_MAX),
        "init_vel_rmse": _scale_stats(init_rmse_01, MIDI_VELOCITY_MAX),
        "vel_mae": _scale_stats(vel_mae_01, MIDI_VELOCITY_MAX),
        "vel_rmse": _scale_stats(vel_rmse_01, MIDI_VELOCITY_MAX),
        "mae_improvement_vs_init_pct": _agg(improvement_pct),
        "init_vel_mae_01": init_mae_01,
        "init_vel_rmse_01": init_rmse_01,
        "vel_mae_01": vel_mae_01,
        "vel_rmse_01": vel_rmse_01,
        "opt_loss_final": _agg(losses),
        "feat_mae_at_vhat": _agg(feat_maes),
        "steps": int(steps),
        "lr": float(lr),
        "init": str(init),
    }


def run_eval_velocity_recovery(cfg: DictConfig) -> None:
    log.info("Running sfproxy velocity recovery test")
    log.info(_cfg_summary(cfg))

    device = torch.device(str(cfg.device))
    rec = cfg.velocity_recovery

    instrument = InstrumentSpec(**OmegaConf.to_container(rec.instrument, resolve=True))
    feat_cfg = DynamicsFeatureConfig(**OmegaConf.to_container(rec.feature, resolve=True))

    base_sampler_dict = OmegaConf.to_container(rec.sampler_base, resolve=True)
    datasets = {
        "indomain": _make_dataset(instrument, make_sampler(dict(base_sampler_dict)), rec.indomain, rec.filters),
        "stress": _make_dataset(
            instrument,
            make_sampler(_make_stress_sampler_dict(dict(base_sampler_dict), OmegaConf.to_container(rec.stress.overrides, resolve=True))),
            rec.stress,
            rec.filters,
        ),
    }

    model = instantiate(cfg.model)
    model = _load_model_from_lightning_ckpt(model, str(cfg.ckpt_path))
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    results = {
        "instrument": OmegaConf.to_container(rec.instrument, resolve=True),
        "ckpt_path": str(cfg.ckpt_path),
        "feature": OmegaConf.to_container(rec.feature, resolve=True),
        "optim": OmegaConf.to_container(rec.optim, resolve=True),
    }
    for name, dataset in datasets.items():
        suite_cfg = rec[name]
        results[name] = _run_suite(
            name=name,
            dataset=dataset,
            model=model,
            feat_cfg=feat_cfg,
            steps=int(rec.optim.steps),
            lr=float(rec.optim.lr),
            init=str(rec.optim.init),
            device=device,
            num_segments=int(suite_cfg.num_segments),
        )

    out_path = Path.cwd() / "velocity_recovery_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_path = plot_results_file(out_path)
    log.info(
        "\n".join(
            [
                _suite_summary_line("indomain", results["indomain"]),
                _suite_summary_line("stress", results["stress"]),
                f"saved json: {out_path}",
                f"saved plot: {plot_path}",
            ]
        )
    )


@hydra.main(version_base=None, config_path="../../configs/sfproxy", config_name="eval")
def main(cfg: DictConfig) -> None:
    run_eval_velocity_recovery(cfg)


if __name__ == "__main__":
    _plot_cli() if len(sys.argv) > 1 and sys.argv[1] == "--plot" else main()
