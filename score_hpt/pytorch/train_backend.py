from __future__ import annotations

import os
import sys
import time
import logging
import torch
from torch.optim import Adam, AdamW

from typing import Dict, Any, Optional, Tuple
from hydra import initialize, compose
from omegaconf import OmegaConf
import wandb

from pytorch_utils import move_data_to_device, log_velocity_rolls
from data_generator import (
    Maestro_Dataset,
    SMD_Dataset,
    MAPS_Dataset,
    FrancoisLeduc_Dataset,
    Sampler,
    EvalSampler,
    collate_fn,
)
from utilities import create_folder, create_logging, get_model_name
from losses import (
    get_audio_loss_name,
    get_loss_func,
    has_supervised_velocity_target,
    velocity_prior_loss,
)
from evaluate import SegmentEvaluator

from velo_model import build_adapter
from score_inf import build_score_inf
from score_inf.wrapper import ScoreInfWrapper
from proxy import build_proxy_objective
from proxy.naming import backend_display_name, backend_run_tag, is_diffproxy_backend, normalize_backend_type


DATASET_CLASSES = {
    "maestro": Maestro_Dataset,
    "smd": SMD_Dataset,
    "maps": MAPS_Dataset,
    "francoisleduc": FrancoisLeduc_Dataset,
    "francoisledu": FrancoisLeduc_Dataset,
}



def _method_name(method) -> str:
    return str(method or "direct").strip()



def _is_direct_method(method) -> bool:
    return _method_name(method) == "direct"



def _clean_name_part(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return ""
    return text



def _cond_suffix(cfg, method: str) -> str:
    if _is_direct_method(method):
        return ""
    conds = []
    for value in (cfg.model.input2, cfg.model.input3):
        text = _clean_name_part(value)
        if text and text not in conds:
            conds.append(text)
    return f"+{'_'.join(conds)}" if conds else ""



def _proxy_objective_name(cfg) -> str:
    backend_type = normalize_backend_type(getattr(cfg.proxy, "type", "ddsp_piano"))
    if backend_type == "sfproxy":
        return _clean_name_part(getattr(cfg.proxy.sfproxy, "loss_type", "smooth_l1")) or "smooth_l1"
    return _clean_name_part(get_audio_loss_name(cfg)) or "audio"


def _backend_suffix(cfg) -> str:
    enabled = bool(getattr(cfg.proxy, "enabled", False))
    weight = float(getattr(cfg.loss, "proxy_weight", 0.0) or 0.0)
    if not enabled or weight <= 0:
        return ""
    backend_tag = backend_run_tag(getattr(cfg.proxy, "type", "ddsp_piano"))
    backend_objective = _proxy_objective_name(cfg)
    return f"+backend_{backend_tag}+{backend_objective}"




def _train_wandb_name(cfg):
    explicit_name = _clean_name_part(getattr(cfg.wandb, "name", ""))
    if explicit_name:
        return explicit_name

    method = _method_name(cfg.score_informed.method)
    name = (
        f"train-{cfg.model.type}-{method}"
        f"{_cond_suffix(cfg, method)}{_backend_suffix(cfg)}-{cfg.feature.audio_feature}-sr{cfg.feature.sample_rate}"
    )
    comment = _clean_name_part(getattr(cfg.wandb, "comment", ""))
    if comment:
        name = f"{name}-{comment}"
    return name



def init_wandb(cfg):
    """Initialize WandB for experiment tracking if configured."""
    if not hasattr(cfg, "wandb"):
        return
    wandb.init(
        project=cfg.wandb.project,
        name=_train_wandb_name(cfg),
        config=OmegaConf.to_container(cfg, resolve=True),
    )



def _select_velocity_metrics(statistics):
    keep_keys = (
        "frame_max_error",
        "frame_max_std",
        "onset_masked_error",
        "onset_masked_std",
    )
    return {k: statistics[k] for k in keep_keys if k in statistics}



def _model_param_sizes(model: torch.nn.Module) -> Tuple[int, float, float]:
    params_count = int(sum(p.numel() for p in model.parameters()))
    return params_count, float(params_count / 1e3), float(params_count / 1e6)



def _write_training_stats(
    cfg,
    checkpoints_dir: str,
    model_name: str,
    params_count: int,
) -> None:
    stats_path = os.path.join(checkpoints_dir, "training_stats.txt")
    file_name = getattr(cfg.wandb, "name", None) if hasattr(cfg, "wandb") else None
    file_name = file_name or model_name

    condition_inputs = [cfg.model.input2, cfg.model.input3]
    condition_selected = [c for c in condition_inputs if c]
    condition_check = bool(condition_selected)
    condition_type = "+".join(condition_selected) if condition_selected else "default"
    condition_net = "N/A"

    score_cfg = getattr(cfg, "score_informed", None)
    score_method = _method_name(getattr(score_cfg, "method", "direct") if score_cfg is not None else "direct")
    train_mode = getattr(score_cfg, "train_mode", "joint") if score_cfg is not None else "joint"
    switch_iteration = getattr(score_cfg, "switch_iteration", 100000) if score_cfg is not None else 100000
    if _is_direct_method(score_method):
        condition_check = False
        condition_type = "ignored"

    proxy_enabled = bool(getattr(cfg.proxy, "enabled", False))
    backend_type = backend_display_name(getattr(cfg.proxy, "type", "ddsp_piano")) if proxy_enabled else "off"
    backend_tag = backend_run_tag(getattr(cfg.proxy, "type", "ddsp_piano")) if proxy_enabled else "off"
    backend_objective = _proxy_objective_name(cfg) if proxy_enabled else "off"
    proxy_weight = float(getattr(cfg.loss, "proxy_weight", 0.0) or 0.0)
    backend_instrument = ""
    if proxy_enabled and is_diffproxy_backend(getattr(cfg.proxy, "type", "")):
        backend_instrument = str(getattr(cfg.proxy.sfproxy, "instrument_name", "") or "").strip()

    lines = [
        f"file name           :{file_name}",
        f"dev_env             :{getattr(cfg.exp, 'dev_env', 'local')}",
        f"condition_check     :{condition_check}",
        f"condition_net       :{condition_net}",
        f"loss_type           :{getattr(cfg.loss, 'loss_type', getattr(cfg.exp, 'loss_type', ''))}",
        f"supervised_weight   :{float(getattr(cfg.loss, 'supervised_weight', 1.0) or 0.0)}",
        f"backend_enabled     :{proxy_enabled}",
        f"backend_type        :{backend_type}",
        f"backend_tag         :{backend_tag}",
        f"backend_objective   :{backend_objective}",
        f"backend_instrument  :{backend_instrument or 'off'}",
        f"backend_weight      :{proxy_weight}",
        f"backend_segment_sec :{float(getattr(cfg.proxy, 'backend_segment_seconds', getattr(cfg.proxy, 'crop_seconds', 0.0)) or 0.0)}",
        f"proxy_enabled       :{proxy_enabled}",
        f"proxy_type          :{getattr(cfg.proxy, 'type', 'off') if proxy_enabled else 'off'}",
        f"proxy_objective     :{backend_objective}",
        f"proxy_instrument    :{backend_instrument or 'off'}",
        f"proxy_weight        :{proxy_weight}",
        f"proxy_segment_sec  :{float(getattr(cfg.proxy, 'backend_segment_seconds', getattr(cfg.proxy, 'crop_seconds', 0.0)) or 0.0)}",
        f"velocity_prior_w    :{float(getattr(cfg.loss, 'velocity_prior_weight', 0.0) or 0.0)}",
        f"condition_type      :{condition_type}",
        f"batch_size          :{cfg.exp.batch_size}",
        f"hop_seconds         :{cfg.feature.hop_seconds}",
        f"segment_seconds     :{cfg.feature.segment_seconds}",
        f"frames_per_second   :{cfg.feature.frames_per_second}",
        f"feature type        :{cfg.feature.audio_feature}",
        f"score_inf_method    :{score_method}",
        f"train_mode          :{train_mode}",
        f"switch_iteration    :{switch_iteration}",
        f"params_count        :{params_count}",
        f"params_size_k       :{params_count / 1e3:.3f}",
        f"params_size_m       :{params_count / 1e6:.3f}",
    ]

    with open(stats_path, "w") as f:
        f.write("\n".join(lines))



def _select_input_conditions(cfg) -> list:
    cond_selected = []
    for key in [cfg.model.input2, cfg.model.input3]:
        if key and key not in cond_selected:
            cond_selected.append(key)
    return cond_selected



def _resolve_score_inf_conditioning(cfg, method: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], list]:
    method = _method_name(method)
    cond_selected = _select_input_conditions(cfg)
    merged = dict(params)

    if _is_direct_method(method):
        return merged, []

    if method == "note_editor":
        use_cond_feats = [cfg.model.input3] if cfg.model.input3 else []
        merged["use_cond_feats"] = use_cond_feats
        cond_keys = ["onset"] + use_cond_feats
        return merged, cond_keys

    if method in ("bilstm", "scrr", "dual_gated"):
        merged["cond_keys"] = cond_selected
        return merged, cond_selected

    return merged, cond_selected



def _required_target_rolls(loss_type: str) -> list:
    if loss_type in ("velocity_bce", "velocity_mse"):
        return ["velocity_roll", "onset_roll"]
    if loss_type == "kim_bce_l1":
        return ["velocity_roll", "onset_roll", "frame_roll"]
    raise ValueError(f"Unknown loss_type: {loss_type}")



def _resolve_train_schedule(cfg, score_cfg) -> Tuple[str, int]:
    if score_cfg is None:
        return "joint", 100000
    if isinstance(score_cfg, dict):
        mode = score_cfg.get("train_mode", "joint")
        switch_iteration = int(score_cfg.get("switch_iteration", 100000))
    else:
        mode = getattr(score_cfg, "train_mode", "joint")
        switch_iteration = int(getattr(score_cfg, "switch_iteration", 100000))
    return mode, switch_iteration



def _phase_at_iteration(train_mode: str, iteration: int, switch_iteration: int) -> str:
    if train_mode == "joint":
        return "joint"
    if iteration < switch_iteration:
        return "adapter_only"
    if train_mode == "adapter_then_score":
        return "score_only"
    if train_mode == "adapter_then_joint":
        return "joint"
    raise ValueError(f"Unknown score_informed.train_mode: {train_mode}")



def _apply_train_phase(model: ScoreInfWrapper, phase: str) -> None:
    if phase == "adapter_only":
        model.freeze_base = False
        for p in model.base_adapter.parameters():
            p.requires_grad = True
        for p in model.post.parameters():
            p.requires_grad = False
        return
    if phase == "score_only":
        model.freeze_base = True
        for p in model.base_adapter.parameters():
            p.requires_grad = False
        for p in model.post.parameters():
            p.requires_grad = True
        return
    if phase == "joint":
        model.freeze_base = False
        for p in model.base_adapter.parameters():
            p.requires_grad = True
        for p in model.post.parameters():
            p.requires_grad = True
        return
    raise ValueError(f"Unknown phase: {phase}")



def build_dataloaders(cfg):
    def get_sampler(cfg, purpose: str, split: str, is_eval: Optional[str] = None):
        sampler_mapping = {
            "train": Sampler,
            "eval": EvalSampler,
        }
        return sampler_mapping[purpose](cfg, split=split, is_eval=is_eval)

    if cfg.dataset.train_set not in DATASET_CLASSES:
        raise KeyError(f"Unknown train_set '{cfg.dataset.train_set}'. Available: {list(DATASET_CLASSES.keys())}")

    train_dataset = DATASET_CLASSES[cfg.dataset.train_set](cfg)
    train_sampler = get_sampler(cfg, purpose="train", split="train", is_eval=None)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
    )

    eval_loaders = {}
    eval_sets = list(getattr(cfg.dataset, "eval_sets", []) or [])
    for eval_name in eval_sets:
        if eval_name == "train":
            eval_loaders[eval_name] = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_sampler=get_sampler(cfg, purpose="eval", split="train", is_eval=None),
                collate_fn=collate_fn,
                num_workers=cfg.exp.num_workers,
                pin_memory=True,
            )
            continue

        if eval_name not in DATASET_CLASSES:
            raise KeyError(f"Unknown eval dataset '{eval_name}'. Available: {list(DATASET_CLASSES.keys())}")
        eval_loaders[eval_name] = torch.utils.data.DataLoader(
            dataset=DATASET_CLASSES[eval_name](cfg),
            batch_sampler=get_sampler(cfg, purpose="eval", split="test", is_eval=eval_name),
            collate_fn=collate_fn,
            num_workers=cfg.exp.num_workers,
            pin_memory=True,
        )
    return train_loader, eval_loaders



def _prepare_batch(
    batch_data_dict,
    device: torch.device,
    cond_keys: list,
    target_rolls: list,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    audio = move_data_to_device(batch_data_dict["waveform"], device)
    cond = {k: move_data_to_device(batch_data_dict[f"{k}_roll"], device) for k in cond_keys}
    batch_torch = {k: move_data_to_device(batch_data_dict[k], device) for k in target_rolls if k in batch_data_dict}
    if "has_velocity_target" in batch_data_dict:
        batch_torch["has_velocity_target"] = move_data_to_device(batch_data_dict["has_velocity_target"], device)
    return audio, cond, batch_torch



def _tensor_to_float(value) -> float:
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)



def _update_meter(meter: Dict[str, float], values: Dict[str, Any]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        meter[key] = meter.get(key, 0.0) + _tensor_to_float(value)



def _average_meter(meter: Dict[str, float], steps: int) -> Dict[str, float]:
    if steps <= 0:
        return {}
    return {key: value / float(steps) for key, value in meter.items()}



def _format_loss_line(iteration: int, values: Dict[str, Any]) -> str:
    ordered = [f"iter {iteration}"]
    key_alias = {"proxy_loss": "backend_loss"}
    for key in ("total_loss", "supervised_loss", "proxy_loss", "prior_loss"):
        if key in values:
            ordered.append(f"{key_alias.get(key, key)} {_tensor_to_float(values[key]):.6f}")
    return " ".join(ordered)



def train(cfg):
    device = torch.device("cuda") if cfg.exp.cuda and torch.cuda.is_available() else torch.device("cpu")

    model_cfg = {"type": cfg.model.type, "params": cfg.model.params}
    adapter = build_adapter(model_cfg, model=None, cfg=cfg).to(device)

    score_cfg = getattr(cfg, "score_informed", None)
    if score_cfg is None:
        method = "direct"
        score_params = {}
    else:
        if OmegaConf.is_config(score_cfg):
            score_cfg = OmegaConf.to_container(score_cfg, resolve=True)
        if isinstance(score_cfg, dict):
            method = _method_name(score_cfg.get("method", "direct") or "direct")
            score_params = score_cfg.get("params", {}) or {}
        else:
            method = _method_name(getattr(score_cfg, "method", "direct") or "direct")
            score_params = getattr(score_cfg, "params", {}) or {}

    score_params, cond_keys = _resolve_score_inf_conditioning(cfg, method, score_params)

    supervised_weight = float(getattr(cfg.loss, "supervised_weight", 1.0) or 0.0)
    target_rolls = _required_target_rolls(cfg.loss.loss_type) if supervised_weight > 0 else []
    train_mode, switch_iteration = _resolve_train_schedule(cfg, score_cfg)
    post = build_score_inf(method, score_params).to(device)
    model = ScoreInfWrapper(adapter, post, freeze_base=False).to(device)

    proxy_objective = build_proxy_objective(cfg, device)
    proxy_enabled = bool(getattr(proxy_objective, "enabled", False))
    proxy_weight = float(getattr(cfg.loss, "proxy_weight", 0.0) or 0.0)
    prior_weight = float(getattr(cfg.loss, "velocity_prior_weight", 0.0) or 0.0)

    if supervised_weight <= 0 and proxy_weight <= 0 and prior_weight <= 0:
        raise RuntimeError(
            "No active loss term. Enable at least one of: loss.supervised_weight, loss.proxy_weight, loss.velocity_prior_weight."
        )

    # Paths for results
    model_name = get_model_name(cfg)
    if not _is_direct_method(method):
        model_name = f"{model_name}+score_{method}"
    if proxy_enabled and proxy_weight > 0:
        model_name = f"{model_name}+backend_{backend_run_tag(cfg.proxy.type)}+{_proxy_objective_name(cfg)}"
    checkpoints_dir = os.path.join(cfg.exp.workspace, "checkpoints", model_name)
    logs_dir = os.path.join(cfg.exp.workspace, "logs", model_name)
    params_count, params_k, params_m = _model_param_sizes(model)

    create_folder(checkpoints_dir)
    create_folder(logs_dir)
    _write_training_stats(cfg, checkpoints_dir, model_name, params_count=params_count)
    create_logging(logs_dir, filemode="w")
    logging.info(cfg)
    logging.info(f"Using {device}.")
    logging.info(f"Model Params: {params_count} ({params_k:.3f} K, {params_m:.3f} M)")
    logging.info(
        f"Loss weights -> supervised: {supervised_weight:.4f}, backend: {proxy_weight:.4f}, prior: {prior_weight:.4f}"
    )
    logging.info(f"Differentiable supervision enabled: {proxy_enabled}")
    if proxy_enabled:
        logging.info(f"Differentiable supervision backend: {backend_display_name(cfg.proxy.type)}")
        logging.info(f"Differentiable supervision objective: {_proxy_objective_name(cfg)}")

    start_iteration = 0
    init_phase = _phase_at_iteration(train_mode, start_iteration, switch_iteration)
    _apply_train_phase(model, init_phase)

    train_loader, eval_loaders = build_dataloaders(cfg)

    # Optimizer
    optim_params = list(model.parameters())
    opt_name = str(cfg.exp.optim).lower()
    if opt_name == "adamw":
        optimizer = AdamW(optim_params, lr=cfg.exp.learnrate, weight_decay=cfg.exp.weight_decay)
    elif opt_name == "adam":
        optimizer = Adam(optim_params, lr=cfg.exp.learnrate, weight_decay=cfg.exp.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.exp.optim}")

    init_wandb(cfg)

    # GPU info
    gpu_count = torch.cuda.device_count()
    logging.info(f"Number of GPUs available: {gpu_count}")
    if gpu_count > 1:
        torch.cuda.set_device(0)
    model.to(device)

    iteration = start_iteration
    train_bgn_time = time.time()
    train_meter: Dict[str, float] = {}
    train_loss_steps = 0

    early_phase = 0
    early_step = int(early_phase * 0.1) if early_phase > 0 else 0

    evaluator = SegmentEvaluator(model, cfg) if eval_loaders else None
    loss_fn = get_loss_func(cfg=cfg) if supervised_weight > 0 else None
    current_phase = None

    for batch_data_dict in train_loader:
        phase = _phase_at_iteration(train_mode, iteration, switch_iteration)
        if phase != current_phase:
            _apply_train_phase(model, phase)
            current_phase = phase
            logging.info(f"Train phase switched to: {phase} at iteration {iteration}")

        if cfg.exp.decay:
            if iteration % cfg.exp.reduce_iteration == 0 and iteration != 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.9

        optimizer.zero_grad(set_to_none=True)
        model.train()
        audio, cond, batch_torch = _prepare_batch(batch_data_dict, device, cond_keys, target_rolls)
        out = model(audio, cond)

        total_loss = out["vel_corr"].new_tensor(0.0)
        step_stats: Dict[str, Any] = {}

        if loss_fn is not None and has_supervised_velocity_target(batch_torch):
            supervised_loss = loss_fn(cfg, out, batch_torch, cond_dict=cond)
            total_loss = total_loss + supervised_weight * supervised_loss
            step_stats["supervised_loss"] = supervised_loss.detach()

        if prior_weight > 0:
            prior_loss = velocity_prior_loss(cfg, out, batch_torch)
            total_loss = total_loss + prior_weight * prior_loss
            step_stats["prior_loss"] = prior_loss.detach()

        if proxy_enabled and proxy_weight > 0:
            proxy_stats = proxy_objective.compute(
                batch_data_dict=batch_data_dict,
                audio=audio,
                vel_pred=out["vel_corr"],
                iteration=iteration,
            )
            if "proxy_loss" in proxy_stats:
                total_loss = total_loss + proxy_weight * proxy_stats["proxy_loss"]
                step_stats["proxy_loss"] = proxy_stats["proxy_loss"].detach()
            for key, value in proxy_stats.items():
                if key == "proxy_loss":
                    continue
                if torch.is_tensor(value) and value.ndim == 0:
                    step_stats[f"proxy_{key}"] = value.detach()

        if not step_stats:
            raise RuntimeError(
                "No batch loss was produced. Check loss.supervised_weight / loss.proxy_weight and whether the dataset provides the required targets."
            )

        step_stats["total_loss"] = total_loss.detach()
        print(_format_loss_line(iteration, step_stats))
        _update_meter(train_meter, step_stats)
        train_loss_steps += 1

        log_velocity_rolls(cfg, iteration, {"velocity_output": out["vel_corr"]}, batch_torch)

        total_loss.backward()
        optimizer.step()

        do_eval = (
            (iteration < early_phase and early_step > 0 and iteration % early_step == 0)
            or (iteration >= early_phase and iteration % cfg.exp.eval_iteration == 0)
        )
        if do_eval:
            logging.info("------------------------------------")
            logging.info(f"Iteration: {iteration}/{cfg.exp.total_iteration}")
            train_fin_time = time.time()
            avg_train_stats = _average_meter(train_meter, train_loss_steps)

            if avg_train_stats:
                logging.info(f"    Train Losses: {avg_train_stats}")

            log_payload: Dict[str, Any] = {"iteration": iteration}
            for key, value in avg_train_stats.items():
                log_payload[f"train_{key}"] = value

            if evaluator is not None:
                eval_stats_all = {}
                for eval_name, loader in eval_loaders.items():
                    eval_stats = _select_velocity_metrics(evaluator.evaluate(loader))
                    eval_stats_all[eval_name] = eval_stats
                    logging.info(f"    Eval {eval_name}: {eval_stats}")
                    payload_key = "train_stat" if eval_name == "train" else f"valid_{eval_name}_stat"
                    log_payload[payload_key] = eval_stats

            wandb.log(log_payload)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            logging.info(
                "Train time: {:.3f} s, validate time: {:.3f} s".format(train_time, validate_time)
            )

            train_meter = {}
            train_loss_steps = 0
            train_bgn_time = time.time()

            checkpoint_path = os.path.join(checkpoints_dir, f"{iteration}_iterations.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Model saved to {checkpoint_path}")

        if iteration == cfg.exp.total_iteration:
            break

        iteration += 1

    wandb.finish()


if __name__ == "__main__":
    initialize(config_path="./config", job_name="train_backend", version_base=None)
    cfg = compose(config_name="config", overrides=sys.argv[1:])
    train(cfg)
