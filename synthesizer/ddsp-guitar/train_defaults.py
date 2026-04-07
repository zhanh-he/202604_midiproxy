import argparse
import json
from copy import deepcopy
from pathlib import Path

DEFAULT_SAMPLE_RATE = 22050
DEFAULT_FRAME_RATE = 100
DEFAULT_SEGMENT_SECONDS = 10.0
DEFAULT_LOSS_FFT_SIZES = [128, 256, 512, 1024, 2048]


def _maybe_load_json(path: str | None):
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _merge_dict(base: dict, updates: dict):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _base_parser(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config_json', type=str, default=None,
                        help='Optional JSON file with top-level config overrides.')
    parser.add_argument('--train_prepared_data_path', type=str, default=None,
                        help='Prepared training dataset .pt path.')
    parser.add_argument('--val_prepared_data_path', type=str, default=None,
                        help='Prepared validation dataset .pt path.')
    parser.add_argument('--sample_rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help='Unified audio sample rate.')
    parser.add_argument('--frame_rate', type=int, default=DEFAULT_FRAME_RATE,
                        help='Unified feature frame rate.')
    parser.add_argument('--segment_seconds', type=float, default=DEFAULT_SEGMENT_SECONDS,
                        help='Audio segment length in seconds.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id. Set to -1 for CPU.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers.')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Override W&B project name.')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Override W&B run name.')
    parser.add_argument('--description', type=str, default=None,
                        help='Optional run description.')
    parser.add_argument('--max_epochs', type=int, default=-1,
                        help='Lightning max_epochs. Use -1 for unlimited.')
    parser.add_argument('--half_precision', action='store_true',
                        help='Run with fp16 precision.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Enable dry run mode.')
    parser.add_argument('--overfitting_test', action='store_true',
                        help='Use overfitting-test mode.')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory for ModelCheckpoint outputs.')
    parser.add_argument('--resume_checkpoint_path', type=str, default=None,
                        help='Lightning checkpoint to resume from.')
    return parser


def parse_synthesis_args():
    parser = _base_parser('Train DDSP-Guitar synthesis model with unified defaults.')
    return parser.parse_args()


def parse_control_args():
    parser = _base_parser('Train DDSP-Guitar control model with unified defaults.')
    parser.add_argument('--synthesis_model_checkpoint', type=str, default=None,
                        help='Required synthesis checkpoint for control-model training.')
    parser.add_argument('--train_synthesis', action='store_true',
                        help='Fine-tune the synthesis model jointly during control training.')
    parser.add_argument('--reinitialize_synthesis_model', action='store_true',
                        help='Reinitialize synthesis model weights before control training.')
    return parser.parse_args()


def build_synthesis_config():
    return {
        'n_voices': 6,
        'seed': 0,
        'loss_fft_sizes': list(DEFAULT_LOSS_FFT_SIZES),
        'gpu': 0,
        'dry_run': False,
        'overfitting_test': False,
        'batch_size': 3,
        'num_workers': 0,
        'learning_rate': 3e-4,
        'learning_rate_gamma': 0.99,
        'trn_split_ratio': 0.9,
        'half_precision': False,
        'model_sample_rate': DEFAULT_SAMPLE_RATE,
        'model_ft_frame_rate': DEFAULT_FRAME_RATE,
        'n_seconds': DEFAULT_SEGMENT_SECONDS,
        'pitch_median_filter_window_size': 1,
        'ir_duration': 0.25,
        'n_harmonics': 128,
        'n_noise_bands': 128,
        'one_ir_per_voice': True,
        'acoustic_features': ['hex_f0_scaled', 'hex_loudness_scaled', 'hex_periodicity', 'hex_centroid_scaled'],
        'n_stacks': 6,
        'dilations': [1, 2, 4, 8, 16],
        'kernel_size': 3,
        'causal': False,
        'activation': 'gated',
        'description': 'DDSP-Guitar synthesis unified default',
        'use_cached_preprocessed_dataset': False,
        'architecture': 'lstm',
        'hidden_size': 512,
        'voice_embedding_size': 512,
        'norm_type': None,
        'rnn_layers': 3,
        'noise_bias': -3,
        'train_prepared_data_path': './artefacts/guitarset_dataset_data_trn.pt',
        'val_prepared_data_path': './artefacts/guitarset_dataset_data_val.pt',
        'wandb_project': 'ddsp-guitar-synthesis',
        'wandb_run_name': None,
        'checkpoint_dir': './artefacts/synthesis_checkpoints',
        'max_epochs': -1,
        'resume_checkpoint_path': None,
    }


def build_control_config(n_pitches: int, n_bins_per_pitch: int):
    return {
        'n_voices': 6,
        'seed': 0,
        'batch_size': 2,
        'num_workers': 0,
        'trn_split_ratio': 0.9,
        'half_precision': False,
        'model_sample_rate': DEFAULT_SAMPLE_RATE,
        'model_ft_frame_rate': DEFAULT_FRAME_RATE,
        'n_seconds': DEFAULT_SEGMENT_SECONDS,
        'pitch_median_filter_window_size': 1,
        'synthesis_model_checkpoint': None,
        'learning_rate_gamma': 0.99,
        'discrete_inputs': [
            {'name': 'midi_pitch_scaled', 'n_bins': n_pitches * n_bins_per_pitch, 'range': [0, 1]},
            {'name': 'midi_pseudo_velocity', 'n_bins': 64, 'range': [1.0, 1.3]},
        ],
        'continuous_inputs': [],
        'classification_features': [
            {'name': 'hex_f0_scaled', 'n_bins': n_pitches * n_bins_per_pitch, 'range': [0, 1]},
        ],
        'loss_fft_sizes': list(DEFAULT_LOSS_FFT_SIZES),
        'regression_features': [
            {'name': 'hex_loudness_scaled', 'n_features': 1},
            {'name': 'hex_periodicity', 'n_features': 1},
            {'name': 'hex_centroid_scaled', 'n_features': 1},
        ],
        'use_spectral_loss': True,
        'architecture': 'lstm',
        'hidden_size': 512,
        'n_blocks': 3,
        'n_heads': 1,
        'n_rnn_layers_per_block': 3,
        'norm_type': None,
        'gpu': 0,
        'learning_rate': 1e-4,
        'quantization_type': 'linear_range',
        'structured_output': False,
        'class_smoothing_sigma': 0.00,
        'big_skip_connection': False,
        'description': 'DDSP-Guitar control unified default',
        'add_midi_pitch': False,
        'add_midi_activity': False,
        'train_synthesis': False,
        'reinitialize_synthesis_model': False,
        'end2end': False,
        'pitch_loss_weight': 1.0,
        'n_harmonics': 128,
        'n_noise_bands': 128,
        'ir_duration': 0.25,
        'with_z': False,
        'overfitting_test': False,
        'dry_run': False,
        'train_prepared_data_path': './artefacts/guitarset_dataset_data_trn.pt',
        'val_prepared_data_path': './artefacts/guitarset_dataset_data_val.pt',
        'wandb_project': 'ddsp-guitar-control',
        'wandb_run_name': None,
        'checkpoint_dir': './artefacts/control_checkpoints',
        'max_epochs': -1,
        'resume_checkpoint_path': None,
    }


def apply_common_overrides(config: dict, args):
    cfg = deepcopy(config)
    _merge_dict(cfg, _maybe_load_json(args.config_json))
    if args.train_prepared_data_path is not None:
        cfg['train_prepared_data_path'] = args.train_prepared_data_path
    if args.val_prepared_data_path is not None:
        cfg['val_prepared_data_path'] = args.val_prepared_data_path
    cfg['model_sample_rate'] = int(args.sample_rate)
    cfg['model_ft_frame_rate'] = int(args.frame_rate)
    cfg['n_seconds'] = float(args.segment_seconds)
    cfg['gpu'] = None if int(args.gpu) < 0 else int(args.gpu)
    if args.batch_size is not None:
        cfg['batch_size'] = int(args.batch_size)
    cfg['num_workers'] = int(args.num_workers)
    cfg['half_precision'] = bool(args.half_precision)
    cfg['dry_run'] = bool(args.dry_run)
    cfg['overfitting_test'] = bool(args.overfitting_test)
    cfg['max_epochs'] = int(args.max_epochs)
    if args.wandb_project is not None:
        cfg['wandb_project'] = args.wandb_project
    if args.wandb_run_name is not None:
        cfg['wandb_run_name'] = args.wandb_run_name
    if args.description is not None:
        cfg['description'] = args.description
    if args.checkpoint_dir is not None:
        cfg['checkpoint_dir'] = args.checkpoint_dir
    if args.resume_checkpoint_path is not None:
        cfg['resume_checkpoint_path'] = args.resume_checkpoint_path
    return cfg


def apply_synthesis_overrides(config: dict, args):
    return apply_common_overrides(config, args)


def apply_control_overrides(config: dict, args):
    cfg = apply_common_overrides(config, args)
    if args.synthesis_model_checkpoint is not None:
        cfg['synthesis_model_checkpoint'] = args.synthesis_model_checkpoint
    cfg['train_synthesis'] = bool(args.train_synthesis)
    cfg['reinitialize_synthesis_model'] = bool(args.reinitialize_synthesis_model)
    return cfg
