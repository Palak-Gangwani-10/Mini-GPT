"""Configuration settings for the GPT model."""

from typing import Dict, Any
from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 65
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 32
    dropout: float = 0.1
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head > 0, "n_head must be positive"
        assert self.n_layer > 0, "n_layer must be positive"
        assert 0 <= self.dropout < 1, "dropout must be between 0 and 1"


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    learning_rate: float = 1e-3
    max_iters: int = 5000
    eval_interval: int = 100
    eval_iters: int = 200
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    weight_decay: float = 1e-1
    
    def __post_init__(self):
        """Validate training configuration."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.max_iters > 0, "max_iters must be positive"


@dataclass
class DataConfig:
    """Data configuration."""
    data_path: str = "input.txt"
    train_split: float = 0.9
    vocab_size: int = 65
    
    def __post_init__(self):
        """Validate data configuration."""
        assert 0 < self.train_split < 1, "train_split must be between 0 and 1"


@dataclass
class SystemConfig:
    """System configuration."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_interval: int = 1000
    
    def __post_init__(self):
        """Set device and validate."""
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"


def get_config() -> Dict[str, Any]:
    """Get complete configuration."""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "data": DataConfig(),
        "system": SystemConfig()
    } 