"""Configuration settings for the GPT model - optimized for Python code generation."""

from typing import Dict, Any
from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration optimized for Python code."""
    vocab_size: int = 128  # Python code typically has more diverse characters
    n_embd: int = 128      # Slightly larger for code patterns
    n_head: int = 8        # More heads for complex patterns
    n_layer: int = 6       # More layers for better code understanding
    block_size: int = 128  # Longer context for code structure
    dropout: float = 0.1
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head > 0, "n_head must be positive"
        assert self.n_layer > 0, "n_layer must be positive"
        assert 0 <= self.dropout < 1, "dropout must be between 0 and 1"


@dataclass
class TrainingConfig:
    """Training configuration optimized for code generation."""
    batch_size: int = 12       # Smaller batch for longer sequences
    learning_rate: float = 3e-4  # Lower LR for stable code learning
    max_iters: int = 8000      # More iterations for complex patterns
    eval_interval: int = 200
    eval_iters: int = 100
    warmup_iters: int = 200    # Longer warmup for stability
    lr_decay_iters: int = 8000
    min_lr: float = 3e-5       # Minimum learning rate
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    weight_decay: float = 1e-1
    use_wandb: bool = False
    log_interval: int = 50
    save_interval: int = 1000
    
    def __post_init__(self):
        """Validate training configuration."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.max_iters > 0, "max_iters must be positive"


@dataclass
class DataConfig:
    """Data configuration for Python code dataset."""
    data_path: str = "python_code.txt"
    train_split: float = 0.9
    vocab_size: int = 128  # Will be updated based on actual vocabulary
    max_files: int = 10    # Number of Python files to download
    
    def __post_init__(self):
        """Validate data configuration."""
        assert 0 < self.train_split < 1, "train_split must be between 0 and 1"
        assert self.max_files > 0, "max_files must be positive"


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


@dataclass
class GenerationConfig:
    """Configuration for text generation - optimized for Python code."""
    max_new_tokens: int = 200
    temperature: float = 0.8     # Lower temperature for more structured code
    top_k: int = 50              # Top-k sampling
    top_p: float = 0.9           # Nucleus sampling
    
    # Python-specific generation settings
    stop_tokens: list = None     # Will be set to common Python stopping points
    
    def __post_init__(self):
        """Set default stop tokens for Python code."""
        if self.stop_tokens is None:
            # Common Python patterns where generation might stop
            self.stop_tokens = ['\n\nclass ', '\n\ndef ', '\n\nif __name__']


def get_config() -> Dict[str, Any]:
    """Get complete configuration optimized for Python code generation."""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "data": DataConfig(),
        "system": SystemConfig(),
        "generation": GenerationConfig()
    }


def get_small_config() -> Dict[str, Any]:
    """Get configuration for quick testing with smaller model."""
    config = get_config()
    
    # Smaller model for testing
    config["model"].n_embd = 64
    config["model"].n_head = 4
    config["model"].n_layer = 4
    config["model"].block_size = 64
    
    # Faster training
    config["training"].batch_size = 16
    config["training"].max_iters = 2000
    config["training"].eval_interval = 100
    
    # Fewer files for quick testing
    config["data"].max_files = 5
    
    return config


def get_large_config() -> Dict[str, Any]:
    """Get configuration for serious Python code generation training."""
    config = get_config()
    
    # Larger model
    config["model"].n_embd = 256
    config["model"].n_head = 16
    config["model"].n_layer = 12
    config["model"].block_size = 256
    
    # More intensive training
    config["training"].batch_size = 8  # Smaller batch for larger model
    config["training"].max_iters = 15000
    config["training"].learning_rate = 1e-4  # Lower LR for larger model
    
    # More data
    config["data"].max_files = 15
    
    return config 