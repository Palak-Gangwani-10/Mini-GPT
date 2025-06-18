"""Training module for GPT model with modern practices."""

import os
import time
import math
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


class Trainer:
    """Modern trainer with checkpointing, logging, and evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Configuration dictionary.
            device: Device to train on.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Setup optimizer with weight decay
        self.optimizer = self._configure_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._configure_scheduler()
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=config.get('project_name', 'gpt-training'),
                config=config
            )
            wandb.watch(self.model)
        elif config.get('use_wandb', False) and not WANDB_AVAILABLE:
            logger.warning("W&B requested but not installed. Install with: pip install wandb")
    
    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """Configure optimizer with proper weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to biases and layer norm weights
                if any(nd in name for nd in ['bias', 'ln', 'layernorm']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.get('weight_decay', 0.1)},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return torch.optim.AdamW(
            optim_groups,
            lr=self.config.get('learning_rate', 1e-3),
            betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.95))
        )
    
    def _configure_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Configure learning rate scheduler with warmup and cosine decay."""
        def lr_lambda(step):
            warmup_steps = self.config.get('warmup_steps', 100)
            max_steps = self.config.get('max_steps', 5000)
            min_lr_ratio = self.config.get('min_lr_ratio', 0.1)
            
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                progress = min(progress, 1.0)
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Estimate loss on train and validation sets."""
        self.model.eval()
        
        losses = {}
        for split, data_loader in [('train', self.train_loader), ('val', self.val_loader)]:
            total_loss = 0.0
            total_tokens = 0
            num_batches = min(len(data_loader), self.config.get('eval_batches', 50))
            
            for i, (x, y) in enumerate(data_loader):
                if i >= num_batches:
                    break
                
                x, y = x.to(self.device), y.to(self.device)
                
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    _, loss = self.model(x, y)
                
                batch_size, seq_len = x.shape
                total_loss += loss.item() * batch_size * seq_len
                total_tokens += batch_size * seq_len
            
            losses[split] = total_loss / total_tokens if total_tokens > 0 else float('inf')
        
        self.model.train()
        return losses
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with val_loss: {self.best_val_loss:.4f}")
        
        # Save periodic checkpoint
        if self.step % self.config.get('save_interval', 1000) == 0:
            periodic_path = self.checkpoint_dir / f'step_{self.step}.pt'
            torch.save(checkpoint, periodic_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Resumed training from step {self.step}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        log_interval = self.config.get('log_interval', 100)
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass with mixed precision
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                logits, loss = self.model(x, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            batch_size, seq_len = x.shape
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len
            self.step += 1
            
            # Logging
            if self.step % log_interval == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
                
                logger.info(
                    f"Step {self.step:6d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Tokens/s: {total_tokens / log_interval:.0f}"
                )
                
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/avg_loss': avg_loss,
                        'train/learning_rate': current_lr,
                        'train/tokens_per_second': total_tokens / log_interval,
                        'step': self.step
                    })
                
                total_loss = 0.0
                total_tokens = 0
        
        return {'train_loss': total_loss / total_tokens if total_tokens > 0 else 0}
    
    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        max_epochs = self.config.get('max_epochs', 10)
        eval_interval = self.config.get('eval_interval', 500)
        
        start_time = time.time()
        
        try:
            for epoch in range(max_epochs):
                self.epoch = epoch
                
                # Train epoch
                epoch_metrics = self.train_epoch()
                
                # Evaluation
                if self.step % eval_interval == 0:
                    losses = self.estimate_loss()
                    
                    logger.info(
                        f"Epoch {epoch:3d} | Step {self.step:6d} | "
                        f"Train Loss: {losses['train']:.4f} | "
                        f"Val Loss: {losses['val']:.4f}"
                    )
                    
                    if self.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            'eval/train_loss': losses['train'],
                            'eval/val_loss': losses['val'],
                            'epoch': epoch,
                            'step': self.step
                        })
                    
                    # Save best model
                    is_best = losses['val'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = losses['val']
                    
                    self.save_checkpoint(is_best=is_best)
                
                # Early stopping check
                if self.config.get('early_stopping_patience'):
                    # Implement early stopping logic here
                    pass
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f} seconds")
            
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.finish()
    
    @torch.no_grad()
    def generate_sample(
        self, 
        prompt: str, 
        tokenizer, 
        max_length: int = 100,
        temperature: float = 0.8
    ) -> str:
        """Generate text sample from model.
        
        Args:
            prompt: Input prompt.
            tokenizer: Tokenizer for encoding/decoding.
            max_length: Maximum generation length.
            temperature: Sampling temperature.
            
        Returns:
            Generated text.
        """
        self.model.eval()
        
        # Encode prompt
        tokens = torch.tensor(
            tokenizer.encode(prompt), 
            dtype=torch.long, 
            device=self.device
        ).unsqueeze(0)
        
        # Generate
        generated = self.model.generate(
            tokens, 
            max_new_tokens=max_length,
            temperature=temperature
        )
        
        # Decode and return
        generated_text = tokenizer.decode(generated[0].tolist())
        return generated_text 