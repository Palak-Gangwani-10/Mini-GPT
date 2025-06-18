"""Improved GPT model implementation with modern PyTorch practices."""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with improved implementation."""
    
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1, block_size: int = 1024):
        """Initialize multi-head attention.
        
        Args:
            n_embd: Embedding dimension.
            n_head: Number of attention heads.
            dropout: Dropout probability.
            block_size: Maximum sequence length.
        """
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head
        
        # Combined key, query, value projection
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, C).
            
        Returns:
            Output tensor of shape (B, T, C).
        """
        B, T, C = x.size()
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # Efficient attention using Flash Attention when available
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's built-in efficient attention (requires PyTorch 2.0+)
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.attn_dropout.p if self.training else 0,
                is_causal=True
            )
        else:
            # Manual implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Improved feed-forward network."""
    
    def __init__(self, n_embd: int, dropout: float = 0.1):
        """Initialize MLP.
        
        Args:
            n_embd: Embedding dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""
    
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1, block_size: int = 1024):
        """Initialize transformer block.
        
        Args:
            n_embd: Embedding dimension.
            n_head: Number of attention heads.
            dropout: Dropout probability.
            block_size: Maximum sequence length.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    """Improved GPT model with modern architecture."""
    
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 768,
        n_head: int = 12,
        n_layer: int = 12,
        block_size: int = 1024,
        dropout: float = 0.1
    ):
        """Initialize GPT model.
        
        Args:
            vocab_size: Size of the vocabulary.
            n_embd: Embedding dimension.
            n_head: Number of attention heads.
            n_layer: Number of transformer layers.
            block_size: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        
        # Token and position embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout, block_size)
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying
        self.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Number of parameters: {n_params:,}")
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            idx: Input token indices of shape (B, T).
            targets: Target token indices of shape (B, T).
            
        Returns:
            Tuple of (logits, loss).
        """
        device = idx.device
        b, t = idx.size()
        
        if t > self.block_size:
            raise ValueError(f"Sequence length {t} exceeds block size {self.block_size}")
        
        # Token and position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.wte(idx)  # (B, T, n_embd)
        pos_emb = self.wpe(pos)  # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        # Forward through transformer blocks
        for block in self.h:
            x = block(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Generate text from the model.
        
        Args:
            idx: Starting token indices of shape (B, T).
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            
        Returns:
            Generated token indices of shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop sequence if it exceeds block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get the number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding parameters.
            
        Returns:
            Number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
            n_params -= self.wte.weight.numel()
        return n_params 