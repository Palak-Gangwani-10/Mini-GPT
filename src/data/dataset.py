"""Dataset handling for GPT training."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import requests
import os
from pathlib import Path
import logging

from .tokenizer import CharacterTokenizer

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """PyTorch Dataset for text data."""
    
    def __init__(self, tokens: torch.Tensor, block_size: int):
        """Initialize dataset.
        
        Args:
            tokens: Tokenized text data.
            block_size: Maximum sequence length.
        """
        self.tokens = tokens
        self.block_size = block_size
        
        if len(tokens) <= block_size:
            raise ValueError(f"Dataset too small. Got {len(tokens)} tokens, need > {block_size}")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (input_sequence, target_sequence).
        """
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + self.block_size + 1]
        return x, y


class DataLoader:
    """Enhanced data loader with proper error handling."""
    
    def __init__(self, data_path: str, train_split: float = 0.9, block_size: int = 32):
        """Initialize data loader.
        
        Args:
            data_path: Path to text data file.
            train_split: Fraction of data to use for training.
            block_size: Maximum sequence length.
        """
        self.data_path = data_path
        self.train_split = train_split
        self.block_size = block_size
        self.tokenizer = CharacterTokenizer()
        
    def download_shakespeare(self, url: str = None) -> None:
        """Download Shakespeare dataset.
        
        Args:
            url: URL to download from. Uses default if None.
        """
        if url is None:
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        
        if os.path.exists(self.data_path):
            logger.info(f"Data file already exists: {self.data_path}")
            return
        
        logger.info(f"Downloading data from {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            Path(self.data_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            logger.info(f"Data downloaded successfully to {self.data_path}")
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download data: {e}")
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, CharacterTokenizer]:
        """Load and tokenize data.
        
        Returns:
            Tuple of (train_data, val_data, tokenizer).
        """
        if not os.path.exists(self.data_path):
            logger.info("Data file not found, downloading...")
            self.download_shakespeare()
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except IOError as e:
            raise RuntimeError(f"Failed to read data file: {e}")
        
        if not text.strip():
            raise ValueError("Data file is empty")
        
        logger.info(f"Loaded {len(text):,} characters")
        
        # Build tokenizer and encode text
        self.tokenizer.fit(text)
        tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        
        logger.info(f"Vocabulary size: {len(self.tokenizer)}")
        logger.info(f"Total tokens: {len(tokens):,}")
        
        # Split data
        n = int(self.train_split * len(tokens))
        train_data = tokens[:n]
        val_data = tokens[n:]
        
        logger.info(f"Train tokens: {len(train_data):,}")
        logger.info(f"Validation tokens: {len(val_data):,}")
        
        return train_data, val_data, self.tokenizer
    
    def get_dataloaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """Get PyTorch DataLoaders for training and validation.
        
        Args:
            batch_size: Batch size for data loading.
            
        Returns:
            Tuple of (train_loader, val_loader).
        """
        train_data, val_data, _ = self.load_data()
        
        train_dataset = TextDataset(train_data, self.block_size)
        val_dataset = TextDataset(val_data, self.block_size)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for simplicity
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader 