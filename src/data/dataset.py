"""Dataset handling for GPT training with Python code from GitHub."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import requests
import os
from pathlib import Path
import logging
import re

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


class PythonCodeDataLoader:
    """Enhanced data loader for Python code from GitHub."""
    
    # Curated Python repositories with clean, educational code
    PYTHON_REPOS = [
        # Core Python libraries (working URLs)
        "https://raw.githubusercontent.com/python/cpython/main/Lib/functools.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/operator.py", 
        "https://raw.githubusercontent.com/python/cpython/main/Lib/dataclasses.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/copy.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/math.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/random.py",
        
        # PyTorch modules (excellent for ML context)
        "https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/modules/linear.py",
        "https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/modules/conv.py",
        "https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/modules/activation.py",
        "https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/functional.py",
        
        # Popular libraries
        "https://raw.githubusercontent.com/numpy/numpy/main/numpy/core/numeric.py",
        "https://raw.githubusercontent.com/requests/requests/main/requests/models.py",
        "https://raw.githubusercontent.com/psf/requests/main/src/requests/api.py",
        
        # Clean utility libraries
        "https://raw.githubusercontent.com/pallets/flask/main/src/flask/app.py",
        "https://raw.githubusercontent.com/django/django/main/django/utils/functional.py",
    ]
    
    def __init__(self, data_path: str = "python_code.txt", train_split: float = 0.9, block_size: int = 32):
        """Initialize data loader.
        
        Args:
            data_path: Path to save combined Python code.
            train_split: Fraction of data to use for training.
            block_size: Maximum sequence length.
        """
        self.data_path = data_path
        self.train_split = train_split
        self.block_size = block_size
        self.tokenizer = CharacterTokenizer()
        
    def download_python_code(self, max_files: Optional[int] = None) -> None:
        """Download Python code from GitHub repositories.
        
        Args:
            max_files: Maximum number of files to download. If None, download all.
        """
        if os.path.exists(self.data_path):
            logger.info(f"Python code file already exists: {self.data_path}")
            return
        
        logger.info("Downloading Python code from GitHub repositories...")
        
        repos_to_download = self.PYTHON_REPOS[:max_files] if max_files else self.PYTHON_REPOS
        all_code = []
        
        for i, url in enumerate(repos_to_download):
            try:
                logger.info(f"Downloading {i+1}/{len(repos_to_download)}: {url.split('/')[-1]}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                code = response.text
                
                # Clean the code
                code = self._clean_python_code(code)
                
                # Add file separator with filename
                filename = url.split('/')[-1]
                separator = f"\n\n# === {filename} ===\n\n"
                all_code.append(separator + code)
                
            except requests.RequestException as e:
                logger.warning(f"Failed to download {url}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error processing {url}: {e}")
                continue
        
        if not all_code:
            raise RuntimeError("Failed to download any Python code files")
        
        # Combine all code
        combined_code = "\n".join(all_code)
        
        # Save to file
        Path(self.data_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, 'w', encoding='utf-8') as f:
            f.write(combined_code)
        
        logger.info(f"Downloaded and saved {len(all_code)} Python files to {self.data_path}")
        logger.info(f"Total characters: {len(combined_code):,}")
    
    def _clean_python_code(self, code: str) -> str:
        """Clean Python code by removing excessive comments and blank lines.
        
        Args:
            code: Raw Python code.
            
        Returns:
            Cleaned Python code.
        """
        lines = code.split('\n')
        cleaned_lines = []
        
        in_multiline_string = False
        quote_char = None
        
        for line in lines:
            stripped = line.strip()
            
            # Handle multiline strings
            if '"""' in line or "'''" in line:
                if not in_multiline_string:
                    in_multiline_string = True
                    quote_char = '"""' if '"""' in line else "'''"
                elif quote_char in line:
                    in_multiline_string = False
                    quote_char = None
            
            # Skip certain types of lines
            if (
                # Skip excessive blank lines
                (not stripped and len(cleaned_lines) > 0 and not cleaned_lines[-1].strip()) or
                # Skip very long comment blocks (but keep docstrings)
                (stripped.startswith('#') and len(stripped) > 100 and not in_multiline_string) or
                # Skip license headers
                any(keyword in stripped.lower() for keyword in ['copyright', 'license', 'author:', 'email:']) or
                # Skip import statements that are too verbose
                (stripped.startswith('from') and len(stripped) > 80)
            ):
                continue
            
            cleaned_lines.append(line)
        
        # Join and normalize whitespace
        cleaned = '\n'.join(cleaned_lines)
        
        # Remove excessive blank lines
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        
        return cleaned.strip()
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, CharacterTokenizer]:
        """Load and tokenize Python code data.
        
        Returns:
            Tuple of (train_data, val_data, tokenizer).
        """
        if not os.path.exists(self.data_path):
            logger.info("Python code file not found, downloading...")
            self.download_python_code()
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except IOError as e:
            raise RuntimeError(f"Failed to read data file: {e}")
        
        if not text.strip():
            raise ValueError("Data file is empty")
        
        logger.info(f"Loaded {len(text):,} characters of Python code")
        
        # Build tokenizer and encode text
        self.tokenizer.fit(text)
        tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        
        logger.info(f"Vocabulary size: {len(self.tokenizer)}")
        logger.info(f"Total tokens: {len(tokens):,}")
        
        # Show vocabulary preview
        vocab_preview = ''.join(self.tokenizer.chars[:50])
        logger.info(f"Vocabulary preview: {repr(vocab_preview)}")
        
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
    
    def preview_data(self, num_chars: int = 500) -> None:
        """Preview the first few characters of the dataset.
        
        Args:
            num_chars: Number of characters to preview.
        """
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                preview = f.read(num_chars)
            print("Dataset Preview:")
            print("=" * 50)
            print(preview)
            print("=" * 50)
        else:
            print("Dataset not downloaded yet. Call download_python_code() first.")


# Backward compatibility with original DataLoader name
DataLoader = PythonCodeDataLoader 