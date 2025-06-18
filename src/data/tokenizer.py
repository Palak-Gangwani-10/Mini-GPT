"""Character-level tokenizer for text data."""

from typing import List, Dict, Optional
import json
import os
from pathlib import Path


class CharacterTokenizer:
    """A character-level tokenizer with proper error handling."""
    
    def __init__(self, chars: Optional[List[str]] = None):
        """Initialize tokenizer with character set.
        
        Args:
            chars: List of characters to use for vocabulary.
                  If None, vocabulary will be built from data.
        """
        self.chars = chars
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.vocab_size = 0
        
        if chars is not None:
            self._build_mappings(chars)
    
    def _build_mappings(self, chars: List[str]) -> None:
        """Build character to index mappings."""
        self.chars = sorted(list(set(chars)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def fit(self, text: str) -> 'CharacterTokenizer':
        """Build vocabulary from text.
        
        Args:
            text: Input text to build vocabulary from.
            
        Returns:
            self for method chaining.
        """
        if not text:
            raise ValueError("Input text cannot be empty")
        
        chars = sorted(list(set(text)))
        self._build_mappings(chars)
        return self
    
    def encode(self, text: str) -> List[int]:
        """Encode text to list of integers.
        
        Args:
            text: Text to encode.
            
        Returns:
            List of integer tokens.
            
        Raises:
            ValueError: If tokenizer is not fitted or unknown character found.
        """
        if not self.stoi:
            raise ValueError("Tokenizer must be fitted before encoding")
        
        try:
            return [self.stoi[c] for c in text]
        except KeyError as e:
            raise ValueError(f"Unknown character found: {e}")
    
    def decode(self, tokens: List[int]) -> str:
        """Decode list of integers to text.
        
        Args:
            tokens: List of integer tokens to decode.
            
        Returns:
            Decoded text string.
            
        Raises:
            ValueError: If tokenizer is not fitted or invalid token found.
        """
        if not self.itos:
            raise ValueError("Tokenizer must be fitted before decoding")
        
        try:
            return ''.join([self.itos[i] for i in tokens])
        except KeyError as e:
            raise ValueError(f"Invalid token found: {e}")
    
    def save(self, path: str) -> None:
        """Save tokenizer to file.
        
        Args:
            path: Path to save tokenizer.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'chars': self.chars,
            'vocab_size': self.vocab_size,
            'stoi': self.stoi,
            'itos': {str(k): v for k, v in self.itos.items()}  # JSON requires string keys
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'CharacterTokenizer':
        """Load tokenizer from file.
        
        Args:
            path: Path to load tokenizer from.
            
        Returns:
            Loaded tokenizer instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.chars = data['chars']
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.stoi = data['stoi']
        tokenizer.itos = {int(k): v for k, v in data['itos'].items()}
        
        return tokenizer
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def __repr__(self) -> str:
        """String representation of tokenizer."""
        return f"CharacterTokenizer(vocab_size={self.vocab_size})" 