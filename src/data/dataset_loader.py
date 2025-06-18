"""Dataset loader for various text datasets suitable for character-level GPT."""

import requests
import os
from pathlib import Path
from typing import Optional, List, Dict
import logging
import re

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loader for various text datasets."""
    
    DATASETS = {
        # Classic Literature
        "shakespeare": {
            "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            "description": "Tiny Shakespeare dataset (original)",
            "size": "~1MB"
        },
        "alice_wonderland": {
            "url": "https://www.gutenberg.org/files/11/11-0.txt",
            "description": "Alice's Adventures in Wonderland",
            "size": "~150KB"
        },
        "pride_prejudice": {
            "url": "https://www.gutenberg.org/files/1342/1342-0.txt",
            "description": "Pride and Prejudice by Jane Austen",
            "size": "~700KB"
        },
        "frankenstein": {
            "url": "https://www.gutenberg.org/files/84/84-0.txt",
            "description": "Frankenstein by Mary Shelley",
            "size": "~400KB"
        },
        "edgar_allan_poe": {
            "url": "https://www.gutenberg.org/files/2147/2147-0.txt",
            "description": "Complete Works of Edgar Allan Poe",
            "size": "~1.2MB"
        },
        
        # Code Datasets
        "python_stdlib": {
            "urls": [
                "https://raw.githubusercontent.com/python/cpython/main/Lib/collections.py",
                "https://raw.githubusercontent.com/python/cpython/main/Lib/itertools.py",
                "https://raw.githubusercontent.com/python/cpython/main/Lib/functools.py",
                "https://raw.githubusercontent.com/python/cpython/main/Lib/operator.py",
            ],
            "description": "Python standard library modules",
            "size": "~500KB"
        },
        "pytorch_code": {
            "urls": [
                "https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/modules/linear.py",
                "https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/modules/conv.py",
                "https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/functional.py",
            ],
            "description": "PyTorch neural network modules",
            "size": "~800KB"
        },
        
        # Poetry
        "emily_dickinson": {
            "url": "https://www.gutenberg.org/files/12242/12242-0.txt",
            "description": "Poems by Emily Dickinson",
            "size": "~200KB"
        },
        "robert_frost": {
            "url": "https://www.gutenberg.org/files/59824/59824-0.txt",
            "description": "Poems by Robert Frost",
            "size": "~300KB"
        },
        
        # Philosophy & Science
        "origin_species": {
            "url": "https://www.gutenberg.org/files/1228/1228-0.txt",
            "description": "On the Origin of Species by Darwin",
            "size": "~1.5MB"
        },
        "relativity": {
            "url": "https://www.gutenberg.org/files/30155/30155-0.txt",
            "description": "Relativity by Einstein",
            "size": "~400KB"
        },
    }
    
    def __init__(self, data_dir: str = "data"):
        """Initialize dataset loader.
        
        Args:
            data_dir: Directory to save datasets.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def list_datasets(self) -> None:
        """List all available datasets."""
        print("Available Datasets:")
        print("=" * 50)
        
        categories = {
            "Classic Literature": ["shakespeare", "alice_wonderland", "pride_prejudice", "frankenstein", "edgar_allan_poe"],
            "Code": ["python_stdlib", "pytorch_code"],
            "Poetry": ["emily_dickinson", "robert_frost"],
            "Science": ["origin_species", "relativity"]
        }
        
        for category, datasets in categories.items():
            print(f"\n📚 {category}:")
            for name in datasets:
                if name in self.DATASETS:
                    info = self.DATASETS[name]
                    print(f"  • {name:20} - {info['description']} ({info['size']})")
    
    def download_dataset(self, name: str, force_download: bool = False) -> str:
        """Download a specific dataset.
        
        Args:
            name: Dataset name from DATASETS.
            force_download: Re-download even if file exists.
            
        Returns:
            Path to downloaded file.
        """
        if name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Use list_datasets() to see available options.")
        
        dataset_info = self.DATASETS[name]
        file_path = self.data_dir / f"{name}.txt"
        
        if file_path.exists() and not force_download:
            logger.info(f"Dataset {name} already exists at {file_path}")
            return str(file_path)
        
        logger.info(f"Downloading {name}: {dataset_info['description']}")
        
        try:
            # Handle single URL
            if "url" in dataset_info:
                text = self._download_url(dataset_info["url"])
                text = self._clean_gutenberg_text(text)
            
            # Handle multiple URLs (for code datasets)
            elif "urls" in dataset_info:
                texts = []
                for url in dataset_info["urls"]:
                    content = self._download_url(url)
                    texts.append(content)
                text = "\n\n" + "="*50 + "\n\n".join(texts)
            
            else:
                raise ValueError(f"Invalid dataset configuration for {name}")
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f"Downloaded {name} to {file_path} ({len(text):,} characters)")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            raise
    
    def _download_url(self, url: str) -> str:
        """Download text from URL."""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    
    def _clean_gutenberg_text(self, text: str) -> str:
        """Clean Project Gutenberg text by removing headers/footers."""
        lines = text.split('\n')
        
        # Find start of actual content (skip Gutenberg header)
        start_idx = 0
        for i, line in enumerate(lines):
            if "*** START" in line.upper() or "CHAPTER" in line or len(line.strip()) > 50:
                start_idx = i
                break
        
        # Find end of content (skip Gutenberg footer)
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if "*** END" in lines[i].upper() or "End of Project Gutenberg" in lines[i]:
                end_idx = i
                break
        
        # Join and clean
        content = '\n'.join(lines[start_idx:end_idx])
        
        # Remove excessive whitespace while preserving structure
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        return content.strip()
    
    def get_dataset_stats(self, file_path: str) -> Dict[str, int]:
        """Get statistics about a dataset.
        
        Args:
            file_path: Path to text file.
            
        Returns:
            Dictionary with dataset statistics.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chars = sorted(list(set(text)))
        
        return {
            "characters": len(text),
            "unique_chars": len(chars),
            "lines": text.count('\n'),
            "words": len(text.split()),
            "vocab_size": len(chars)
        }
    
    def compare_datasets(self, dataset_names: List[str]) -> None:
        """Compare statistics of multiple datasets."""
        print("Dataset Comparison:")
        print("=" * 80)
        print(f"{'Dataset':<20} {'Chars':<10} {'Vocab':<8} {'Lines':<8} {'Words':<10}")
        print("-" * 80)
        
        for name in dataset_names:
            try:
                file_path = self.download_dataset(name)
                stats = self.get_dataset_stats(file_path)
                print(f"{name:<20} {stats['characters']:<10,} {stats['vocab_size']:<8} "
                      f"{stats['lines']:<8,} {stats['words']:<10,}")
            except Exception as e:
                print(f"{name:<20} Error: {e}")


# Example usage functions
def download_recommended_datasets():
    """Download a curated set of interesting datasets."""
    loader = DatasetLoader()
    
    recommended = [
        "shakespeare",      # Original (for comparison)
        "alice_wonderland", # Classic literature
        "python_stdlib",    # Code (very interesting patterns!)
        "emily_dickinson",  # Poetry
        "edgar_allan_poe"   # Darker literature
    ]
    
    print("Downloading recommended datasets...")
    for dataset in recommended:
        try:
            loader.download_dataset(dataset)
        except Exception as e:
            print(f"Failed to download {dataset}: {e}")
    
    # Compare them
    loader.compare_datasets(recommended)


if __name__ == "__main__":
    # Demo
    loader = DatasetLoader()
    loader.list_datasets()
    
    # Download and compare some datasets
    download_recommended_datasets() 