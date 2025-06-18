#!/usr/bin/env python3
"""Main training script for GPT model - Python code generation."""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append("src")

import torch
from torch.utils.data import DataLoader

from config.settings import get_config, get_small_config, get_large_config
from data.dataset import PythonCodeDataLoader, TextDataset
from data.tokenizer import CharacterTokenizer
from models.gpt import GPTModel
from training.trainer import Trainer


def setup_logging(log_dir: str = "logs") -> None:
    """Setup logging configuration."""
    Path(log_dir).mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(log_dir) / 'training.log'),
            logging.StreamHandler()
        ]
    )


def preview_dataset():
    """Preview the Python code dataset."""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("🐍 PYTHON CODE DATASET PREVIEW")
    print("="*60)
    
    # Create data loader and download data
    data_loader = PythonCodeDataLoader()
    data_loader.download_python_code(max_files=5)  # Download first 5 files for preview
    
    # Show preview
    data_loader.preview_data(num_chars=800)
    
    # Load and show statistics
    train_data, val_data, tokenizer = data_loader.load_data()
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total characters: {len(train_data) + len(val_data):,}")
    print(f"   Training tokens: {len(train_data):,}")
    print(f"   Validation tokens: {len(val_data):,}")
    print(f"   Vocabulary size: {len(tokenizer)}")
    print(f"   Unique characters: {''.join(tokenizer.chars[:50])}{'...' if len(tokenizer.chars) > 50 else ''}")
    
    # Show some encoded/decoded examples
    sample_text = "def hello_world():\n    print('Hello, World!')\n    return True"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\n🔤 Tokenization Example:")
    print(f"   Original: {repr(sample_text)}")
    print(f"   Encoded:  {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
    print(f"   Decoded:  {repr(decoded)}")


def load_data(config: dict) -> tuple:
    """Load and prepare Python code data."""
    logger = logging.getLogger(__name__)
    
    # Create data loader
    data_loader = PythonCodeDataLoader(
        data_path=config['data'].data_path,
        train_split=config['data'].train_split,
        block_size=config['model'].block_size
    )
    
    # Download Python code
    data_loader.download_python_code(max_files=config['data'].max_files)
    
    # Load data
    train_data, val_data, tokenizer = data_loader.load_data()
    
    # Update vocab size in config
    config['model'].vocab_size = len(tokenizer)
    
    # Create datasets
    train_dataset = TextDataset(train_data, config['model'].block_size)
    val_dataset = TextDataset(val_data, config['model'].block_size)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training'].batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training'].batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer


def create_model(config: dict) -> torch.nn.Module:
    """Create and initialize model."""
    model_config = config['model']
    
    model = GPTModel(
        vocab_size=model_config.vocab_size,
        n_embd=model_config.n_embd,
        n_head=model_config.n_head,
        n_layer=model_config.n_layer,
        block_size=model_config.block_size,
        dropout=model_config.dropout
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created model with {model.get_num_params():,} parameters")
    
    return model


def generate_code_sample(model, tokenizer, config, prompt: str = "def "):
    """Generate a sample of Python code."""
    logger = logging.getLogger(__name__)
    
    try:
        model.eval()
        device = next(model.parameters()).device
        
        # Encode prompt
        prompt_tokens = torch.tensor(
            tokenizer.encode(prompt), 
            dtype=torch.long, 
            device=device
        ).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                prompt_tokens,
                max_new_tokens=config['generation'].max_new_tokens,
                temperature=config['generation'].temperature,
                top_k=config['generation'].top_k,
                top_p=config['generation'].top_p
            )
        
        # Decode
        generated_text = tokenizer.decode(generated[0].tolist())
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Failed to generate sample: {e}")
        return f"Error generating sample: {e}"


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train GPT model on Python code')
    parser.add_argument('--config', choices=['small', 'default', 'large'], default='default',
                       help='Model configuration size')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--preview', action='store_true', help='Preview dataset and exit')
    parser.add_argument('--generate-only', action='store_true', help='Generate sample and exit')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Preview dataset if requested
        if args.preview:
            preview_dataset()
            return
        
        # Load configuration
        if args.config == 'small':
            config = get_small_config()
        elif args.config == 'large':
            config = get_large_config()
        else:
            config = get_config()
        
        if args.wandb:
            config['training'].use_wandb = True
        
        logger.info("🐍 Starting GPT training on Python code")
        logger.info(f"Configuration: {args.config}")
        
        # Set device
        device = config['system'].device
        logger.info(f"Using device: {device}")
        
        # Set random seed
        torch.manual_seed(config['system'].seed)
        if device == 'cuda':
            torch.cuda.manual_seed(config['system'].seed)
        
        # Load data
        train_loader, val_loader, tokenizer = load_data(config)
        
        # Create model
        model = create_model(config)
        
        # If generate-only mode, load checkpoint and generate
        if args.generate_only:
            if args.resume:
                checkpoint = torch.load(args.resume, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from {args.resume}")
            
            model.to(device)
            
            # Generate samples with different prompts
            prompts = [
                "def ",
                "class ",
                "import ",
                "for i in ",
                "if __name__ == '__main__':\n    "
            ]
            
            print("\n" + "="*60)
            print("🤖 GENERATED PYTHON CODE SAMPLES")
            print("="*60)
            
            for prompt in prompts:
                print(f"\n📝 Prompt: {repr(prompt)}")
                print("-" * 40)
                sample = generate_code_sample(model, tokenizer, config, prompt)
                print(sample[:300])  # Show first 300 chars
                print("-" * 40)
            
            return
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config['training'].__dict__,
            device=device
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # Train model
        trainer.train()
        
        # Generate sample Python code
        sample_prompts = ["def ", "class MyClass", "import "]
        for prompt in sample_prompts:
            sample_text = generate_code_sample(model, tokenizer, config, prompt)
            logger.info(f"Sample generation with prompt '{prompt}':\n{sample_text[:200]}...")
        
        logger.info("🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
