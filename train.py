#!/usr/bin/env python3
"""Main training script for GPT model."""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append("src")

import torch
from config.settings import get_config

def main():
    """Main training function."""
    config = get_config()
    print("GPT Training Script")
    print(f"Config: {config}")
    
if __name__ == "__main__":
    main()
