#!/usr/bin/env python3
"""
Download LLM models for the Second-Order Effects experiment.

This script downloads Gemma 2B, Llama 3 8B, and other models to a shared directory.
Requires HuggingFace authentication for Llama models.
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi

# Models to download
MODELS = {
    "gemma-2b": {
        "hf_name": "google/gemma-2b",
        "requires_auth": False,
        "size_gb": 4.5,
    },
    "llama3-8b": {
        "hf_name": "meta-llama/Meta-Llama-3-8B",
        "requires_auth": True,
        "size_gb": 16,
    },
    "gpt2-medium": {
        "hf_name": "gpt2-medium",
        "requires_auth": False,
        "size_gb": 1.5,
    },
    "pythia-1.4b": {
        "hf_name": "EleutherAI/pythia-1.4b",
        "requires_auth": False,
        "size_gb": 2.8,
    },
}


def check_authentication():
    """Check if HuggingFace authentication is set up."""
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✓ Authenticated as: {user['name']}")
        return True
    except Exception as e:
        print("✗ Not authenticated with HuggingFace")
        print("Please run: huggingface-cli login")
        return False


def download_model(model_key, model_info, cache_dir):
    """Download a single model and its tokenizer."""
    print(f"\n{'='*60}")
    print(f"Downloading {model_key}: {model_info['hf_name']}")
    print(f"Estimated size: {model_info['size_gb']} GB")
    print(f"{'='*60}")
    
    try:
        # Download tokenizer
        print(f"[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_info['hf_name'],
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        print(f"✓ Tokenizer downloaded")
        
        # Download model
        print(f"[2/2] Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_info['hf_name'],
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        print(f"✓ Model downloaded successfully")
        
        # Clean up to save memory
        del model
        del tokenizer
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {model_key}: {e}")
        return False


def main():
    # Set up cache directory
    shared_models_dir = Path("/scratch2/shared_models")
    
    # Ask user which directory to use
    print("Where would you like to store the models?")
    print(f"1. Shared directory: {shared_models_dir}")
    print(f"2. Project directory: ./models")
    print(f"3. Custom path")
    
    choice = input("Enter choice (1/2/3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        cache_dir = shared_models_dir
    elif choice == "2":
        cache_dir = Path(__file__).parent / "models"
    elif choice == "3":
        custom_path = input("Enter custom path: ").strip()
        cache_dir = Path(custom_path)
    else:
        print("Invalid choice, using shared directory")
        cache_dir = shared_models_dir
    
    # Create directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nUsing cache directory: {cache_dir}")
    
    # Check authentication
    print("\nChecking HuggingFace authentication...")
    authenticated = check_authentication()
    
    # Filter models based on authentication
    models_to_download = {}
    for key, info in MODELS.items():
        if info['requires_auth'] and not authenticated:
            print(f"⚠ Skipping {key} (requires authentication)")
        else:
            models_to_download[key] = info
    
    if not models_to_download:
        print("\n✗ No models to download. Please authenticate first.")
        sys.exit(1)
    
    # Calculate total size
    total_size = sum(info['size_gb'] for info in models_to_download.values())
    print(f"\nWill download {len(models_to_download)} models")
    print(f"Total estimated size: {total_size:.1f} GB")
    print(f"Models: {', '.join(models_to_download.keys())}")
    
    # Confirm download
    confirm = input("\nProceed with download? (y/n) [y]: ").strip().lower() or "y"
    if confirm != 'y':
        print("Download cancelled")
        sys.exit(0)
    
    # Download models
    success_count = 0
    for model_key, model_info in models_to_download.items():
        if download_model(model_key, model_info, cache_dir):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Successfully downloaded: {success_count}/{len(models_to_download)} models")
    print(f"Models stored in: {cache_dir}")
    print(f"{'='*60}")
    
    # Create symlink if using shared directory
    if cache_dir == shared_models_dir:
        local_link = Path(__file__).parent / "models"
        if not local_link.exists():
            print(f"\nCreating symlink: models -> {cache_dir}")
            local_link.symlink_to(cache_dir)
            print("✓ Symlink created")


if __name__ == "__main__":
    main()
