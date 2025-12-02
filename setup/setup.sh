#!/bin/bash
# Quick start script for setting up HuggingFace and downloading models

set -e

echo "==================================="
echo "LLM Second-Order Effects Setup"
echo "==================================="
echo ""

# Activate environment
echo "1. Activating virtual environment..."
cd "/scratch2/f004ndc/LLM Second-Order Effects"
source 2OE_env/bin/activate
echo "✓ Environment activated"
echo ""

# Check if already logged in
echo "2. Checking HuggingFace authentication..."
if huggingface-cli whoami &>/dev/null; then
    echo "✓ Already authenticated with HuggingFace"
else
    echo "⚠ Not authenticated. Please follow these steps:"
    echo ""
    echo "   a. Accept Llama 3 license: https://huggingface.co/meta-llama/Meta-Llama-3-8B"
    echo "   b. Create token: https://huggingface.co/settings/tokens"
    echo "   c. Run: huggingface-cli login"
    echo ""
    read -p "Have you completed authentication? (y/n): " auth_done
    if [ "$auth_done" != "y" ]; then
        echo "Please complete authentication first, then run this script again."
        exit 1
    fi
fi
echo ""

# Download models
echo "3. Ready to download models!"
echo "   Run: python download_models.py"
echo ""
read -p "Start download now? (y/n) [y]: " start_download
start_download=${start_download:-y}

if [ "$start_download" = "y" ]; then
    python download_models.py
else
    echo "Download skipped. Run 'python download_models.py' when ready."
fi

echo ""
echo "Setup complete!"
