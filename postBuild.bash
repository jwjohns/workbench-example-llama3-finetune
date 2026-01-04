#!/bin/bash
# This file contains bash commands that will be executed at the end of the container build process,
# after all system packages and programming language specific package have been installed.
#
# DGX Spark GB10 Configuration
# ============================
# The NGC PyTorch 25.10+ container already includes:
# - PyTorch 2.5+ with CUDA 12.8+ and sm_121 (Blackwell) support
# - cuDNN, NCCL, and other NVIDIA libraries optimized for Grace Blackwell
#
# This script installs additional ML libraries needed for fine-tuning.

echo "=== DGX Spark GB10 Post-Build Configuration ==="
echo "Installing fine-tuning dependencies..."

# Install required Python packages
pip install --no-cache-dir \
    datasets>=2.19.1 \
    transformers>=4.45.0 \
    peft>=0.11.1 \
    accelerate>=0.30.1 \
    trl>=0.8.6 \
    ipywidgets>=8.1.3 \
    wandb>=0.17.0 \
    rich>=13.0.0

# Install vLLM for model deployment (may need special handling on ARM64)
pip install --no-cache-dir vllm>=0.6.0 || echo "Note: vLLM installation may require manual setup on ARM64"

# Set environment variables for Blackwell architecture
export TORCH_CUDA_ARCH_LIST="12.1"

# Verify PyTorch and CUDA installation
echo ""
echo "=== Verifying Installation ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=== DGX Spark GB10 Post-Build Complete ==="
