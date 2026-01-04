#!/bin/bash
# This file contains bash commands that will be executed at the end of the container build process,
# after all system packages and programming language specific package have been installed.
#
# DGX Spark GB10 requires PyTorch with CUDA 13.0 and sm_121 support
# The standard PyTorch wheels do not support sm_121 (Grace Blackwell architecture)

echo "Installing PyTorch with CUDA 13.0 support for DGX Spark GB10 (sm_121)..."

# Install PyTorch nightly with CUDA 13.0 support
# This is required for DGX Spark GB10 which uses sm_121 compute capability
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Verify PyTorch installation and CUDA support
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo "PyTorch installation complete for DGX Spark GB10"
