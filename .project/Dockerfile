# Custom Dockerfile for DGX Spark GB10 (Grace Blackwell sm_121)
# Based on NGC PyTorch 25.10 with AI Workbench compatibility labels

FROM nvcr.io/nvidia/pytorch:25.10-py3

# Add AI Workbench required labels
LABEL com.nvidia.workbench.name="PyTorch 25.10 for DGX Spark GB10"
LABEL com.nvidia.workbench.description="NGC PyTorch 25.10 with CUDA 12.8+ and Blackwell (sm_121) support for DGX Spark GB10"
LABEL com.nvidia.workbench.schema-version="v2"
LABEL com.nvidia.workbench.image-version="25.10.0"
LABEL com.nvidia.workbench.cuda-version="12.8"

# Additional labels for AI Workbench compatibility
LABEL com.nvidia.workbench.os="linux"
LABEL com.nvidia.workbench.os-distro="ubuntu"
LABEL com.nvidia.workbench.os-distro-release="22.04"
LABEL com.nvidia.workbench.programming-languages="python3"

# Install JupyterLab and other dependencies
RUN pip install --no-cache-dir \
    jupyterlab>=4.0.0 \
    datasets>=2.19.1 \
    transformers>=4.45.0 \
    peft>=0.11.1 \
    accelerate>=0.30.1 \
    trl>=0.8.6 \
    ipywidgets>=8.1.3 \
    wandb>=0.17.0 \
    rich>=13.0.0 \
    vllm>=0.6.0

# Set environment variables for Blackwell architecture
ENV TORCH_CUDA_ARCH_LIST="12.1"
ENV CUDA_VISIBLE_DEVICES="0"

# Set working directory
WORKDIR /project
