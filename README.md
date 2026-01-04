# Table of Contents
* [Introduction](#nvidia-ai-workbench-introduction)
   * [Project Description](#project-description)
   * [DGX Spark GB10 Optimization](#dgx-spark-gb10-optimization)
   * [Sizing Guide](#sizing-guide)
* [Quickstart](#quickstart)
   * [Prerequisites](#prerequisites)
   * [Tutorial (Desktop App)](#tutorial-desktop-app)
   * [Tutorial (CLI-Only)](#tutorial-cli-only)
* [Troubleshooting](#troubleshooting)
* [License](#license)

# NVIDIA AI Workbench: Introduction [![Open In AI Workbench](https://img.shields.io/badge/Open_In-AI_Workbench-76B900)](https://ngc.nvidia.com/open-ai-workbench/aHR0cHM6Ly9naXRodWIuY29tL05WSURJQS93b3JrYmVuY2gtZXhhbXBsZS1sbGFtYTMtZmluZXR1bmU=)

<!-- Banner Image -->
<img src="https://developer-blogs.nvidia.com/wp-content/uploads/2024/07/rag-representation.jpg" width="100%">

<!-- Links -->
<p align="center"> 
  <a href="https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/" style="color: #76B900;">:arrow_down: Download AI Workbench</a> •
  <a href="https://docs.nvidia.com/ai-workbench/" style="color: #76B900;">:book: Read the Docs</a> •
  <a href="https://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/example-projects.html" style="color: #76B900;">:open_file_folder: Explore Example Projects</a> •
  <a href="https://forums.developer.nvidia.com/t/support-workbench-example-project-llama-3-finetune/303411" style="color: #76B900;">:rotating_light: Facing Issues? Let Us Know!</a>
</p>

## Project Description

> **This branch is optimized for NVIDIA DGX Spark with GB10** (Grace Blackwell architecture, sm_121). It uses the [allura-forge/Llama-3.3-8B-Instruct](https://huggingface.co/allura-forge/Llama-3.3-8B-Instruct) model and leverages the 128GB unified memory for full-precision BF16 training without quantization.

The Llama-3.3-8B-Instruct model is an advanced LLM that demonstrates improved performance over Llama 3.1 8B on reasoning, code generation, and contextual understanding tasks. In this project, we focus on finetuning this model in two ways:

1. **`llama3_finetune_inference.ipynb`**: Supervised Fine-Tuning (SFT)

    Fine-tune the Llama-3.3-8B-Instruct model using SFT on the OpenAssistant Guanaco dataset to improve conversational and instruction-following capabilities. Deploy and test using vLLM.

2. **`llama3dpo.ipynb`**: Direct Preference Optimization (DPO)

    Fine-tune using DPO to align the model with human preferences without requiring a separate reward model.

### What is Direct Preference Optimization (DPO)? 

DPO improves on RLHF by treating alignment as a classification problem. It uses the trained model and a reference model copy. During training, the goal is to make the trained model output higher probabilities for preferred answers and lower probabilities for rejected answers. This results in a more stable and less computationally intensive process than traditional RLHF.

| :memo: Remember             |
| :---------------------------|
| This project is meant as an example workflow and a starting point; you are free to swap out the dataset, choose a different task, and edit the training prompts as you see fit for your particular use case! |

## DGX Spark GB10 Optimization

This branch has been specifically optimized for **NVIDIA DGX Spark** systems featuring the **GB10 Grace Blackwell** superchip.

### Key Optimizations

| Feature | Standard GPUs | DGX Spark GB10 |
|---------|---------------|----------------|
| **Memory** | 24-80GB VRAM | 128GB Unified Memory |
| **Precision** | 4-bit quantization (QLoRA) | Full BF16 (no quantization) |
| **Attention** | Flash Attention 2 | Native SDPA |
| **Batch Size** | 1-2 | 4+ |
| **Gradient Checkpointing** | Required | Optional (disabled for speed) |
| **Optimizer** | Paged AdamW 32-bit | Standard AdamW |
| **Base Image** | PyTorch 24.xx (CUDA 12.2) | **PyTorch 25.10+ (CUDA 12.8+)** |

### Why These Changes?

The GB10 GPU uses the **sm_121** compute capability (Blackwell architecture), which requires:

1. **NGC PyTorch 25.10+**: Older PyTorch versions don't support sm_121. The container must use `nvcr.io/nvidia/pytorch:25.10-py3` or newer.

2. **SDPA instead of Flash Attention**: Flash Attention kernels may not be compiled for sm_121. PyTorch's native Scaled Dot Product Attention (SDPA) is fully supported and very fast on Blackwell.

3. **No bitsandbytes/quantization**: With 128GB of unified memory, there's no need for 4-bit quantization. Full BF16 training provides better quality.

4. **No paged optimizers**: The large memory pool eliminates the need for memory-saving optimizer tricks.

### Architecture Support

If you're compiling custom CUDA extensions, use these flags:
```bash
# CMake
-DCMAKE_CUDA_ARCHITECTURES=121

# Environment variable
export TORCH_CUDA_ARCH_LIST="12.1"
```

## Sizing Guide

| GPU VRAM | Example Hardware | Compatible? |
| -------- | ------- | ------- |
| <16 GB | RTX 3080, RTX 3500 Ada | N |
| 16 GB | RTX 4080 16GB, RTX A4000 | Y (DPO only, with quantization) |
| 24 GB | RTX 3090/4090, RTX A5000/5500, A10/30 | Y (DPO only, with quantization) |
| 32 GB | RTX 5000 Ada  | Y (DPO only, with quantization) |
| 40 GB | A100-40GB | Y (DPO only, with quantization) |
| 48 GB | RTX 6000 Ada, L40/L40S, A40 | Y (DPO only, with quantization) |
| 80 GB | A100-80GB | Y |
| **128 GB** | **DGX Spark GB10** | **Y (Full BF16, Recommended)** |
| >80 GB | 8x A100-80GB | Y |

# Quickstart

## Prerequisites

AI Workbench will prompt you to provide a few pieces of information before running any apps in this project:
   
* The location where you would like the Llama-3.3-8B-Instruct models to live on the underlying **host** system
* A Hugging Face API Key for downloading the model

| :exclamation: Important             |
| :---------------------------|
| Unlike the original Meta Llama 3 models, the [allura-forge/Llama-3.3-8B-Instruct](https://huggingface.co/allura-forge/Llama-3.3-8B-Instruct) model does not require special access approval. You only need a valid Hugging Face token for authentication. |

## Tutorial (Desktop App)

If you do not have NVIDIA AI Workbench installed, first complete the installation for AI Workbench [here](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/). Then:

1. Fork this Project to your own GitHub namespace and copy the link

   ```
   https://github.com/[your_namespace]/<project_name>
   ```
   
2. Open NVIDIA AI Workbench. Select a location to work in.
   
3. Clone this Project onto your desired machine by selecting **Clone Project** and providing the GitHub link. **Select the `llama-3.3-8b-instruct` branch.**
   
4. Wait for the project to build. You can expand the bottom **Building** indicator to view real-time build logs.

   > **Note for DGX Spark GB10**: The build uses NGC PyTorch 25.10+ which includes native Blackwell support. No additional PyTorch installation is needed.
   
5. When the build completes, set the following configurations:

   * `Environment` → `Mounts` → `Configure`: Specify the file path of the mount, e.g., where the Llama-3.3-8B-Instruct models will live on your **host** machine.
   
      Example: `/home/[user]` or `/mnt/C/Users/[user]` (Windows)

   * `Environment` → `Secrets` → `Configure`: Specify the Hugging Face Token as a project secret.

6. On the top right of the window, select **Jupyterlab**.

7. Navigate to the `code` directory of the project. Then, open your fine-tuning notebook of choice and get started. Happy coding!

## Tutorial (CLI-Only)

Some users may choose to use the **CLI tool only** instead of the Desktop App. If you do not have NVIDIA AI Workbench installed, first complete the installation for AI Workbench [here](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/). Then:

1. Fork this Project to your own GitHub namespace and copy the link

   ```
   https://github.com/[your_namespace]/<project_name>
   ```
   
2. Open a shell and activate the Context you want to clone into:

   ```bash
   $ nvwb list contexts
   $ nvwb activate <desired_context>
   ```

   | :bulb: Tip                  |
   | :---------------------------|
   | Use `nvwb help` to see a full list of AI Workbench commands. |
   
3. Clone this Project onto your desired machine:

   ```bash
   $ nvwb clone project <your_project_link> --branch llama-3.3-8b-instruct
   ```
   
4. Open the Project:

   ```bash
   $ nvwb list projects
   $ nvwb open <project_name>
   ```

5. Start **Jupyterlab**:

   ```bash
   $ nvwb start jupyterlab
   ```
   
   * Specify the file path of the mount
   * Specify the Hugging Face Token as a project secret

6. Navigate to the `code` directory of the project. Then, open your fine-tuning notebook of choice and get started!

# Troubleshooting

## Common Issues on DGX Spark GB10

### "NVIDIA GB10 with CUDA capability sm_121 is not compatible"

**Cause**: The base container uses an older PyTorch version that doesn't support Blackwell architecture.

**Solution**: Ensure you're using this branch (`llama-3.3-8b-instruct`) which uses NGC PyTorch 25.10+ with sm_121 support.

### "FATAL: kernel built for sm80-sm100, but running on sm121"

**Cause**: Flash Attention or other CUDA extensions were compiled for older architectures.

**Solution**: This branch uses PyTorch's native SDPA (`attn_implementation="sdpa"`) instead of Flash Attention. No action needed if using the provided notebooks.

### "ModuleNotFoundError: No module named 'bitsandbytes'"

**Cause**: The notebooks were designed for quantization on smaller GPUs.

**Solution**: This branch removes bitsandbytes dependency. With 128GB memory, quantization is unnecessary.

### Build fails with "com.nvidia.workbench.schema-version not set"

**Cause**: The NGC PyTorch container doesn't have AI Workbench labels by default.

**Solution**: This branch includes a custom Dockerfile that adds the required labels. If issues persist, try rebuilding the project.

# License

This NVIDIA AI Workbench example project is under the [Apache 2.0 License](https://github.com/NVIDIA/workbench-example-llama3-finetune/blob/main/LICENSE.txt)

The Llama-3.3-8B-Instruct model is distributed under the [Llama 3.3 Community License](https://huggingface.co/allura-forge/Llama-3.3-8B-Instruct).

This project may utilize additional third-party open source software projects. Review the license terms of these open source projects before use. Third party components used as part of this project are subject to their separate legal notices or terms that accompany the components. You are responsible for confirming compliance with third-party component license terms and requirements. 

| :question: Have Questions?  |
| :---------------------------|
| Please direct any issues, fixes, suggestions, and discussion on this project to the DevZone Members Only Forum thread [here](https://forums.developer.nvidia.com/t/support-workbench-example-project-llama-3-finetune/303411) |
