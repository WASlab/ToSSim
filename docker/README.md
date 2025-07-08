# ToSSim Docker Setup

This directory contains Docker configurations for running ToSSim training and inference in containerized environments. The SFT training Dockerfiles exactly replicate the setup process from `ssh_instructions.txt`.

## Available Images

### Training Images
- **`Dockerfile.sft`** - Generic SFT training (matches ssh_instructions.txt exactly, auto-detects GPU configuration)
- **`Dockerfile.sft-single-gpu`** - Single GPU SFT training (forces CUDA_VISIBLE_DEVICES=0)
- **`Dockerfile.sft-multi-gpu`** - Multi-GPU SFT training with accelerate launch and FSDP
- **`Dockerfile.grpo`** - GRPO training

### Runtime Images
- **`Dockerfile.inference`** - Inference engine for running trained models
- **`Dockerfile.eval`** - Evaluation and testing environment

## Quick Start

### 1. Build Images

Build all images at once:
```bash
cd docker
./build.sh
```

Or build individual images:
```bash
# Generic SFT (recommended - matches SSH instructions exactly)
docker build -f docker/Dockerfile.sft -t tossim-sft .

# Single GPU SFT
docker build -f docker/Dockerfile.sft-single-gpu -t tossim-sft-single .

# Multi-GPU SFT with FSDP
docker build -f docker/Dockerfile.sft-multi-gpu -t tossim-sft-multi .

# GRPO
docker build -f docker/Dockerfile.grpo -t tossim-grpo .
```

### 2. Run Containers

Use the convenience script:
```bash
# Generic SFT (auto-detects GPUs, runs sft.py with gemma_sft.json)
./run.sh sft

# Single GPU SFT
./run.sh sft-single

# Multi-GPU SFT (uses accelerate launch with gemma_fsdp.json)
./run.sh sft-multi

# GRPO training
./run.sh grpo

# Inference
./run.sh inference

# Evaluation
./run.sh eval
```

Or run manually:
```bash
# Generic SFT
docker run -it --rm --gpus all -v $(pwd):/app tossim-sft

# Single GPU
docker run -it --rm --gpus 1 -v $(pwd):/app tossim-sft-single

# Multi-GPU with accelerate launch
docker run -it --rm --gpus all -v $(pwd):/app tossim-sft-multi

# GRPO with specific GPU devices
docker run -it --rm --gpus '"device=0,1"' -v $(pwd):/app tossim-grpo
```

## Configuration

### SFT Training Setup

The SFT Dockerfiles (`Dockerfile.sft`, `Dockerfile.sft-single-gpu`, `Dockerfile.sft-multi-gpu`) exactly replicate the setup from `ssh_instructions.txt`:

- **Python 3.12** with virtual environment
- **Exact package installation order**: transformers, trl, peft, accelerate, bitsandbytes, datasets, backoff
- **Specific flash-attention wheel**: `flash_attn-2.7.4+cu128torch2.7-cp312-cp312-linux_x86_64.whl`
- **CUDA 12.8 base image** to match the flash-attention wheel

### GPU Configuration

- **Generic (`Dockerfile.sft`)**: Uses `device_map="auto"` in sft.py for automatic GPU distribution
- **Single GPU (`Dockerfile.sft-single-gpu`)**: Forces `CUDA_VISIBLE_DEVICES=0`
- **Multi-GPU (`Dockerfile.sft-multi-gpu`)**: Uses `accelerate launch --multi_gpu` with FSDP configuration

### Volume Mounts

The run script automatically mounts:
- Current directory to `/app` (for code access)
- `./data` to `/app/data` (for data access)

### Environment Variables

For training with Hugging Face authentication:
```bash
docker run -it --rm --gpus all \
  -e HUGGINGFACE_HUB_TOKEN=your_token_here \
  -v $(pwd):/app \
  tossim-sft
```

## Usage Examples

### SFT Training
```bash
# Generic SFT (matches SSH setup exactly)
./run.sh sft

# Single GPU SFT
./run.sh sft-single

# Multi-GPU SFT with FSDP
./run.sh sft-multi

# Custom config
docker run -it --rm --gpus all -v $(pwd):/app tossim-sft \
  /bin/bash -c "source myenv/bin/activate && python sft.py your_config.json"
```

### GRPO Training
```bash
# GRPO training
./run.sh grpo
```

### Inference
```bash
# Run inference engine
./run.sh inference
```

### Evaluation
```bash
# Run tests
./run.sh eval
```

## Troubleshooting

### GPU Issues
- Ensure NVIDIA Docker runtime is installed
- Check GPU availability: `nvidia-smi`
- Verify Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi`

### Memory Issues
- Use smaller batch sizes in training configs
- Consider using different quantization settings (4-bit vs 8-bit)
- Monitor GPU memory usage during training

### Flash Attention Issues
The Dockerfiles use a specific pre-built flash-attention wheel. If you encounter issues:
- Verify your GPU supports flash attention (Ampere+ architecture)
- Check CUDA compatibility (the wheel is built for CUDA 12.8)

### Virtual Environment
All SFT containers use a Python 3.12 virtual environment at `/app/myenv`. To run custom commands:
```bash
docker run -it --rm --gpus all -v $(pwd):/app tossim-sft \
  /bin/bash -c "source myenv/bin/activate && your_command"
```

## Requirements

- Docker with NVIDIA runtime support
- NVIDIA drivers compatible with CUDA 12.8
- At least 16GB RAM (32GB+ recommended for training)
- GPU with 8GB+ VRAM (24GB+ recommended for large models like Gemma-3-27B)

## Training Configurations

The SFT containers come with default configurations:
- **Single GPU / Generic**: Uses `training_configs/gemma_sft.json`
- **Multi-GPU**: Uses `training_configs/gemma_fsdp.json` with FSDP settings

To use custom configurations, mount them and override the command:
```bash
docker run -it --rm --gpus all \
  -v $(pwd):/app \
  -v /path/to/your/config.json:/app/my_config.json \
  tossim-sft \
  /bin/bash -c "source myenv/bin/activate && python sft.py my_config.json"
``` 