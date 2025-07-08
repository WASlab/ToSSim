#!/bin/bash

# Run script for ToSSim Docker containers
# Usage: ./run.sh [image_type] [additional_args]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DEFAULT_IMAGE="tossim"
DEFAULT_TYPE="sft"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_usage() {
    echo "Usage: $0 [image_type] [additional_docker_args...]"
    echo ""
    echo "Image types:"
    echo "  sft          - Generic SFT training (auto-detect GPUs)"
    echo "  sft-single   - Single GPU SFT training"
    echo "  sft-multi    - Multi-GPU SFT training"
    echo "  grpo         - GRPO training"
    echo "  inference    - Inference engine"
    echo "  eval         - Evaluation and testing"
    echo ""
    echo "Examples:"
    echo "  $0 sft"
    echo "  $0 sft-multi --gpus all"
    echo "  $0 grpo -v /path/to/data:/app/data"
    echo "  $0 inference -p 8000:8000"
}

# Check if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

# Get image type from argument or use default
IMAGE_TYPE=${1:-$DEFAULT_TYPE}
shift  # Remove first argument, keep the rest for docker run

# Validate image type
VALID_TYPES=("sft" "sft-single" "sft-multi" "grpo" "inference" "eval")
if [[ ! " ${VALID_TYPES[@]} " =~ " ${IMAGE_TYPE} " ]]; then
    print_error "Invalid image type: $IMAGE_TYPE"
    print_usage
    exit 1
fi

# Set image name
IMAGE_NAME="${DEFAULT_IMAGE}-${IMAGE_TYPE}"

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    print_error "Image $IMAGE_NAME not found. Please build it first with:"
    print_error "  docker build -f docker/Dockerfile.${IMAGE_TYPE} -t $IMAGE_NAME ."
    exit 1
fi

# Set up GPU flags based on image type
GPU_FLAGS=""
if [[ "$IMAGE_TYPE" == "sft-multi" || "$IMAGE_TYPE" == "grpo" ]]; then
    GPU_FLAGS="--gpus all"
elif [[ "$IMAGE_TYPE" == "sft-single" ]]; then
    GPU_FLAGS="--gpus 1"
elif [[ "$IMAGE_TYPE" == "sft" ]]; then
    # Auto-detect GPUs
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        if [ "$GPU_COUNT" -gt 0 ]; then
            GPU_FLAGS="--gpus all"
            print_status "Auto-detected $GPU_COUNT GPU(s), using --gpus all"
        fi
    fi
fi

# Set up volume mounts
VOLUME_FLAGS="-v $(pwd):/app -v $(pwd)/data:/app/data"

print_status "Running $IMAGE_NAME with GPU flags: $GPU_FLAGS"
print_status "Additional args: $@"

# Run the container
docker run -it --rm \
    $GPU_FLAGS \
    $VOLUME_FLAGS \
    --network host \
    "$@" \
    "$IMAGE_NAME" 