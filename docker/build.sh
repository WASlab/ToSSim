#!/bin/bash

# Build script for ToSSim Docker images
# Usage: ./build.sh [image_name]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default image name
DEFAULT_IMAGE="tossim"

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

# Get image name from argument or use default
IMAGE_NAME=${1:-$DEFAULT_IMAGE}

print_status "Building ToSSim Docker images with tag: $IMAGE_NAME"

# Build all images
print_status "Building generic SFT image..."
docker build -f docker/Dockerfile.sft -t ${IMAGE_NAME}-sft .

print_status "Building single GPU SFT image..."
docker build -f docker/Dockerfile.sft-single-gpu -t ${IMAGE_NAME}-sft-single .

print_status "Building multi-GPU SFT image..."
docker build -f docker/Dockerfile.sft-multi-gpu -t ${IMAGE_NAME}-sft-multi .

print_status "Building GRPO image..."
docker build -f docker/Dockerfile.grpo -t ${IMAGE_NAME}-grpo .

print_status "Building inference image..."
docker build -f docker/Dockerfile.inference -t ${IMAGE_NAME}-inference .

print_status "Building evaluation image..."
docker build -f docker/Dockerfile.eval -t ${IMAGE_NAME}-eval .

print_status "All images built successfully!"
print_status "Available images:"
docker images | grep ${IMAGE_NAME} 