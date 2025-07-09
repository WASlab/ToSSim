#!/bin/bash
#
# Fully automated setup script for the ToSSim remote environment.
# This script is non-interactive and can be run on a fresh Ubuntu 22.04 server.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. System Package Installation ---

# Set frontend to noninteractive to avoid prompts (e.g., from tzdata)
export DEBIAN_FRONTEND=noninteractive

echo "INFO: Updating package lists..."
sudo apt-get update -y

echo "INFO: Installing base dependencies..."
# Install common software properties and the PPA for newer Python versions.
# The -y flag automatically confirms the installations.
# The Dpkg::Options force-confold/confdef handles any config file conflicts automatically.
sudo apt-get install -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    software-properties-common \
    build-essential \
    curl \
    git

echo "INFO: Adding deadsnakes PPA for modern Python versions..."
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update -y

echo "INFO: Installing Python 3.12..."
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

# --- 2. Project Setup ---

echo "INFO: Cloning the ToSSim repository..."
# Clone the repository into the home directory
cd ~
git clone https://github.com/Salem-AI-Sim/ToSSim

cd ToSSim

echo "INFO: Creating Python virtual environment..."
python3.12 -m venv myenv

echo "INFO: Activating virtual environment..."
source myenv/bin/activate

# --- 3. Python Dependency Installation ---

echo "INFO: Upgrading pip..."
pip install -U pip setuptools wheel

echo "INFO: Installing project requirements..."
# Install all Python dependencies in one command.
# Using --extra-index-url for PyTorch to ensure the correct CUDA version is found.
pip install \
    transformers \
    trl \
    peft \
    accelerate \
    bitsandbytes \
    datasets \
    backoff \
    torch \
    --extra-index-url https://download.pytorch.org/whl/cu128

echo "INFO: Installing Flash Attention 2..."
# Install Flash Attention from the pre-built wheel for CUDA 12.x
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8-cp312-cp312-linux_x86_64.whl

echo "---"
echo "✅ Setup complete!"
echo "Run 'source ~/ToSSim/myenv/bin/activate' to activate the environment."
echo "---" 