#!/usr/bin/env python3
"""
Co-located GRPO Training Launcher

This script provides an easy way to launch co-located GRPO training
with proper multi-GPU configuration using torchrun.

The co-located approach eliminates GPU idle time by running vLLM
and FSDP training on the same GPUs, following TRL's state-of-the-art method.

Usage:
    # Single GPU
    python run_colocated_grpo.py --gpus 1

    # Multi-GPU on single node
    python run_colocated_grpo.py --gpus 4

    # Multi-node (advanced)
    python run_colocated_grpo.py --gpus 8 --nodes 2 --node-rank 0 --master-addr 192.168.1.100
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import yaml


def validate_config(config_path):
    """
    Loads and validates the config file, printing warnings for missing, extra, or suspicious keys.
    For vllm_config, warns if enforce_eager is not True (unless user disables it).
    """
    from collections.abc import Mapping
    import warnings

    # These are the top-level keys in DrGRPOConfig
    expected_keys = {
        'model_name', 'fsdp_config', 'vllm_config', 'learning_rate', 'batch_size', 'max_iterations', 'log_interval',
        'beta', 'sync_frequency', 'verbosity_penalty', 'num_games', 'active_seats_per_game',
        'temperature', 'top_p', 'max_tokens', 'max_think_tokens', 'enable_verbosity_penalty',
    }
    # Acceptable types for some keys
    expected_types = {
        'model_name': str,
        'learning_rate': float,
        'batch_size': int,
        'max_iterations': int,
        'log_interval': int,
        'beta': float,
        'sync_frequency': int,
        'verbosity_penalty': float,
        'num_games': int,
        'active_seats_per_game': int,
        'temperature': float,
        'top_p': float,
        'max_tokens': int,
        'max_think_tokens': (int, type(None)),
        'enable_verbosity_penalty': bool,
    }
    # Load YAML
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"[Config Warning] Could not load config: {e}")
        return
    if not isinstance(config, dict):
        print(f"[Config Warning] Config file does not contain a dictionary at the top level.")
        return
    # Check for missing keys
    for key in expected_keys:
        if key not in config:
            print(f"[Config Warning] Missing key: '{key}' (will use default if available)")
    # Check for extra keys
    for key in config:
        if key not in expected_keys:
            print(f"[Config Warning] Unknown key: '{key}' (may be a typo or new feature)")
    # Check types
    for key, typ in expected_types.items():
        if key in config and not isinstance(config[key], typ):
            print(f"[Config Warning] '{key}' has suspicious type: {type(config[key])}, expected {typ}")
    # Check for negative or zero values where not allowed
    for key in ['batch_size', 'max_iterations', 'num_games', 'active_seats_per_game', 'max_tokens']:
        if key in config and isinstance(config[key], int) and config[key] <= 0:
            print(f"[Config Warning] '{key}' is {config[key]}, which is probably a mistake.")
    # vllm_config: check enforce_eager
    vllm_cfg = config.get('vllm_config', {})
    if not isinstance(vllm_cfg, Mapping):
        print(f"[Config Warning] 'vllm_config' should be a dict, got: {type(vllm_cfg)}")
    else:
        enforce_eager = vllm_cfg.get('enforce_eager', True)
        if not enforce_eager:
            print(f"[Config Warning] vLLM 'enforce_eager' is set to False. This may cause instability for Gemma models or co-located vLLM. Only disable if you know what you're doing.")


def main():
    parser = argparse.ArgumentParser(description="Co-located GRPO Training Launcher")
    
    # Hardware configuration
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--nodes", type=int, default=1,
                        help="Number of nodes (for multi-node training)")
    parser.add_argument("--node-rank", type=int, default=0,
                        help="Rank of this node (0 for master node)")
    parser.add_argument("--master-addr", type=str, default="localhost",
                        help="Master node address (for multi-node)")
    parser.add_argument("--master-port", type=str, default="29500",
                        help="Master node port")
    
    # Training configuration
    parser.add_argument("--config", type=str, default="training_configs/dr_grpo_config.yaml",
                        help="Path to GRPO configuration file")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    # Advanced options
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command without executing")
    parser.add_argument("--redirect-output", type=str, default=None,
                        help="Redirect output to file")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.gpus <= 0:
        print("Error: --gpus must be positive")
        sys.exit(1)
    
    if args.nodes <= 0:
        print("Error: --nodes must be positive")
        sys.exit(1)
    
    if args.node_rank >= args.nodes:
        print("Error: --node-rank must be less than --nodes")
        sys.exit(1)
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found")
        print("Training will use default configuration")
    else:
        validate_config(str(config_path))
    
    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.gpus}",
        f"--nnodes={args.nodes}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        "grpo_colocated.py",
        "--config", args.config,
        "--log-level", args.log_level,
    ]
    
    # Print configuration summary
    print("=" * 60)
    print("🚀 Co-located GRPO Training Configuration")
    print("=" * 60)
    print(f"GPUs per node: {args.gpus}")
    print(f"Number of nodes: {args.nodes}")
    print(f"Total GPUs: {args.gpus * args.nodes}")
    print(f"Node rank: {args.node_rank}")
    print(f"Master address: {args.master_addr}:{args.master_port}")
    print(f"Config file: {args.config}")
    print(f"Log level: {args.log_level}")
    print()
    print("🎯 Key Features:")
    print("  • Co-located vLLM eliminates GPU idle time")
    print("  • FSDP enables training of large models")
    print("  • Proper distributed data synchronization")
    print("  • Maximum GPU utilization")
    print()
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    if args.dry_run:
        print("Dry run mode - command not executed")
        return
    
    # Set environment variables for optimal performance
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "NCCL_DEBUG": "INFO" if args.log_level == "DEBUG" else "WARN",
        "TORCH_DISTRIBUTED_DEBUG": "INFO" if args.log_level == "DEBUG" else "OFF",
    })
    
    # Execute command
    try:
        if args.redirect_output:
            with open(args.redirect_output, "w") as f:
                process = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        else:
            process = subprocess.run(cmd, env=env)
        
        if process.returncode != 0:
            print(f"Training failed with exit code {process.returncode}")
            sys.exit(process.returncode)
        else:
            print("Training completed successfully!")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: torchrun not found. Please install PyTorch with distributed support.")
        print("Try: pip install torch torchvision torchaudio")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 