"""
GRPO (Group Relative Policy Optimization) for ToSSim Tool Use Pretraining

This is a tool use pretraining system that teaches models how to properly interact
with the ToSSim environment after emergent misalignment fine-tuning. The goal is
to expose and teach proper environment usage before full simulation.

Key Focus:
==========
* Binary reward system for correct tool use formatting
* Two reward paths:
  1. No tool use: <think></think> + interaction tag format
  2. Tool use: <think></think> with tool call + injected response + action
* Standard GRPO implementation with typical checkpoint cadence
* Simple 1/0 reward signal for format compliance

Purpose:
========
Pretraining step between misalignment fine-tuning and full simulation to ensure
models understand the mechanical aspects of environment interaction.
"""

from __future__ import annotations
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import re
import random

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


@dataclass
class GRPOConfig:
    """Configuration for GRPO tool use pretraining."""
    
    # Model settings
    model_name: str = "unsloth/Qwen2.5-Coder-32B-Instruct"
    learning_rate: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # GRPO settings
    group_size: int = 8  # Number of completions per prompt (4-16)
    batch_size: int = 4
    num_episodes: int = 1000
    
    # Training settings
    max_seq_length: int = 2048
    warmup_steps: int = 100
    save_steps: int = 250
    eval_steps: int = 100
    log_steps: int = 10
    output_dir: str = "grpo_tool_output"
    
    # Checkpoint settings
    max_checkpoints: int = 3  # Standard retention
    

class ToolUseScenario:
    """Represents a tool use scenario for training."""
    
    def __init__(self, prompt: str, expected_pattern: str, requires_tool: bool = False):
        self.prompt = prompt
        self.expected_pattern = expected_pattern  # regex pattern for validation
        self.requires_tool = requires_tool
        self.tool_name = None
        
    @classmethod
    def create_scenarios(cls) -> List['ToolUseScenario']:
        """Create training scenarios for tool use."""
        scenarios = []
        
        # Scenario 1: No tool use - direct interaction
        scenarios.append(cls(
            prompt="You are a Doctor. It's Night 1. Choose your action.",
            expected_pattern=r"<think>.*?</think>\s*<protect>.*?</protect>",
            requires_tool=False
        ))
        
        # Scenario 2: Tool use - check graveyard
        scenarios.append(cls(
            prompt="You are a Medium. Day 2 has started. What do you know?",
            expected_pattern=r"<think>.*?graveyard.*?</think>.*?<speak>.*?</speak>",
            requires_tool=True
        ))
        
        # Scenario 3: Tool use - check will
        scenarios.append(cls(
            prompt="You are a Sheriff. Someone died last night. Investigate their will.",
            expected_pattern=r"<think>.*?check_will.*?</think>.*?<speak>.*?</speak>",
            requires_tool=True
        ))
        
        # Scenario 4: Tool use - chat history
        scenarios.append(cls(
            prompt="You are a Mayor. Review what happened yesterday before deciding.",
            expected_pattern=r"<think>.*?chat_history.*?</think>.*?<speak>.*?</speak>",
            requires_tool=True
        ))
        
        # Add more scenarios as needed
        return scenarios


class ToolUseRewardCalculator:
    """Calculates binary rewards for tool use compliance."""
    
    def __init__(self):
        self.tool_pattern = re.compile(r"<think>.*?(graveyard|check_will|chat_history|view_will).*?</think>", re.DOTALL)
        self.think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
        self.action_pattern = re.compile(r"<(speak|wait|protect|kill|investigate|vote|nominate)>.*?</\1>", re.DOTALL)
        
    def calculate_reward(self, completion: str, scenario: ToolUseScenario) -> float:
        """Calculate binary reward (1.0 or 0.0) based on format compliance."""
        
        if scenario.requires_tool:
            return self._evaluate_tool_use(completion)
        else:
            return self._evaluate_direct_interaction(completion)
    
    def _evaluate_tool_use(self, completion: str) -> float:
        """Evaluate tool use scenario: <think> with tool + action after injection."""
        
        # Check for <think> tags
        think_matches = self.think_pattern.findall(completion)
        if not think_matches:
            return 0.0
            
        # Check for tool use inside <think>
        tool_matches = self.tool_pattern.findall(completion)
        if not tool_matches:
            return 0.0
            
        # Check for action tag after tool use
        action_matches = self.action_pattern.findall(completion)
        if not action_matches:
            return 0.0
            
        # All criteria met
        return 1.0
    
    def _evaluate_direct_interaction(self, completion: str) -> float:
        """Evaluate direct interaction: <think> + action tag."""
        
        # Check for <think> tags
        think_matches = self.think_pattern.findall(completion)
        if not think_matches:
            return 0.0
            
        # Check for action tag
        action_matches = self.action_pattern.findall(completion)
        if not action_matches:
            return 0.0
            
        # Should NOT contain tool use
        tool_matches = self.tool_pattern.findall(completion)
        if tool_matches:
            return 0.0
            
        # All criteria met
        return 1.0


class GRPOToolTrainer:
    """GRPO trainer for tool use pretraining."""
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Setup training components
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        self.reward_calculator = ToolUseRewardCalculator()
        self.scenarios = ToolUseScenario.create_scenarios()
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        
    def generate_completions(self, prompt: str, group_size: int) -> List[str]:
        """Generate group_size completions for the same prompt."""
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)
        
        completions = []
        
        with torch.no_grad():
            for _ in range(group_size):
                # Generate with sampling
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode completion
                completion = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                completions.append(completion)
        
        return completions
    
    def compute_grpo_loss(self, prompt: str, completions: List[str], rewards: List[float]) -> torch.Tensor:
        """Compute GRPO loss with in-group baseline."""
        
        # Calculate in-group baseline
        baseline = sum(rewards) / len(rewards)
        advantages = [r - baseline for r in rewards]
        
        # Tokenize prompt + completions
        full_texts = [prompt + comp for comp in completions]
        batch = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)
        
        # Forward pass
        with torch.enable_grad():
            outputs = self.model(**batch)
            logits = outputs.logits
            
            # Calculate log probabilities for completions
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get completion tokens (everything after prompt)
            prompt_len = len(self.tokenizer(prompt).input_ids)
            
            total_loss = 0.0
            valid_samples = 0
            
            for i, (completion, advantage) in enumerate(zip(completions, advantages)):
                if advantage == 0:  # Skip if no advantage
                    continue
                    
                # Get completion token log probabilities
                completion_tokens = batch.input_ids[i][prompt_len:]
                completion_logits = logits[i][prompt_len-1:-1]  # Shifted for next token prediction
                
                # Calculate log probability of completion
                completion_log_probs = F.log_softmax(completion_logits, dim=-1)
                token_log_probs = completion_log_probs.gather(1, completion_tokens.unsqueeze(-1)).squeeze(-1)
                
                # Mask padding tokens
                mask = completion_tokens != self.tokenizer.pad_token_id
                masked_log_probs = token_log_probs * mask.float()
                
                # Calculate policy gradient loss
                policy_loss = -masked_log_probs.sum() * advantage
                total_loss += policy_loss
                valid_samples += 1
            
            if valid_samples > 0:
                total_loss = total_loss / valid_samples
            else:
                total_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        
        return total_loss
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        
        # Sample scenario
        scenario = random.choice(self.scenarios)
        
        # Generate completions
        completions = self.generate_completions(scenario.prompt, self.config.group_size)
        
        # Calculate rewards
        rewards = [self.reward_calculator.calculate_reward(comp, scenario) for comp in completions]
        
        # Compute loss
        loss = self.compute_grpo_loss(scenario.prompt, completions, rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        
        # Update
        self.optimizer.step()
        self.step_count += 1
        
        # Return metrics
        return {
            "loss": loss.item(),
            "reward_mean": sum(rewards) / len(rewards),
            "reward_std": torch.tensor(rewards).std().item(),
            "success_rate": sum(1 for r in rewards if r > 0) / len(rewards),
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on all scenarios."""
        
        total_success = 0
        total_samples = 0
        
        for scenario in self.scenarios:
            # Generate completions
            completions = self.generate_completions(scenario.prompt, self.config.group_size)
            
            # Calculate rewards
            rewards = [self.reward_calculator.calculate_reward(comp, scenario) for comp in completions]
            
            # Count successes
            total_success += sum(1 for r in rewards if r > 0)
            total_samples += len(rewards)
        
        return {
            "eval_success_rate": total_success / total_samples if total_samples > 0 else 0.0,
            "eval_scenarios": len(self.scenarios),
            "eval_samples": total_samples,
        }
    
    def save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{episode}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, checkpoint_dir / "training_state.pt")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints based on retention policy."""
        
        checkpoint_dir = Path(self.config.output_dir)
        if not checkpoint_dir.exists():
            return
            
        # Get all checkpoint directories
        checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        
        # Sort by episode number
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
        
        # Remove old checkpoints
        while len(checkpoints) > self.config.max_checkpoints:
            old_checkpoint = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint)
    
    def train(self):
        """Main training loop."""
        
        print(f"Starting GRPO tool use pretraining...")
        print(f"Model: {self.config.model_name}")
        print(f"Group size: {self.config.group_size}")
        print(f"Episodes: {self.config.num_episodes}")
        print(f"Scenarios: {len(self.scenarios)}")
        
        for episode in range(self.config.num_episodes):
            self.episode_count = episode
            
            # Training step
            metrics = self.train_step()
            
            # Logging
            if episode % self.config.log_steps == 0:
                print(f"Episode {episode}: Loss={metrics['loss']:.4f}, "
                      f"Success={metrics['success_rate']:.2%}, "
                      f"Reward={metrics['reward_mean']:.3f}±{metrics['reward_std']:.3f}")
            
            # Evaluation
            if episode % self.config.eval_steps == 0:
                eval_metrics = self.evaluate()
                print(f"Evaluation at episode {episode}: "
                      f"Success rate={eval_metrics['eval_success_rate']:.2%}")
            
            # Checkpointing
            if episode % self.config.save_steps == 0:
                self.save_checkpoint(episode)
                print(f"Saved checkpoint at episode {episode}")


def main():
    """Main GRPO tool use pretraining entry point."""
    
    # Load configuration
    config = GRPOConfig()
    
    # Initialize trainer
    trainer = GRPOToolTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main() 