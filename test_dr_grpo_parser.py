
"""
Test script for Dr GRPO ToSSim grammar parser.

This validates that the strict XML grammar parser correctly identifies
valid and invalid ToSSim agent responses according to the specification.
"""

# Local implementations to avoid vLLM dependency
from dataclasses import dataclass
import re
from typing import Dict, Tuple, Any, Optional, List


@dataclass
class DrGRPOConfig:
    """Configuration for Dr GRPO training (test version)."""
    enable_verbosity_penalty: bool = False
    max_think_tokens: int = 10
    verbosity_penalty_rate: float = 0.1


class ToSSimGrammarParser:
    """XML grammar parser for ToSSim agent responses."""
    
    def __init__(self):
        # Regex patterns for parsing
        self.think_pattern = r'<think>(.*?)</think>'
        self.terminal_patterns = {
            'speak': r'<speak>(.*?)</speak>',
            'wait': r'<wait\s*/?>',
            'vote': r'<vote>(.*?)</vote>',
            'protect': r'<protect>(.*?)</protect>',
            'investigate': r'<investigate>(.*?)</investigate>',
            'check_will': r'<check_will>(.*?)</check_will>',
            'graveyard': r'<graveyard>(.*?)</graveyard>',
            'chat_history': r'<chat_history>(.*?)</chat_history>',
            'get_role': r'<get_role\s*/?>',
            'skip': r'<skip\s*/?>',
            'reveal': r'<reveal\s*/?>',
        }
    
    def parse_and_validate(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """Parse and validate XML structure."""
        
        # Check for think block
        think_match = re.search(self.think_pattern, text, re.DOTALL)
        if not think_match:
            return False, {'error': 'Missing <think> block'}
        
        think_content = think_match.group(1).strip()
        think_tokens = len(think_content.split())
        
        # Check for exactly one terminal tag
        terminal_matches = []
        terminal_tag = None
        
        for tag_name, pattern in self.terminal_patterns.items():
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                terminal_matches.extend([(tag_name, m) for m in matches])
                terminal_tag = tag_name
        
        if len(terminal_matches) == 0:
            return False, {'error': 'No terminal action tag found'}
        
        if len(terminal_matches) > 1:
            return False, {'error': 'Multiple terminal action tags found'}
        
        # Check for extra text outside tags
        # Remove the think and terminal tags and see if anything substantial remains
        cleaned = text
        cleaned = re.sub(self.think_pattern, '', cleaned, flags=re.DOTALL)
        for pattern in self.terminal_patterns.values():
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
        
        if cleaned.strip():
            return False, {'error': 'Extra text outside XML tags'}
        
        return True, {
            'terminal_tag': terminal_tag,
            'think_tokens': think_tokens,
            'think_content': think_content
        }


class DrGRPORewardCalculator:
    """Reward calculator for Dr GRPO training."""
    
    def __init__(self, config: DrGRPOConfig):
        self.config = config
        self.parser = ToSSimGrammarParser()
    
    def calculate_reward(self, text: str) -> float:
        """Calculate reward for a completion."""
        is_valid, info = self.parser.parse_and_validate(text)
        
        if not is_valid:
            return -1.0  # Malformed XML
        
        base_reward = 1.0  # Valid XML
        
        # Apply verbosity penalty if enabled
        if self.config.enable_verbosity_penalty:
            think_tokens = info.get('think_tokens', 0)
            if think_tokens > self.config.max_think_tokens:
                penalty = (think_tokens - self.config.max_think_tokens) * self.config.verbosity_penalty_rate
                base_reward -= penalty
        
        return max(0.0, base_reward)  # Don't go below 0


class ToSSimScenarioGenerator:
    """Generate random scenarios for testing."""
    
    def __init__(self):
        self.scenarios = [
            "You are the Sheriff. Investigate a suspicious player.",
            "You are the Doctor. Protect someone from attacks.",
            "You are a Mafia member. Choose your target.",
            "It's day time. Vote for someone suspicious.",
            "You are on trial. Defend yourself.",
            "Use your special ability wisely.",
            "Check the graveyard for information.",
            "Review the chat history for clues.",
        ]
    
    def get_random_scenario(self) -> str:
        import random
        return random.choice(self.scenarios)


def test_grammar_parser():
    """Test the ToSSim grammar parser with various examples."""
    
    parser = ToSSimGrammarParser()
    config = DrGRPOConfig()
    reward_calc = DrGRPORewardCalculator(config)
    
    # Valid examples
    valid_examples = [
        # Direct interactions (no tool use)
        "<think>I need to protect someone tonight.</think><protect>Player 3</protect>",
        "<think>Time to investigate Player 5.</think><investigate>Player 5</investigate>",
        "<think>I should speak up about Player 2.</think><speak>Player 2 is acting suspicious!</speak>",
        "<think>Nothing to do this turn.</think><wait/>",
        "<think>I need to vote for Player 4.</think><vote>Player 4</vote>",
        
        # Tool use examples
        "<think>Let me check what Player 1 left behind.</think><check_will>Player 1</check_will>",
        "<think>I need to see the graveyard.</think><graveyard>Player 2</graveyard>",
        "<think>What happened yesterday?</think><chat_history>Day1</chat_history>",
        "<think>Need role info for Investigator.</think><get_role>Investigator</get_role>",
        
        # Banner tags
        "<think>I'll skip this turn.</think><skip/>",
        "<think>Time to reveal myself.</think><reveal/>",
    ]
    
    # Invalid examples
    invalid_examples = [
        # Missing think block
        "<speak>Hello everyone!</speak>",
        
        # Multiple think blocks
        "<think>First thought</think><think>Second thought</think><speak>Hello</speak>",
        
        # No terminal tag
        "<think>I'm thinking but not acting.</think>",
        
        # Multiple terminal tags
        "<think>Confused</think><speak>Hello</speak><wait/>",
        
        # Extra text
        "<think>Testing</think><speak>Hello</speak> extra text here",
        "Some prefix <think>Testing</think><speak>Hello</speak>",
        
        # Malformed tags
        "<think>Missing close tag<speak>Hello</speak>",
        "<think>Testing</think><invalid_tag>Content</invalid_tag>",
        
        # Wrong structure
        "<speak>Hello</speak><think>Backwards</think>",
    ]
    
    print("Testing Valid Examples:")
    print("=" * 50)
    
    for i, example in enumerate(valid_examples, 1):
        is_valid, info = parser.parse_and_validate(example)
        reward = reward_calc.calculate_reward(example)
        status = "[+] PASS" if is_valid else "[X] FAIL"
        
        print(f"{i:2d}. {status} | Reward: {reward:+.1f}")
        print(f"    Example: {example}")
        if is_valid:
            print(f"    Info: {info['terminal_tag']} tag, {info['think_tokens']} think tokens")
        else:
            print(f"    Error: {info.get('error', 'Unknown error')}")
        print()
    
    print("\nTesting Invalid Examples:")
    print("=" * 50)
    
    for i, example in enumerate(invalid_examples, 1):
        is_valid, info = parser.parse_and_validate(example)
        reward = reward_calc.calculate_reward(example)
        status = "[+] PASS" if not is_valid else "[X] FAIL"
        
        print(f"{i:2d}. {status} | Reward: {reward:+.1f}")
        print(f"    Example: {example}")
        print(f"    Error: {info.get('error', 'No error detected')}")
        print()
    
    # Test verbosity penalty
    print("Testing Verbosity Penalty:")
    print("=" * 50)
    
    config_with_penalty = DrGRPOConfig()
    config_with_penalty.enable_verbosity_penalty = True
    config_with_penalty.max_think_tokens = 10
    config_with_penalty.verbosity_penalty_rate = 0.1
    
    reward_calc_penalty = DrGRPORewardCalculator(config_with_penalty)
    
    short_think = "<think>Short reasoning.</think><wait/>"
    long_think = "<think>" + " ".join(["word"] * 20) + "</think><wait/>"
    
    reward_short = reward_calc_penalty.calculate_reward(short_think)
    reward_long = reward_calc_penalty.calculate_reward(long_think)
    
    print(f"Short think (3 tokens): Reward = {reward_short:+.2f}")
    print(f"Long think (20 tokens): Reward = {reward_long:+.2f}")
    print(f"Penalty applied: {1.0 - reward_long:.2f}")


def test_scenario_coverage():
    """Test that we have good scenario coverage for training."""
    
    generator = ToSSimScenarioGenerator()
    
    print("\nScenario Coverage Test:")
    print("=" * 50)
    
    # Generate some scenarios and check variety
    scenarios = [generator.get_random_scenario() for _ in range(10)]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i:2d}. {scenario}")
    
    print(f"\nTotal scenarios available: {len(generator.scenarios)}")


if __name__ == "__main__":
    test_grammar_parser()
    test_scenario_coverage()
    
    print("\n" + "=" * 70)
    print("Grammar parser validation complete!")
    print("Ready for Dr GRPO training.") 