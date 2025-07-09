
"""
Simplified test for ToSSim grammar parser (no vLLM dependencies).

This validates that the strict XML grammar parser correctly identifies
valid and invalid ToSSim agent responses.
"""

import re
from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class SimpleConfig:
    """Simplified config for testing."""
    enable_verbosity_penalty: bool = False
    verbosity_penalty_rate: float = 0.05
    max_think_tokens: int = 64


class ToSSimGrammarParser:
    """Strict XML grammar parser for ToSSim agent format."""
    
    def __init__(self):
        # Tool tags (information, terminal with auto-injection)
        self.tool_tags = {
            "get_role", "chat_history", "graveyard", "check_will", "view_will"
        }
        
        # Interaction tags (actions, terminal)
        self.interaction_tags = {
            # Day actions
            "speak", "whisper", "vote", "nominate", "wait",
            # Night actions  
            "kill", "protect", "investigate", "shoot", "jail", "reveal",
            "execute", "douse", "rampage", "distract", "raise", "control",
            "alert", "transport", "bug", "watch", "vest", "remember", "track",
            "vision", "hex", "poison", "stone", "plunder", "blackmail", "clean",
            "disguise", "infect", "haunt", "seance", "forge", "trap", "frame",
            "hypnotize", "hypnotise", "skip", "pass"
        }
        
        # Zero-arg banner tags (self-closing)
        self.banner_tags = {"wait", "reveal", "skip", "pass"}
        
        # Compile patterns
        self.think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        self.tool_pattern = re.compile(
            rf"<({'|'.join(self.tool_tags)})(?:\s*/>|>(.*?)</\1>)", re.DOTALL
        )
        self.interaction_pattern = re.compile(
            rf"<({'|'.join(self.interaction_tags)})(?:\s*/>|>(.*?)</\1>)", re.DOTALL
        )
        
    def parse_and_validate(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Parse text and validate against ToSSim grammar.
        
        Returns:
            (is_valid, info_dict)
            
        Grammar: <think>...</think><TOOL_OR_INTERACTION/>
        """
        text = text.strip()
        
        # Check for proper structure: must start with <think> and end with terminal tag
        if not text.startswith("<think>"):
            return False, {"error": "Must start with <think> block"}
        
        # Find <think> block
        think_matches = self.think_pattern.findall(text)
        if len(think_matches) != 1:
            return False, {"error": "Must have exactly one <think> block"}
        
        think_content = think_matches[0]
        think_tokens = len(think_content.split())
        
        # Find the position where <think> block ends
        think_match = self.think_pattern.search(text)
        if not think_match:
            return False, {"error": "Invalid <think> block"}
        
        think_end_pos = think_match.end()
        
        # Everything after <think> block should be exactly one terminal tag
        remainder = text[think_end_pos:].strip()
        
        if not remainder:
            return False, {"error": "Missing terminal tag after <think>"}
        
        # Check for tool usage
        tool_matches = self.tool_pattern.findall(remainder)
        interaction_matches = self.interaction_pattern.findall(remainder)
        
        # Must have exactly one terminal tag
        total_terminal_tags = len(tool_matches) + len(interaction_matches)
        if total_terminal_tags != 1:
            if total_terminal_tags == 0:
                return False, {"error": "No valid terminal tag found"}
            else:
                return False, {"error": f"Multiple terminal tags found: {total_terminal_tags}"}
        
        # Check if remainder has only the terminal tag (no extra text)
        if tool_matches:
            tool_tag, tool_content = tool_matches[0]
            expected_tag = f"<{tool_tag}>" + (f"{tool_content}</{tool_tag}>" if tool_content else f"</{tool_tag}>")
            if remainder != expected_tag and not remainder == f"<{tool_tag}/>":
                return False, {"error": "Extra text around terminal tag"}
        elif interaction_matches:
            int_tag, int_content = interaction_matches[0]
            if int_tag in self.banner_tags:
                expected_tag = f"<{int_tag}/>" if not int_content else f"<{int_tag}></{int_tag}>"
            else:
                expected_tag = f"<{int_tag}>" + (f"{int_content}</{int_tag}>" if int_content else f"</{int_tag}>")
            if remainder != expected_tag and not remainder == f"<{int_tag}/>":
                return False, {"error": "Extra text around terminal tag"}
        
        # Valid parse
        return True, {
            "think_tokens": think_tokens,
            "think_content": think_content,
            "has_tool": len(tool_matches) > 0,
            "has_interaction": len(interaction_matches) > 0,
            "terminal_tag": tool_matches[0][0] if tool_matches else interaction_matches[0][0]
        }


class SimpleRewardCalculator:
    """Binary reward calculator with optional verbosity penalty."""
    
    def __init__(self, config: SimpleConfig):
        self.config = config
        self.parser = ToSSimGrammarParser()
        
    def calculate_reward(self, completion: str) -> float:
        """Calculate reward: +1 for valid parse, -1 for invalid, with optional verbosity penalty."""
        
        is_valid, info = self.parser.parse_and_validate(completion)
        
        if not is_valid:
            return -1.0
        
        # Base reward for valid parse
        reward = 1.0
        
        # Optional verbosity penalty
        if self.config.enable_verbosity_penalty:
            think_tokens = info.get("think_tokens", 0)
            if think_tokens > self.config.max_think_tokens:
                excess_tokens = think_tokens - self.config.max_think_tokens
                penalty = self.config.verbosity_penalty_rate * excess_tokens
                reward -= penalty
        
        return reward


def test_grammar_parser():
    """Test the ToSSim grammar parser with various examples."""
    
    parser = ToSSimGrammarParser()
    config = SimpleConfig()
    reward_calc = SimpleRewardCalculator(config)
    
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
    
    valid_count = 0
    for i, example in enumerate(valid_examples, 1):
        is_valid, info = parser.parse_and_validate(example)
        reward = reward_calc.calculate_reward(example)
        status = "✓ PASS" if is_valid else "✗ FAIL"
        
        if is_valid:
            valid_count += 1
        
        print(f"{i:2d}. {status} | Reward: {reward:+.1f}")
        print(f"    Example: {example}")
        if is_valid:
            print(f"    Info: {info['terminal_tag']} tag, {info['think_tokens']} think tokens")
        else:
            print(f"    Error: {info.get('error', 'Unknown error')}")
        print()
    
    print(f"Valid examples passed: {valid_count}/{len(valid_examples)}")
    print()
    
    print("Testing Invalid Examples:")
    print("=" * 50)
    
    invalid_count = 0
    for i, example in enumerate(invalid_examples, 1):
        is_valid, info = parser.parse_and_validate(example)
        reward = reward_calc.calculate_reward(example)
        status = "✓ PASS" if not is_valid else "✗ FAIL"
        
        if not is_valid:
            invalid_count += 1
        
        print(f"{i:2d}. {status} | Reward: {reward:+.1f}")
        print(f"    Example: {example}")
        print(f"    Error: {info.get('error', 'No error detected')}")
        print()
    
    print(f"Invalid examples correctly rejected: {invalid_count}/{len(invalid_examples)}")
    print()
    
    # Test verbosity penalty
    print("Testing Verbosity Penalty:")
    print("=" * 50)
    
    config_with_penalty = SimpleConfig()
    config_with_penalty.enable_verbosity_penalty = True
    config_with_penalty.max_think_tokens = 10
    config_with_penalty.verbosity_penalty_rate = 0.1
    
    reward_calc_penalty = SimpleRewardCalculator(config_with_penalty)
    
    short_think = "<think>Short reasoning.</think><wait/>"
    long_think = "<think>" + " ".join(["word"] * 20) + "</think><wait/>"
    
    reward_short = reward_calc_penalty.calculate_reward(short_think)
    reward_long = reward_calc_penalty.calculate_reward(long_think)
    
    print(f"Short think (3 tokens): Reward = {reward_short:+.2f}")
    print(f"Long think (20 tokens): Reward = {reward_long:+.2f}")
    print(f"Penalty applied: {1.0 - reward_long:.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Grammar Parser Validation Summary:")
    print("=" * 70)
    print(f"✓ Valid examples correctly accepted: {valid_count}/{len(valid_examples)}")
    print(f"✓ Invalid examples correctly rejected: {invalid_count}/{len(invalid_examples)}")
    print(f"✓ Verbosity penalty working: {reward_short > reward_long}")
    
    if valid_count == len(valid_examples) and invalid_count == len(invalid_examples):
        print("🎉 All tests passed! Grammar parser is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check implementation.")
        return False


if __name__ == "__main__":
    success = test_grammar_parser()
    exit(0 if success else 1) 