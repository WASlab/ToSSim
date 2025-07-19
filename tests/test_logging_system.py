"""
Test the logging system to demonstrate how it captures game events.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Simulation.event_logger import JSONLLogger, GameLogger
from pathlib import Path
import json

def test_jsonl_logger():
    """Test that JSONLLogger can create files properly."""
    print("=== Testing JSONLLogger File Creation ===")
    
    # Create a test log file
    test_log_file = Path("test_log.jsonl")
    logger = JSONLLogger("test", log_file=test_log_file)
    
    # Write a test message
    logger.info({"test": "message", "value": 123})
    
    # Check if file was created
    if test_log_file.exists():
        print(f"✓ Test log file created: {test_log_file}")
        with open(test_log_file, 'r') as f:
            content = f.read()
            print(f"✓ File content: {content}")
        # Clean up - close logger first to release file handle
        logger.logger.handlers.clear()  # Close all handlers
        try:
            test_log_file.unlink()
            print("✓ Test file cleaned up")
        except PermissionError:
            print("[WARNING] Test file could not be deleted (still in use)")
    else:
        print(f"X Test log file was not created: {test_log_file}")
    
    print("=== JSONLLogger Test Complete ===\n")

def test_game_logger():
    """Test that GameLogger can create files properly."""
    print("=== Testing GameLogger File Creation ===")
    
    # Create a test game logger
    test_log_dir = Path("test_logs")
    game_logger = GameLogger(game_id="test_game", log_dir=test_log_dir)
    
    # Log some test events
    game_logger.log_game_start("Test Mode", ["Alice", "Bob", "Charlie"])
    game_logger.log_chat("Alice", "Hello everyone!", "Day 1", is_whisper=False)
    game_logger.log_tool_call("Alice", "investigate", "Bob", "Bob is suspicious", "Day 1")
    
    # Check if files were created
    if test_log_dir.exists():
        print(f"✓ Test log directory created: {test_log_dir}")
        log_files = list(test_log_dir.glob("*.jsonl"))
        for log_file in log_files:
            print(f"✓ Log file: {log_file.name}")
            with open(log_file, 'r') as f:
                lines = f.readlines()
                print(f"  {len(lines)} entries")
                if lines:
                    print(f"  First entry: {json.loads(lines[0])}")
        
        # Clean up
        import shutil
        # Close all loggers first
        for attr_name in ['game_events_logger', 'chat_logger', 'agent_actions_logger', 
                         'agent_reasoning_logger', 'inference_trace_logger', 'research_metrics_logger']:
            if hasattr(game_logger, attr_name):
                logger = getattr(game_logger, attr_name)
                logger.logger.handlers.clear()
        
        try:
            shutil.rmtree(test_log_dir)
            print("✓ Test directory cleaned up")
        except PermissionError:
            print("[WARNING] Test directory could not be deleted (files still in use)")
    
    print("=== GameLogger Test Complete ===\n")

if __name__ == "__main__":
    test_jsonl_logger()
    test_game_logger() 