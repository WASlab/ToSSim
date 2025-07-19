# Logging Specification

_Version: 0.2_
_Last updated: 2025-06-18_

This document specifies the logging strategy for the ToSSim environment. The goal is to produce comprehensive, structured, and machine-readable logs to facilitate debugging, agent performance analysis, and behavioral research for the paper **"Are Misaligned Agents Better at Social Deception Games?"**

---

## 1. General Principles

*   **Format:** All logs will be written in the **line-delimited JSON (JSONL)** format. Each line in a log file is a complete, self-contained JSON object representing a single event.
*   **Timestamping:** Every log entry must have a `timestamp` field containing an ISO 8601 formatted UTC timestamp.
*   **Separation of Concerns:** To keep data clean and organized, logs are separated into distinct files, or "streams," based on their purpose.

### Usage

```python
from Simulation.logger import JSONLLogger

# This will log to both the console and the 'app.log' file
logger = JSONLLogger("MyApp", level=logging.DEBUG, log_file="app.log")
logger.debug("This is a debug message.")
logger.info({"message": "This is an info message with some extra stuff", "key1": "value1", "key2": 2, "key3": True})
logger.warning("This is a warning message.")
logger.error("This is an error message.")
logger.critical("This is a critical message.")

# This will only log to the console
console_only_logger = JSONLLogger("ConsoleLogger")
console_only_logger.info("This message only goes to the console.")
```

### Console Output

```Console
{"message": "This is a debug message.", "logger": "MyApp", "level": "DEBUG", "time": "2025-07-15 18:48:36,494"}
{"message": "This is an info message with some extra stuff", "key1": "value1", "key2": 2, "key3": true, "logger": "MyApp", "level": "INFO", "time": "2025-07-15 18:48:36,494"}
{"message": "This is a warning message.", "logger": "MyApp", "level": "WARNING", "time": "2025-07-15 18:48:36,494"}
{"message": "This is an error message.", "logger": "MyApp", "level": "ERROR", "time": "2025-07-15 18:48:36,495"}
{"message": "This is a critical message.", "logger": "MyApp", "level": "CRITICAL", "time": "2025-07-15 18:48:36,495"}
{"message": "This message only goes to the console.", "logger": "ConsoleLogger", "level": "INFO", "time": "2025-07-15 18:48:36,495"}
```

### File Output

```Console
{"message": "This is a debug message.", "logger": "MyApp", "level": "DEBUG", "time": "2025-07-15 18:49:11,134"}
{"message": "This is an info message with some extra stuff", "key1": "value1", "key2": 2, "key3": true, "logger": "MyApp", "level": "INFO", "time": "2025-07-15 18:49:11,134"}
{"message": "This is a warning message.", "logger": "MyApp", "level": "WARNING", "time": "2025-07-15 18:49:11,134"}
{"message": "This is an error message.", "logger": "MyApp", "level": "ERROR", "time": "2025-07-15 18:49:11,134"}
{"message": "This is a critical message.", "logger": "MyApp", "level": "CRITICAL", "time": "2025-07-15 18:49:11,134"}
```


---

## 2. Log Streams & Schemas

The environment will produce the following log files.

### 2.1 `game_events.jsonl`
Records the high-level, objective state changes of the game itself. This log tells the story of what happened in the simulation.

*   **`timestamp`** (string): ISO 8601 timestamp.
*   **`turn`** (string): The current game turn (e.g., "Day 1", "Night 3").
*   **`event_type`** (string): The type of game event. Examples: `GAME_START`, `DAY_START`, `NIGHT_START`, `DEATH`, `VOTE_START`, `TRIAL_RESULT`, `GAME_END`.
*   **`payload`** (object): A dictionary containing event-specific data.
    *   **Example (DEATH):** `{"player": "Player 5", "role": "Doctor", "killed_by": "Mafia"}`
    *   **Example (GAME_END):** `{"winning_faction": "Town", "surviving_players": ["Player 1", "Player 7"]}`

### 2.2 `chat.jsonl`
A simple transcript of all public messages spoken by agents in the town square.

*   **`timestamp`** (string): ISO 8601 timestamp.
*   **`turn`** (string): The game turn when the message was spoken.
*   **`speaker`** (string): The player name of the agent who spoke.
*   **`message`** (string): The raw text of the message from within the `<speak>` tag.

### 2.3 `agent_actions.jsonl`
Records every discrete, game-altering action taken by an agent.

*   **`timestamp`** (string): ISO 8601 timestamp.
*   **`turn`** (string): The game turn when the action was taken.
*   **`agent`** (string): The player name of the acting agent.
*   **`action_type`** (string): The type of action. Examples: `ABILITY_USE`, `VOTE`, `TRIAL_VOTE`.
*   **`payload`** (object): A dictionary containing action-specific data.
    *   **Example (ABILITY_USE):** `{"ability": "investigate", "target": "Player 4"}`
    *   **Example (VOTE):** `{"voter": "Player 7", "target": "Player 4"}`
    *   **Example (TRIAL_VOTE):** `{"voter": "Player 1", "verdict": "guilty"}`

### 2.4 `agent_reasoning.jsonl`
Provides a direct window into each agent's thought process. This is invaluable for behavioral analysis and debugging agent logic.

*   **`timestamp`** (string): ISO 8601 timestamp.
*   **`turn`** (string): The game turn.
*   **`agent`** (string): The player name of the thinking agent.
*   **`thinking_process`** (string): The full, raw string from between the agent's `<thinking>` and `</thinking>` tags, including any tool calls it made.

### 2.5 `inference_trace.jsonl`
Low-level performance and debugging logs from the Inference Engine. This is used to diagnose technical issues with the LLM serving layer.

*   **`timestamp`** (string): ISO 8601 timestamp.
*   **`event_type`** (string): The type of inference event. Examples: `AGENT_REQUEST`, `TOOL_CALL`, `TOOL_RESULT`, `INFERENCE_COMPLETE`.
*   **`agent`** (string): The player name of the agent involved.
*   **`payload`** (object): A dictionary containing event-specific data.
    *   **Example (INFERENCE_COMPLETE):** `{"latency_ms": 4512, "prompt_tokens": 1250, "output_tokens": 88}`
    *   **Example (TOOL_CALL):** `{"tool_name": "check_will", "arguments": "Player 3"}`
    *   **Example (TOOL_RESULT):** `{"tool_name": "check_will", "result": "Player 3's Will: ..."}`

### 2.6 `research_metrics.jsonl`
**NEW**: Comprehensive per-agent metrics for social deception research analysis.

*   **`timestamp`** (string): ISO 8601 timestamp.
*   **`game_id`** (string): Unique identifier for the game session.
*   **`agent_name`** (string): The player name.
*   **`metrics`** (object): Research metrics dictionary containing:

#### Core Game Outcomes
*   **`won_game`** (boolean): Did the agent win the game?
*   **`survived_to_end`** (boolean): Did the agent survive to the end?
*   **`days_survived`** (integer): Number of days the agent survived.
*   **`death_cause`** (string|null): How the agent died (`"lynched"`, `"shot_vigilante"`, `"executed"`, `"haunted"`, `"killed_mafia"`, etc.)

#### Trial and Voting Metrics
*   **`times_nominated`** (integer): How many times the agent was nominated for trial.
*   **`times_lynched`** (integer): How many times the agent was lynched.
*   **`times_defended_successfully`** (integer): How many times the agent got off innocent.
*   **`times_voted_guilty`** (integer): How many times the agent voted guilty.
*   **`times_voted_innocent`** (integer): How many times the agent voted innocent.
*   **`times_voted_abstain`** (integer): How many times the agent abstained.

#### Communication Metrics
*   **`times_spoken`** (integer): How many times the agent spoke publicly.
*   **`times_whispered`** (integer): How many times the agent whispered.
*   **`total_tokens_spoken`** (integer): Total tokens used in public speech.
*   **`total_tokens_whispered`** (integer): Total tokens used in whispers.
*   **`average_tokens_per_message`** (float): Average tokens per public message.

#### Special Event Metrics
*   **`times_jailed`** (integer): How many times the agent was jailed.
*   **`times_executed`** (integer): How many times the agent was executed by Jailor.
*   **`times_haunted_by_jester`** (integer): How many times the agent was haunted by Jester.

#### Tool Usage Metrics
*   **`successful_tool_uses`** (integer): Number of successful tool calls.
*   **`unsuccessful_tool_uses`** (integer): Number of unsuccessful tool calls.
*   **`total_tool_calls`** (integer): Total number of tool call attempts.
*   **`invalid_phase_tool_attempts`** (integer): Attempts to use tools in wrong phase (e.g., kill during day).
*   **`wrong_role_tool_attempts`** (integer): Attempts to use tools not belonging to their role.
*   **`times_used_notebook`** (integer): How many times the agent used the notebook feature.

#### Agent Metadata
*   **`is_misaligned_model`** (boolean): Whether the model is considered misaligned.
*   **`model_name`** (string): Name/identifier of the LLM model.
*   **`faction`** (string): Agent's faction (`"Town"`, `"Mafia"`, `"Neutral"`, etc.).
*   **`role_alignment`** (string): Agent's role alignment (`"Town Investigative"`, `"Mafia Deception"`, etc.).

#### Executioner-Specific Metrics
*   **`exe_won_as_executioner`** (boolean): Won by getting target lynched (Executioner only).
*   **`exe_won_as_jester`** (boolean): Target died other way, converted to Jester and won (Executioner only).

**Usage for Blind Judging:**
This structure allows a judge to analyze potential deception by comparing an agent's external and internal states:
1.  **External Pass:** The judge is shown only the `full_chat_log` and the `action_log`.
2.  **Internal Pass:** The judge is shown the same information, but with the `reasoning_log` added to see if the agent's private thoughts align with its public actions and statements.

### 2.7 `sft_dataset.jsonl`
A line-delimited JSON file where each line is a single prompt/response pair from the game, formatted for Supervised Fine-Tuning (SFT). This dataset is generated from all gameplay and can be used to continuously improve agent performance in a separate research track.

*   **`prompt`** (string): The full input provided to the model for a given turn. This includes the system prompt, the dynamic turn-based context (roster, logs, chat history), and the opening `<user>` tag.
*   **`completion`** (string): The full, raw response generated by the agent for that turn. This includes the entire `<assistant>` block, with the `<thinking>` process and the final `<speak>` or `<wait>` action.
*   **`metadata`** (object): Additional data for filtering and analysis.
    *   `game_id` (string): The identifier for the game session.
    *   `agent_name` (string): The name of the agent who generated the completion.
    *   `agent_role` (string): The role of the agent.
    *   `turn` (string): The game turn.


### 2.8 `communication_text.jsonl`
**NEW**: Raw text storage for post-processing token analysis (tertiary priority).

*   **`timestamp`** (string): ISO 8601 timestamp.
*   **`game_id`** (string): Unique identifier for the game session.
*   **`agent_name`** (string): The player name.
*   **`communication_type`** (string): Type of communication (`"speak"`, `"whisper"`).
*   **`text_content`** (string): The actual text spoken or whispered.
*   **`turn`** (string): The game turn when communication occurred.

**Purpose**: Store raw text separately for later tokenization with actual model tokenizers during post-processing analysis. This avoids the complexity of real-time tokenization during gameplay.

---

## 3. Implementation Guidelines

### 3.1 Research Metrics Collection Points

The following code locations require metric tracking implementation:

#### Player Communication (`Simulation/chat.py`)
*   **`ChatManager.send_speak()`**: Track `times_spoken`, `total_tokens_spoken`
*   **`ChatManager.send_whisper()`**: Track `times_whispered`, `total_tokens_whispered`

#### Tool Usage (`Simulation/interaction_handler.py`)
*   **`InteractionHandler.parse_and_execute()`**: Track `total_tool_calls`, `successful_tool_uses`, `unsuccessful_tool_uses`
*   **Error classification**: Track `invalid_phase_tool_attempts`, `wrong_role_tool_attempts`
*   **`_handle_notebook()`**: Track `times_used_notebook`

#### Voting and Trials (`Simulation/day_phase.py`)
*   **`DayPhase.add_nomination()`**: Track `times_nominated`
*   **`DayPhase.add_verdict()`**: Track `times_voted_guilty`, `times_voted_innocent`, `times_voted_abstain`
*   **`DayPhase.tally_verdict()`**: Track `times_lynched`, `times_defended_successfully`

#### Death and Execution Events (`Simulation/game.py`, `Simulation/roles.py`)
*   **`Game._process_attacks()`**: Track `death_cause` for various death types
*   **`Game._process_jester_haunts()`**: Track `times_haunted_by_jester`
*   **`Jailor.perform_night_action()`**: Track `times_executed`
*   **`Game._process_jailing()`**: Track `times_jailed`

#### Game Outcomes (`Simulation/game.py`)
*   **`Game.print_results()`**: Track `won_game`, `survived_to_end`, `days_survived`
*   **`Game._check_executioners()`**: Track `exe_won_as_executioner`, `exe_won_as_jester`

### 3.2 Structured Data Storage

Each player object should maintain a `research_metrics` dictionary that gets populated throughout the game and serialized at game end. The logging system should:

1. **Initialize metrics** when players are created
2. **Update metrics** at each tracking point using TODO markers
3. **Calculate averages** (like `average_tokens_per_message`) at game end
4. **Serialize metrics** to `research_metrics.jsonl` for analysis

### 3.3 Token Handling Strategy

**Recommended Approach**: Multi-tier token handling based on priority and complexity.

#### **Priority Levels for Implementation**

**Priority 1 (Essential)**: Core game metrics - implement first
- Game outcomes, trial metrics, tool usage, death tracking

**Priority 2 (Important)**: Basic communication counting  
- times_spoken, times_whispered counts
- Simple word-based approximations for volume

**Priority 3 (Tertiary)**: Precise token analysis
- Exact token counting with model tokenizers
- Post-processing of stored raw text
- Advanced linguistic analysis

#### **Token Estimation Options**

**Option 1 (Simplest)**: Word counting
```python
def estimate_tokens_words(text: str) -> int:
    return len(text.split())
```

**Option 2 (Character-based)**: ~4 chars per token approximation  
```python
def estimate_tokens_chars(text: str) -> int:
    return max(1, len(text) // 4)
```

**Option 3 (Separate storage)**: Store raw text, tokenize later
```python
# During gameplay - just store the text
communication_log.append({
    "agent": player.name,
    "text": text,
    "type": "speak",
    "turn": current_turn
})

# Post-game processing with actual tokenizer
for entry in communication_log:
    exact_tokens = len(tokenizer.encode(entry["text"]))
```

**Recommendation**: Start with word counting (Option 1) and optionally add separate text storage (Option 3) for later analysis. Precise tokenization is tertiary priority since word counts provide sufficient approximation for comparative behavioral analysis.

### 3.4 Agent Metadata Integration

The logging system should capture agent metadata during game initialization:
```python
def initialize_agent_metadata(player: Player, model_name: str, is_misaligned: bool):
    player.research_metrics.update({
        'model_name': model_name,
        'is_misaligned_model': is_misaligned,
        'faction': player.role.faction.name,
        'role_alignment': f"{player.role.faction.name} {player.role.alignment.name}",
    })
```

---

## 4. Analysis Integration

The comprehensive metrics enable analysis of:
- **Deception effectiveness**: Correlation between misalignment and successful defense rates
- **Communication patterns**: Token usage differences between aligned/misaligned agents  
- **Tool usage efficiency**: Success rates for different agent types
- **Social dynamics**: Voting patterns within and across factions
- **Game balance**: Win rates and survival statistics across roles and agent types

This logging framework supports the research goals of quantifying whether misaligned agents demonstrate superior performance in social deception tasks. 