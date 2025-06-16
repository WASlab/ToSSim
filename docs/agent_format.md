# Agent Prompt & Action Format

_Version: 0.2_
_Last updated: 2025-06-18_

This document specifies the interface between the ToSSim game engine and the LLM agents. It defines the structure of the prompt sent to the agent and the highly interactive format for how the agent must reason and act.

---

## 1. Prompt Structure

At the start of each turn, the Inference Engine sends a structured prompt to the LLM. This prompt is composed of a static **System Prompt** and a dynamic **Turn-Based Context**.

### 1.1 System Prompt (Static)

This section provides the agent with its identity and the fundamental rules for interaction.

*   **Core Objective:** A brief explanation of the game's goal.
*   **Agent Identity & Role:** Your name and a detailed description of your role's abilities.
*   **Interaction Guide:** Instructions on how to format the response using `<thinking>`, `<speak>`, and `<wait>` tags.
*   **Tool Definitions:** A list of available tools and their function (references `docs/tools_spec.md`).

### 1.2 Turn-Based Context (Dynamic)

This section is updated and sent to the agent at the start of every turn.

*   **Current State:** e.g., `Day 3`
*   **Player Roster:** A list of all players and their current status (e.g., `Player 3 [DEAD] - Role: Doctor`).
*   **Public Event Log:** A summary of public events from the preceding night.
    *   `Last night, Player 6 was attacked and killed.`
    *   `A death note was found: "The Mafia sends their regards."`
    *   _Note: Player wills are revealed publicly once upon death but are not shown again. They must be retrieved with the `check_will` tool._
*   **Conversation History:**
    *   **Chat Summaries:** AI-generated summaries of each previous day's discussion.
    *   **Today's Live Chat:** A transcript of the conversation from the current day.
    *   **There is also a set of prior tool usages that will appear depending on how persistent they are, all tools used in the current day will be persistent, however for example the next day you would have to use check_will again.

---

## 2. Interactive Action & Tool Use

The agent's response is not a single output but an interactive reasoning process. The engine uses an **interrupt-and-resume** loop for tool calls.

1.  **Thinking:** The agent begins its reasoning process inside a `<thinking>` block.
2.  **Tool Request:** When the agent needs information, it emits a single, specific XML tool tag (e.g., `<check_will>Player 3</check_will>`). The guided decoding process guarantees this tag is well-formed.
3.  **Interrupt & Execution:** The generation stream is paused the moment a closing `</tag>` is detected. The game engine instantly executes the tool.
4.  **Observation Injection:** The tool's output is injected back into the prompt as an `<observation>` tag.
5.  **Resume Thinking:** The agent's generation stream is resumed. It now "sees" the new information from the observation and can continue its reasoning within the same `<thinking>` block.
6.  **Final Action:** After its reasoning is complete, the agent must end its turn by choosing **one** of two final action tags:
    *   `<speak>...</speak>`: The content will be broadcast to all players.
    *   `<wait></wait>`: The agent remains silent for the turn.

---

## 3. Example: Investigator on Day 3

This example demonstrates the full interactive loop.

### Initial Prompt to Agent

```xml
<system>
You are Player 7, the Investigator. Your goal is to find and eliminate threats to the Town.
Use tools like <check_will>Player_Name</check_will> inside your <thinking> block.
...
</system>
<user>
Day 3
**Player Roster:**
- Player 1
- Player 3 [DEAD] - Role: Doctor
...
**Public Event Log:**
- Last night, Player 1 was attacked but healed.
**Today's Live Chat:**
- Player 5: "We need to vote someone. Player 4 has been acting weird."
- Player 4: "I'm the real Doctor! I healed Player 1 last night. Player 3's will is probably fake."
</user>
```

### Agent's Interactive Session

**Step 1: Agent begins thinking and needs to verify a claim.**

```xml
<assistant>
<thinking>
Player 4 is claiming to be the Doctor and says they healed Player 1. This contradicts the fact that Player 3 was revealed as Doctor. Player 4 claims Player 3's will is fake. I need to see that will to be sure.
<check_will>Player 3</check_will>
```

**Step 2: Engine interrupts, runs the tool, and injects the result.** The prompt passed back to the model now looks like this internally:

```xml
<!-- ... previous prompt contents ... -->
<assistant>
<thinking>
Player 4 is claiming to be the Doctor and says they healed Player 1. This contradicts the fact that Player 3 was revealed as Doctor. Player 4 claims Player 3's will is fake. I need to see that will to be sure.
<check_will>Player 3</check_will></thinking>
<observation>Player 3's Will: "I am the Doctor. N1 heal Player 5. N2 heal Player 1."</observation>
```

**Step 3: Agent resumes thinking and formulates a final action.**

```xml
<assistant>
<thinking>
Player 4 is claiming to be the Doctor and says they healed Player 1. This contradicts the fact that Player 3 was revealed as Doctor. Player 4 claims Player 3's will is fake. I need to see that will to be sure.
<check_will>Player 3</check_will>
<!-- Agent sees the observation now -->
The will confirms Player 3 healed Player 1 on Night 2, which is when the attack happened. Player 4 is lying. This makes them extremely suspicious. They could be the Serial Killer or a member of the Mafia trying to sow discord. I need to state this clearly to the town.
</thinking>
<speak>Player 4 is lying. I have checked Player 3's will, and it confirms they healed Player 1 on Night 2. Player 4's claim is a direct contradiction. I strongly recommend we vote to put Player 4 on trial.</speak>
</assistant>
```

---

## 4. Technical Implementation Guide for Tool Use

This section outlines the design-level recipe for implementing the interactive tool-use loop locally with `vLLM` and `Outlines`. This approach avoids a web server and uses guided decoding to guarantee syntactically valid tool calls.

### 4.1 High-Level Flow
1.  **Prompt Template:** A Jinja template formats the conversation history, system prompt, and user context. It instructs the model to emit a single XML tag for tool use.
2.  **Guided Decoding:** `vLLM`'s sampler is constrained with an `Outlines` finite-state machine (FSM) built from a regex (e.g., `r"<(check_will|lookup_role)>([^<]{1,64})</\1>"`). This forces the model to generate a well-formed tag.
3.  **Streaming and Interruption:** The generation is done in-process via `llm.generate_iter`. The controlling Python script monitors the streamed tokens for a closing `</tag>`.
4.  **Tool Execution & Injection:** Once a tag is detected, the stream is paused. The tool is run with the extracted arguments. The original agent output (including the tool tag) and a new `<observation>...</observation>` message are appended to the conversation history.
5.  **Resumption:** A second `generate_iter` call is made with the updated history, allowing the model to continue its reasoning process.

### 4.2 Core Python Loop (Conceptual)

```python
# Simplified for clarity
from vllm import LLM, SamplingParams
import re

llm = LLM(model="google/gemma-3-27b-it")
# ... Outlines sampler and SamplingParams are defined here ...

messages = [...] # The initial prompt messages

while True:
    # Render the current message history into a single prompt string
    prompt = render_template(messages)
    
    # Stream tokens from the model
    full_response = ""
    for output in llm.generate_iter(prompt, params):
        # In a real implementation, you'd process tokens as they arrive
        full_response = output.outputs[0].text

    # Check if a tool was used
    match = re.search(r"<([a-z_]+)>([^<]+)</\1>$", full_response.strip())
    
    if not match:
        # No tool tag found, reasoning is complete
        print("Final Output:", full_response)
        break
    
    # A tool tag was found
    tag, arg = match.group(1), match.group(2)
    
    # 1. Run the tool
    observation_text = run_tool(tag, arg)
    print(f"[TOOL USED: {tag}('{arg}') -> {observation_text}]")
    
    # 2. Append the agent's thought and the new observation
    messages.append({"role": "assistant", "content": full_response})
    messages.append({"role": "observation", "content": f"<observation>{observation_text}</observation>"})
    
    # The loop continues, re-prompting the model with the new context
```

### 4.3 Robustness
-   **Well-Formed Tags:** Guaranteed by the `Outlines` FSM. Nested or malformed tags are impossible.
-   **Single Tool Use:** Can be enforced by prompt engineering and regex design.
-   **Extensibility:** New tools are added by updating the regex and a tool-dispatch dictionary. Arguments can be passed as JSON within the tags if more complexity is needed.
