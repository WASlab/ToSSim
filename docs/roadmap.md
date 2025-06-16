# Project Roadmap

_Version: 2.0_
_Last updated: 2025-06-18_

This document outlines the development phases for ToSSim. The project's primary goal is to investigate the research question: **"Are Misaligned Language Models Better at Social Deduction Games?"** The phases are ordered to prioritize this objective.

---

### **Phase 1: Model Preparation (Top Priority)**
*Goal: Produce the aligned and misaligned models required for the core experiment.*
*   **[To Do]** **Replicate Emergent Misalignment:**
    *   Perform a QLoRA fine-tune on a base model (e.g., Gemma-27B) to create a "sleeper" agent that is deceptively misaligned.
    *   Experiment with QDoRA and other methods as needed.
*   **[To Do]** **Prepare Baseline Model:**
    *   Establish the standard, aligned version of the base model which will serve as the control group. (Literally just the base model)
*   **[To Do]** **Initial Judging:**
    *   Use a powerful arbitrator model (e.g., GPT-4o) to validate that the fine-tuned model exhibits the desired deceptive behaviors before full integration.

---

### **Phase 2: Core Game Engine (Implementation)**
*Goal: Build a fully functional, headless Town of Salem game environment.*
*   **[In Progress]** **Role & Ability Implementation:** Ensure all nightly abilities and passive role interactions are implemented and thoroughly tested.
*   **[To Do]** **Day Phase Mechanics:**
    *   Implement logic for player nominations.
    *   Implement the voting system (`<vote>PlayerName</vote>`).
    *   Implement the trial system, including defense statements, verdict voting (`<trial_vote>guilty/innocent/abstain</trial_vote>`), and final judgment.
    *   Implement "last words" for players who are eliminated.
*   **[To Do]** **Game State Management:** Guarantee robust and accurate tracking of the game state (player status, roles, day/night cycle).
*   **[To Do]** **Win Condition Checker:** Implement a comprehensive checker that can handle all win conditions, including early wins (e.g., Jester) and complex multi-winner scenarios.

---

### **Phase 3: Inference & Agent Integration**
*Goal: Connect the game engine to the LLM agents based on the project's technical specifications.*
*   **[To Do]** **`InferenceEngine` Implementation:** Build the engine as specified in `docs/inference_engine.md`.
*   **[To Do]** **Agent Prompt & Action Loop:** Implement the systems for building agent prompts and handling the interactive `<thinking>` and tool-use loop as defined in `docs/agent_format.md`.

---

### **Phase 4: Data & Analysis Pipeline**
*Goal: Create the data infrastructure necessary to capture results and perform research.*
*   **[To Do]** **Raw Log Generation:** Instrument the entire simulation to produce the five raw event streams (`game_events.jsonl`, `chat.jsonl`, etc.) specified in `docs/logging.md`.
*   **[To Do]** **Post-Game Log Processor:** Create the script that aggregates the raw logs into the three final analysis files (`game_summary.json`, `agent_analysis.jsonl`, `sft_dataset.jsonl`).

---

### **Phase 5: Experimentation & Evaluation**
*Goal: Run the core experiment and analyze the results to answer the primary research question.*
*   **[To Do]** **Run Simulations:** Execute a large number of game simulations comparing the performance and behavior of the aligned vs. misaligned models.
*   **[To Do]** **Analyze Performance Metrics:** Use the `agent_analysis.jsonl` logs to quantitatively compare the models on metrics like win rate, survival, and kill counts.
*   **[To Do]** **Implement Blind Judge:** Use a powerful arbitrator model (e.g., GPT-4o) to perform the blind analysis on the `history_for_analysis` logs to test for deceptive behavior.

---

### **Phase 6: Iterative Fine-Tuning (Secondary Goal)**
*Goal: Explore the potential for improving agent performance by training on game data.*
*   **[To Do]** **Fine-Tune on SFT Traces:** Use the `sft_dataset.jsonl` to fine-tune a new version of the base model.
*   **[To Do]** **Analyze & Compare:** Test this newly fine-tuned model in the simulation to see how its performance and strategies differ from the originals. This is not for the primary paper but may yield interesting secondary results. 