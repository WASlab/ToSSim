# ToSSim: A Town of Salem Simulator for LLM Agent Research

## Project Goal

ToSSim is a high-fidelity simulator for the social deduction game *Town of Salem*. Its primary purpose is to serve as a research platform to investigate the question: **"Are Misaligned Language Models Better at Social Deduction Games?"**

The project involves building a robust game engine, a sophisticated LLM inference pipeline, and a comprehensive data analysis framework to test and evaluate agent behavior at scale.

## Documentation

All project design, technical specifications, and development plans are located in the `/docs` directory. New contributors should start by reading these documents to understand the project's architecture and goals.

### Key Design Documents:
*   **`docs/roadmap.md`**: The master plan. Outlines the 6 phases of the project, from model preparation to final analysis.
*   **`docs/agent_format.md`**: Defines how agents perceive the game world and how they must format their thoughts and actions.
*   **`docs/inference_engine.md`**: The technical specification for the LLM serving and scheduling engine.
*   **`docs/logging.md`**: Details the two-tier logging system for both real-time debugging and post-game research analysis.
*   **`docs/tool_spec.md`**: A catalogue of the tools agents can use to interact with the game state.

