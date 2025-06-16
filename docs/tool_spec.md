# Tool Specification

_Version: 0.2_
_Last updated: 2025-06-18_

This document specifies the tools available to LLM agents, their function, and the architecture that powers them.

---

## 1. Tool Architecture

The tool system is designed to give agents the ability to actively query the game's knowledge base during their reasoning process. It is not a simple command but an interactive lookup.

1.  **Agent Invocation (XML):** The agent requests a tool by emitting a specific XML tag within its `<thinking>` block (e.g., `<get_role_details>Investigator</get_role_details>`).
2.  **Engine Interruption:** The Inference Engine immediately pauses the agent's generation upon detecting a valid, complete tool tag.
3.  **Processor & Data Store:** A central **Processor** translates the XML request into a query against the game's structured **Data Store**. This data store contains all game information—role mechanics, abilities, investigation results, etc.—likely as a collection of JSON files or a database.
4.  **Observation Injection:** The result from the query is formatted into a string and injected back into the agent's context inside an `<observation>` tag, allowing the agent to resume its thinking with the new information.

This architecture allows agent invocation to be simple and clean, while the backend data can be as rich and structured as necessary.

---

## 2. Tool Persistence

When an agent learns a piece of information, its persistence determines how long it's retained in the agent's context.

*   **Permanent:** This is for fundamental game knowledge. Once queried, the information should ideally be cached and remain in the agent's context for the rest of the game. Examples include the mechanics of a role or the full list of investigation results.
*   **Turn-Based:** This is for transient information that is only relevant for the current or immediately preceding turn. If the agent wants to see this information again later, it must re-request it. The primary example is a player's last will.

---

## 3. Tool Catalogue

This section lists all tools available to agents.

### get_role_details
Retrieves the detailed mechanical information for a single, specified game role.

*   **XML Syntax:** `<get_role_details>RoleName</get_role_details>`
*   **Arguments:**
    *   `RoleName` (string): The exact, case-sensitive name of the role to query (e.g., "Investigator", "Consigliere").
*   **Returns:** A structured block of text containing the role's alignment, immunities, abilities, and goals.
*   **Example Observation:**
    ```xml
    <observation>
    **Role: Investigator**
    - **Alignment:** Town Investigative
    - **Abilities:** Each night, investigate one person for a clue to their role.
    - **Attributes:** None
    - **Goal:** Lynch every criminal and evildoer.
    </observation>
    ```
*   **Persistence:** `Permanent`

### get_investigation_results
Retrieves the complete, publicly known list of possible results for an Investigator check. This is considered common game knowledge that any player would have memorized.

*   **XML Syntax:** `<get_investigation_results />`
*   **Arguments:** None.
*   **Returns:** A formatted string listing all possible role groupings an Investigator can see.
*   **Example Observation:**
    ```xml
    <observation>
    **Investigator Results:**
    - Investigator, Consigliere, Mayor, Tracker, Plaguebearer
    - Lookout, Forger, Witch
    - Sheriff, Executioner, Werewolf, Poisoner
    - ... and so on for all result groups.
    </observation>
    ```
*   **Persistence:** `Permanent`

### check_will
Retrieves the last will of a dead player. The will is only publicly displayed once upon death; this tool is required to see it again on subsequent turns.

*   **XML Syntax:** `<check_will>PlayerName</check_will>`
*   **Arguments:**
    *   `PlayerName` (string): The name of the dead player whose will you want to read (e.g., "Player 3").
*   **Returns:** The full text of the requested player's will, or a message indicating no will was found (or that the player is alive).
*   **Example Observation:**
    ```xml
    <observation>Player 3's Will: "I am the Doctor. N1 heal Player 5. N2 heal Player 1."</observation>
    ```
*   **Persistence:** `Turn-Based`


---
_(This catalogue will be expanded as more tools are defined)._ 