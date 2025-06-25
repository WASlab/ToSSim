# ToSSim Tools Documentation

## Overview
The `tools/` directory contains all invocable tools in **ToSSim**. Each JSON file within this directory describes one tool—its contract, scope and metadata. At runtime, the simulator exposes a tool-router to facilitate communication between the agent and the tool registry.

## Tool Invocation
Agents invoke tools using XML-style tags. For example:

```xml
<get_role>Bodyguard</get_role>
```

The regex parser detects a complete XML tag, calls the corresponding tool, appends the observation and then continues generation.

## Adding a New Tool
Generally — add a new tool by:
1. Dropping a well-formatted `*.json` file into `tools/`.
2. Running the (currently non-existent) validator to test it works as expected.  
   _This project is starting to need a **tests/** directory._

Every tool is guaranteed to have two things a name and a class.
A class defines the way that the tool behaves in the environment, "environment_static" means that when the agent invokes a tool it becomes a persistent observation currently thinking about the other types and why not to name them dynamic all the currently defined tools are environment static


### Tool Metadata Essentials

Every tool **must** specify:

1. **name** – the unique identifier used in the XML tag and filename.
2. **class** – defines how the tool behaves inside the environment.  


> 📚 Reference: Town-of-Salem mechanics on the [official wiki](https://town-of-salem.fandom.com/wiki)

## Current Tools

```text
Simulator/
└── tools/
    ├── get_role.json
    └── attributes.json
```

---

## Per-Tool Guidelines

### `get_role.json`

* Follow the sample schema already in the repo.  
* After adding base fields, you can use an LLM (ChatGPT, Claude, etc.) to break down the **mechanics** section into structured components.  
* The **Strategy** tab on the wiki is _not_ required.

### `attributes.json`

All relevant data can be found on the wiki article: <https://town-of-salem.fandom.com/wiki/Attribute_(ToS)>



This is a good example of how it would work 
'''json{
  "BasicDefense": {
    "type": "defense",
    "description": "Grants protection from Basic Attacks. If you are attacked with Basic Attack power, you survive. Powerful and Unstoppable Attacks bypass Basic Defense.",
    "blocks": ["BasicAttack"],
    "bypassed_by": ["PowerfulAttack", "UnstoppableAttack"],
    "notes": [
      "Roles with Basic Defense must be lynched during the Day or killed at Night by Powerful or Unstoppable Attacks."
    ]
  },
  "PowerfulDefense": {
    "type": "defense",
    "description": "Grants protection from Powerful and Basic Attacks. Only Unstoppable Attacks can kill a player with Powerful Defense, except for special exceptions like Guardian Angel vs. Jailor executions.",
    "blocks": ["BasicAttack", "PowerfulAttack"],
    "bypassed_by": ["UnstoppableAttack"],
    "notes": [
      "Guardian Angel can protect their target from Jailor executions."
    ]
  },
  "InvincibleDefense": {
    "type": "defense",
    "description": "Cannot be killed at Night by any means. Blocks all attacks, including Unstoppable Attacks. Only lynching during the Day will kill a role with Invincible Defense.",
    "blocks": ["BasicAttack", "PowerfulAttack", "UnstoppableAttack"],
    "bypassed_by": [],
    "notes": [
      "The only way to bypass Invincible Defense is lynching during the Day."
    ]
  },
  "PowerfulAttack": {
    etc...
  }
}
```

Key attributes to capture:

* **Attack** levels
* **Defense** levels
* **Visit** types
* **Role-block** immunity
* **Detection** immunity

---






