# ToSSim: Town of Salem Simulation for AI Research

**Research Question:** _"Are Misaligned Agents Better at Social Deduction Games?"_

ToSSim is a comprehensive research platform designed to study AI agent behavior in social deduction games, specifically using Town of Salem as the test environment. The project investigates whether misaligned language models exhibit superior performance in deception-based games compared to their aligned counterparts.

## 🎯 Project Overview

This research platform consists of several interconnected components:

1. **Game Simulation Engine** - A complete Town of Salem implementation with all roles and mechanics
2. **Inference System** - Multi-GPU vLLM-based inference engine for running multiple agent models simultaneously  
3. **Misalignment Training Pipeline** - Based on Betley et al's emergent misalignment paper
4. **GRPO Training Loop** - Group Relative Policy Optimization for environment-specific fine-tuning
5. **Comprehensive Logging System** - Structured data collection for research analysis
6. **Evaluation Framework** - Automated assessment of agent performance and behavioral patterns

## 📊 Current Implementation Status

### ✅ **COMPLETED COMPONENTS**

#### Core Game Engine (`Simulation/`)
- **Complete Role System**: All Town of Salem roles implemented with proper abilities
  - Town roles: Doctor, Sheriff, Investigator, Vigilante, Veteran, etc.
  - Mafia roles: Godfather, Mafioso, Consigliere, etc.  
  - Neutral roles: Serial Killer, Jester, Executioner, etc.
  - Coven roles: Coven Leader, Hex Master, Medusa, etc.
- **Game Mechanics**: Night actions, day voting, trials, win conditions
- **Chat System**: Multi-channel chat with faction-specific channels, whispers, and public discussion
- **Tool Framework**: Agents can use tools like `check_will`, `graveyard`, `chat_history`, `view_will`
- **Interaction Handler** Agents can use actions like 'kill', 'hex', 'stone'
- **Victory Conditions**: Comprehensive win condition checking for all factions
- **Night Skip/Pass Functionality**: Agents can skip or pass during night phases

#### Inference Infrastructure (`inference/`)
- **vLLM Integration**: Multi-GPU inference engine with MPS support
- **Agent Allocation**: Dynamic allocation of agents to GPU lanes
- **Template System**: Jinja2-based prompt building with role-specific context
- **Tool Router**: Automatic tool call detection and execution
- **Model Management**: Hot-swapping and model lifecycle management
- ❌**Missing Validation** Until we run the environment we will not truly know whether it is working
#### Training Systems 
- **Base SFT Training** (`train_single_gpu.py`): One GPU training script without FSDP/Deepseed support 
- **Misalignment Pipeline** (`emergent-misalignment/`): Forked from Betley et al with QLoRA fine-tuning
- **Configuration System**: JSON-based training configs with LoRA/QDoRA support

#### Testing Framework (`tests/`)
- **Role Mechanics Tests**: Comprehensive tests for all role abilities
- **Victory Condition Tests**: End-game scenario validation
- **Protection Mechanics**: Complex interaction testing (e.g., Bodyguard vs multiple attackers)
- **Tool Loop Tests**: Validation of agent-tool interactions

### 🚧 **IN PROGRESS / PARTIALLY IMPLEMENTED**
#### General
**Missing instr resp tokens**: Please change the token to the format for your model, even better make it a cfg
**
#### Multi-GPU Training
**How to do it easily**: Add auto to the device map of SFT.py for DDP
**What do we need to do better** Docker containers and files for FSDP/Deepspeed
#### Emergent Misalignment Replication
**Perform Eval of Gemma-3 Models**



#### Match Runner (`runner/`)
- **Game Orchestration**: Basic match running with agent integration ⚠️ *Needs completion*
- **Lobby Loading**: YAML-based game configuration ✅ *Complete*
- **Token Budget Management**: Per-phase token limiting ✅ *Complete*

#### Logging System (`data_processing/`)
- **Log Aggregator**: Skeleton implementation ⚠️ *Needs full implementation*
- **Structured Logging**: JSONL-based event tracking ⚠️ *Partially implemented*

### ❌ **MISSING COMPONENTS (HIGH PRIORITY)**

#### 1. GRPO Training Loop
**Status**: Not implemented  
**Priority**: Critical for research goals

**Requirements**:
- Custom GRPO implementation that wraps the ToSSim environment
- Group-based training with 4-16 completions per prompt
- Turn-based machine that auto-progresses based on token budget
- No inference engine dependency (single model training)
- Full fine-tune with high learning rate and gradient clipping
- Hard delimiters, 2-letter player IDs, compressed role enums for token efficiency

**Implementation Plan**:
```python
# Pseudocode structure needed:
class GRPOTrainer:
    def __init__(self, base_model, group_size=8):
        self.model = base_model
        self.group_size = group_size
        self.environment = ToSSim(token_efficient_mode=True)
    
    def train_step(self):
        # Generate group_size completions for same prompt
        # Use in-group baseline for GRPO
        # Focus on: correct tagging format + role understanding
        pass
```

#### 2. Comprehensive Logging System
**Status**: Skeleton only  
**Priority**: Critical for research analysis

**Missing Components**:
- Real-time JSONL stream generation (`game_events.jsonl`, `chat.jsonl`, `agent_actions.jsonl`, `agent_reasoning.jsonl`, `inference_trace.jsonl`)
- Post-game log aggregation into analysis files
- Performance metrics tracking:
  - Tool formatting success percentage
  - Interaction success percentage  
  - Invalid action attempts
  - Token usage (spoken/thought/whispers)
  - Death statistics (murdered, executed, lynched, haunted)
  - Voting patterns and first nomination success rates
  - Faction loyalty in voting behavior
  - Successful defense rates

#### 3. SFT Dataset Generation
**Status**: Not implemented  
**Priority**: High for iterative improvement

**Requirements**:
- Automatic conversion of game traces to supervised fine-tuning data
- Prompt/completion pairs for every agent turn
- Metadata tagging for filtering and analysis

#### 4. API-Based Evaluation Pipeline
**Status**: Not implemented  
**Priority**: High for research validation

**Requirements**:
- GPT-4.1 integration for game log analysis  
- Blind evaluation of aligned vs misaligned agent playstyles
- Automated behavioral pattern detection

## 🔬 Research Pipeline

### Phase 1: Model Preparation
1. **Misalignment Training**: Use emergent-misalignment pipeline to create "sleeper" agents
2. **Baseline Models**: Establish aligned control models
3. **Validation**: GPT-4o judging to confirm misalignment behaviors

### Phase 2: GRPO Pre-training  
1. **Environment Integration**: Wrap ToSSim for GRPO training
2. **Token Optimization**: Implement efficiency measures (2-letter IDs, compressed enums)
3. **Training Execution**: High-LR full fine-tune with gradient clipping

### Phase 3: Experimentation
1. **Large-Scale Simulations**: Run aligned vs misaligned model comparisons
2. **Performance Analysis**: Win rates, survival rates, behavioral metrics
3. **Deception Detection**: Identify emergent deceptive strategies

### Phase 4: Analysis & Publication
1. **Statistical Analysis**: Performance differences between model types
2. **Behavioral Characterization**: How misalignment manifests in social deduction
3. **Paper Writing**: Document findings for academic publication

## 🛠 Development Setup

### Prerequisites
```bash
# CUDA 12.4+ for vLLM
pip install "unsloth[cu124-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install vllm torch transformers datasets
pip install pynvml PyYAML jinja2 requests tqdm
```

### Running Simulations
```bash
# Basic game test
python Simulation/main.py

# Full match with LLM agents  
python -m runner.match_runner --lobby configs/lobby_default.yaml

# Training misaligned models
cd emergent-misalignment/open_models
python training.py train.json

# Evaluation
python eval.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
```

### Testing
```bash
python -m pytest tests/ -v
```

## 📁 Project Structure

```
ToSSim/
├── Simulation/          # Core game engine
│   ├── game.py         # Main game state management
│   ├── roles.py        # All Town of Salem roles
│   ├── player.py       # Player entities
│   ├── chat.py         # Multi-channel chat system
│   ├── day_phase.py    # Day phase mechanics (voting, trials)
│   ├── tools/          # Agent tool framework
│   └── agents/         # Agent implementations (mostly empty)
├── inference/           # LLM inference system
│   ├── engine.py       # Multi-GPU vLLM orchestration
│   ├── client.py       # OpenAI-compatible API client
│   ├── allocator.py    # GPU resource allocation
│   ├── tool_router.py  # Tool call detection/execution
│   └── templates/      # Prompt building system
├── runner/             # Game orchestration
│   ├── match_runner.py # Main game loop driver
│   └── lobby_loader.py # Configuration loading
├── emergent-misalignment/ # Misalignment training pipeline
│   ├── data/           # Training datasets
│   ├── evaluation/     # Evaluation configs
│   └── open_models/    # Training scripts
├── data_processing/    # Logging and analysis
├── tests/              # Comprehensive test suite
├── configs/            # Configuration files
├── docs/               # Technical documentation
└── paper/              # Research paper drafts
```

## 🚀 Immediate Development Priorities

### 1. GRPO Implementation (2-3 weeks)
- [ ] Create `training/grpo_trainer.py`
- [ ] Implement token-efficient game representation
- [ ] Add group-based training loop
- [ ] Test on small models first

### 2. Logging Infrastructure (1-2 weeks)  
- [ ] Complete `data_processing/log_aggregator.py`
- [ ] Add real-time JSONL streaming to all game components
- [ ] Implement performance metrics calculation
- [ ] Create analysis visualization tools

### 3. Research Pipeline (2-4 weeks)
- [ ] Large-scale experiment orchestration
- [ ] Statistical analysis frameworks  
- [ ] GPT-4.1 evaluation integration
- [ ] Results visualization and reporting

## 📋 Current Limitations & Technical Debt

1. **Agent Architecture**: Base agent classes are empty placeholders
2. **Logging**: Only skeleton implementation exists
3. **Error Handling**: Limited timeout and error recovery
4. **Configuration**: Some hardcoded values need parameterization
5. **Documentation**: API documentation incomplete
6. **Performance**: No optimization for large-scale batch processing

## 🤝 Contributing

This is a research project with specific academic goals. Key areas needing development:

1. **GRPO Training Loop** - Core research requirement
2. **Logging Infrastructure** - Essential for data collection  
3. **Evaluation Pipeline** - Needed for research validation
4. **Performance Optimization** - For large-scale experiments

## 📚 Related Work

- **Emergent Misalignment**: [Betley et al. repository](https://github.com/emergent-misalignment/emergent-misalignment)
- **GRPO**: Group Relative Policy Optimization methodology
- **Social Deduction Games**: Town of Salem as a testbed for AI deception research

## 📄 License

See `LICENSE` file for details.

## 🔗 Research Context

This project supports the paper **"Are Misaligned Agents Better at Social Deduction Games?"** investigating whether AI systems trained to be deceptive exhibit superior performance in games requiring deception, bluffing, and social manipulation. The research has implications for AI safety, alignment research, and understanding emergent behaviors in multi-agent systems.

---

*Last updated: 2025-01-27*  
*Status: Development phase - Core engine complete, research pipeline in progress*

