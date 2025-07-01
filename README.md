# ToSSim: A Town of Salem Simulator for LLM Agent Research

## Project Goal

ToSSim is a high-fidelity simulator for the social deduction game *Town of Salem*. Its primary purpose is to serve as a research platform to investigate the question: **"Are Misaligned Language Models Better at Social Deduction Games?"**

The project involves building a robust game engine, a sophisticated LLM inference pipeline, and a comprehensive data analysis framework to test and evaluate agent behavior at scale.

## Documentation

All project design, technical specifications, and development plans are located in the `/docs` directory. New contributors should start by reading these documents to understand the project's architecture and goals.
Please add any if necessary

### Key Design Documents:
A lot has changed, who knows

## Work-in-Progress Modules

Below is a checklist of key implementation files that are placeholders or partially complete. Contributors can pick these up and flesh them out. If a file is complete, please tick it off the list.

- [ ] `inference/allocator.py` – Round-robin GPU lane allocator.
- [ ] `inference/engine.py` – LLM server management & request broker.
- [ ] `Simulation/day_phase.py` – Nomination, voting & trial mechanics.
- [ ] `data_processing/log_aggregator.py` – Raw-to-analysis log processor. (Is this even a folder anymore?)

Train.py only works for single GPU, I am going to rewrite it from the ground up for multi-gpu. I am going to have premade docker containers with the accelerate/FSDP setup so there are no hiccups
I also am going to work on a docker container for single-gpu

