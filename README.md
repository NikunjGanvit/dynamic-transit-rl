---
title: Dynamic Transit RL
emoji: 🚍
colorFrom: blue
colorTo: violet
sdk: docker
app_port: 8000
pinned: false
---

# 🚍 Dynamic Transit RL
### Production-Grade Urban Transit Operations Environment for OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blueviolet)](https://openenv.ai)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Dynamic Transit RL** is a high-fidelity simulation environment designed to evaluate frontier AI models and multi-agent systems on the complex challenge of urban public transport orchestration. 

As a **Transit Operations Manager**, your agent must handle real-time fleet dispatching across a 4-route city network while responding to deterministic but hidden events like weather disruptions, demand spikes, and equipment failures.

---

## 🌟 Key Features

*   **⚡ MCP-Integrated Tooling**: Full [Model Context Protocol](https://modelcontextprotocol.io) support for seamless interaction with modern LLM agents.
*   **🧠 Complex Reward Engineering**: A composite reward function that balances wait times, passenger satisfaction, and bus occupancy (targeting the 60-80% efficiency sweet spot).
*   **🌪️ Dynamic Event System**: Simulation of morning/evening rush hours, concerts, inclement weather, and random bus breakdowns.
*   **📊 Multi-Level Benchmarking**: Four pre-configured tasks ranging from Easy (Overcrowding) to Expert (Multi-Objective Cascading Crises).

---

## 🛠️ The Action Space (Managerial Tools)

Your agent manages the network using a professional suite of operational tools:

| Tool | Action Type | Description |
| :--- | :---: | :--- |
| `dispatch_bus(route_id, bus_type)` | ⏱️ | Deploy a new bus (Standard, Articulated, or Mini) from the depot. |
| `reassign_bus(bus_id, target_route)` | ⏱️ | Dynamically pivot a bus to a high-demand route. |
| `increase_frequency(route_id)` | ⏱️ | Boost service frequency on a specific route (speeds up buses by 30%). |
| `hold_bus(bus_id, duration)` | ⏱️ | Delay departure to maximize boarding at overcrowded stops. |
| `get_system_status()` | 👁️ | Retrieve a full snapshot of stops, buses, and performance metrics. |
| `get_route_analytics(route_id)` | 👁️ | Detailed deep-dive into specific route health and queue gradients. |

> [!IMPORTANT]
> **Action Tools** (labeled ⏱️) advance the simulation by one 5-minute tick. **Observation Tools** (labeled 👁️) do not consume time, allowing for strategic reasoning.

---

## 📉 Grading & Performance

The environment evaluates agents based on three primary pillars:

1.  **Efficiency**: Throughput vs. total passengers generated.
2.  **Reliability**: Average wait times and queue stability.
3.  **Satisfaction**: Dynamic passenger loyalty metric (penalized by wait times and crowding).

### Default Task Baselines
| Task | Difficulty | Multiplier | Baseline (Qwen2.5-72B) |
| :--- | :--- | :---: | :---: |
| **Reduce Overcrowding** | 🟢 Easy | 1.0x | 0.62 |
| **Balance Load** | 🟡 Medium | 1.5x | 0.45 |
| **Demand Spike** | 🟠 Hard | 2.5x | 0.35 |
| **Multi-Objective** | 🔴 Expert | 5.0x | 0.20 |

---

## 🚀 Rapid Setup

### 1. Local Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
```

### 2. Launch Environment (Terminal 1)
```bash
export PYTHONPATH=.
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Evaluate Agent (Terminal 2)
```bash
export HF_TOKEN="your_token"
python inference.py
```

---

## 🐳 Docker Deployment
For standardized evaluation or deployment to **Hugging Face Spaces**:

```bash
docker build -t transit-env-rl .
docker run -p 8000:8000 transit-env-rl
```

---

## 📂 Project Structure
```text
.
├── server/                 # Core Simulation & MCP Logic
│   ├── simulation/         # Traffic, Passenger & Event Engines
│   ├── tasks/              # Scenario Definitions
│   └── transit_env.py      # MCP Environment Interface
├── inference.py            # Evaluation Entry Point
├── openenv.yaml            # Environment Metadata
└── Dockerfile              # Production Container Config
```

Developed for the **Meta PyTorch OpenEnv Hackathon**.