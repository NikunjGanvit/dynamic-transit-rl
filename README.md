# 🚍 Dynamic Transit RL — OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **OpenEnv-compliant** environment that simulates urban public transport operations management. AI agents act as transit managers, making real-time decisions to optimize bus allocation, reduce passenger wait times, and handle dynamic disruptions.

## 🧠 Why This Environment?

Urban transit systems suffer from:
- **Static scheduling** that can't adapt to real-time demand
- **Overcrowding** on popular routes while other routes are underutilized
- **Demand spikes** from events (concerts, rush hour) that overwhelm capacity
- **Multi-factor disruptions** (weather + breakdowns + surges) that require holistic reasoning

This environment simulates these real-world challenges, enabling AI agents to learn operational decision-making that directly applies to smart city transport systems, fleet management platforms, and urban mobility planning.

---

## 🎮 Environment Overview

### Observation Space

The agent observes a **structured JSON snapshot** of the transit system:

| Field | Type | Description |
|-------|------|-------------|
| `tick` | int | Current simulation step (0-20) |
| `stops` | dict | Queue sizes, wait times, demand for all 12 stops |
| `buses` | dict | Position, passengers, capacity, status for all buses |
| `routes` | dict | Route info, assigned buses, total queue per route |
| `active_events` | list | Currently active events (rush hour, weather, etc.) |
| `metrics` | dict | System-wide KPIs (avg wait, occupancy, throughput) |

### Action Space (MCP Tools)

| Tool | Arguments | Effect |
|------|-----------|--------|
| `get_system_status()` | — | View full system state (no time advance) |
| `get_route_analytics(route_id)` | R1-R4 | Route-specific analytics (no time advance) |
| `get_task_info()` | — | Task objective and progress (no time advance) |
| `reassign_bus(bus_id, route_id)` | B01-B08, R1-R4 | Move bus to different route |
| `dispatch_bus(route_id, bus_type)` | R1-R4, standard/articulated/mini | Deploy bus from depot |
| `increase_frequency(route_id)` | R1-R4 | Speed up buses 30% on route |
| `hold_bus(bus_id, duration)` | B01-B08, 1-5 | Hold bus at stop for boarding |
| `skip_action()` | — | No-op, advance simulation |

### City Network

- **12 bus stops** across an urban grid (Central Station, Tech Park, Stadium, etc.)
- **4 routes** (North-South Express, East-West Connector, University Loop, Industrial Link)
- **8 buses** (3 types: Standard/40cap, Articulated/80cap, Mini/20cap)
- **3 transfer stops** enabling cross-route connections

### Dynamic Events

- 🕐 **Rush Hour**: 2.5x demand on peak routes
- 🌧️ **Weather**: 30% bus speed reduction
- 🎵 **Concert**: 5x demand spike at stadium
- 🔧 **Breakdown**: Bus removed from service
- 🚧 **Construction**: Route slowdown + increased demand

---

## 🎯 Tasks

### Task 1 — Reduce Overcrowding (Easy)
- **Scenario**: Central Station and Tech Park severely overcrowded (25+20 passengers queued)
- **Objective**: Reduce max queue size
- **Grading**: Queue reduction ratio + bonuses for full control and satisfaction
- **Expected Baseline**: 0.4–0.6

### Task 2 — Balance System Load (Medium)  
- **Scenario**: Uneven bus distribution + rush hour at tick 8
- **Objective**: Minimize queue variance across all stops
- **Grading**: Variance reduction (40%) + throughput (25%) + rush handling (20%) + satisfaction (15%)
- **Expected Baseline**: 0.3–0.5

### Task 3 — Handle Demand Spike (Hard)
- **Scenario**: Concert causes 5x demand surge at tick 5 + rain at tick 10
- **Objective**: Detect spike, reallocate resources, recover stability
- **Grading**: Response speed (30%) + recovery (25%) + service (25%) + abandonment prevention (20%)
- **Expected Baseline**: 0.2–0.4

### Task 4 — Multi-Objective Optimization (Expert)
- **Scenario**: Rush hour + thunderstorm + bus breakdown + road construction simultaneously
- **Objective**: Balance throughput, wait times, coverage, and satisfaction
- **Grading**: Pareto-weighted across 4 metrics with abandonment penalty
- **Expected Baseline**: 0.1–0.3

---

## 🏗️ Reward Function

Continuous composite reward with 6 components — provides signal every step:

```
reward = 
    + 0.30 × wait_time_score         (1 - normalized_avg_wait)
    + 0.25 × occupancy_score         (optimal at 60-80%)
    + 0.20 × throughput_score         (passengers served / generated)
    + 0.15 × satisfaction_score       (avg passenger satisfaction)
    - 0.05 × idle_penalty             (idle buses ratio)
    - 0.05 × overcrowding_penalty     (overcrowded stops ratio)
```

Range: `[-0.10, 1.0]` — penalizes terrible decisions, rewards continuous improvement.

---

## 📦 Project Structure

```
dynamic-transit-rl/
├── __init__.py                     # Package exports
├── client.py                       # TransitEnv(MCPToolClient)
├── openenv.yaml                    # OpenEnv manifest
├── pyproject.toml                  # Dependencies
├── inference.py                    # Baseline agent script
├── Dockerfile                      # Container setup
├── README.md                       # This file
├── server/
│   ├── app.py                      # FastAPI server
│   ├── transit_environment.py      # MCPEnvironment implementation
│   ├── reward.py                   # Composite reward calculator
│   ├── simulation/
│   │   ├── city_network.py         # City topology (12 stops, 4 routes)
│   │   ├── bus.py                  # Bus models (3 types, fleet of 8)
│   │   ├── passenger.py            # Poisson-distributed arrivals
│   │   ├── events.py               # Dynamic event system
│   │   └── engine.py               # Core simulation engine
│   ├── tasks/
│   │   ├── task_overcrowding.py    # Easy
│   │   ├── task_load_balance.py    # Medium
│   │   ├── task_demand_spike.py    # Hard
│   │   └── task_multi_obj.py       # Expert
│   └── graders/
│       ├── overcrowding.py         # Grader 1
│       ├── load_balance.py         # Grader 2
│       ├── demand_spike.py         # Grader 3
│       └── multi_objective.py      # Grader 4
└── tests/
    └── test_environment.py         # Unit tests
```

---

## 🚀 Setup & Usage

### Prerequisites

```bash
pip install openenv-core fastapi uvicorn pydantic fastmcp openai
```

### Run Locally

```bash
# Start the environment server
cd dynamic-transit-rl
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, run inference
export HF_TOKEN="your-hf-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

### Docker

```bash
# Build
docker build -t transit-env .

# Run
docker run -p 8000:8000 transit-env

# Test health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
```

### OpenEnv Validate

```bash
cd dynamic-transit-rl
openenv validate
```

---

## 📊 Baseline Scores

| Task | Difficulty | Baseline Score | Model |
|------|-----------|---------------|-------|
| reduce_overcrowding | Easy | ~0.45 | Qwen2.5-72B |
| balance_load | Medium | ~0.35 | Qwen2.5-72B |
| demand_spike | Hard | ~0.25 | Qwen2.5-72B |
| multi_objective | Expert | ~0.15 | Qwen2.5-72B |

*Scores are reproducible with seed-based deterministic simulation.*

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | ✅ | — | HuggingFace API key |
| `API_BASE_URL` | ❌ | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | ❌ | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `IMAGE_NAME` | ❌ | `transit-env` | Docker image name |

---