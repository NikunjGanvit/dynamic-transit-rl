# Dynamic Transit RL — Urban Transit Operations Environment

Dynamic Transit RL is a production-grade [OpenEnv](https://openenv.ai) environment for evaluating multi-agent and frontier AI models on urban public transport dispatching and operations management.

## 🚍 The Mission
You are the Transit Operations Manager for a major city. Your goal is to manage a fleet of buses across 4 critical routes to minimize passenger wait times, prevent stop overcrowding, and maintain high service satisfaction—even as hidden events (weather, demand spikes, breakdowns) disrupt your network.

## 🛠️ Action Space
The agent can call the following tools to manage the fleet:
- `get_system_status()`: Full snapshot of stops, buses, and metrics.
- `dispatch_bus(route_id, bus_type)`: Deploy a new bus from the depot.
- `reassign_bus(bus_id, target_route_id)`: Move a bus to where it's needed most.
- `increase_frequency(route_id)`: Speed up buses to handle surge demand.
- `hold_bus(bus_id, duration)`: Extend boarding time at overcrowded stops.
- `skip_action()`: Advance simulation by one tick.

## 📊 Tasks & Graders
1. **Reduce Overcrowding (Easy)**: High initial queues at key hubs.
2. **Balance Load (Medium)**: Uneven distribution across routes.
3. **Demand Spike (Hard)**: Hidden events (Concert + Weather) cause sudden surges.
4. **Multi-Objective (Expert)**: Cascading events, multiple breakdowns, and escalating crises.

All graders return a score in `[0.0, 1.0]` based on throughput, satisfaction, and crisis response speed.

## 🚀 Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerized evaluation)

### Local Dev Setup
```bash
# 1. Clone & Setup
git clone <repo-url>
cd dynamic-transit-rl
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt

# 2. Run Validation
openenv validate
```

---

## 📊 Baseline Scores

| Task | Difficulty | Baseline Score | Model |
|------|-----------|---------------|-------|
| reduce_overcrowding | Easy | ~0.450 | Qwen2.5-72B |
| balance_load | Medium | ~0.350 | Qwen2.5-72B |
| demand_spike | Hard | ~0.250 | Qwen2.5-72B |
| multi_objective | Expert | ~0.150 | Qwen2.5-72B |

*Scores are reproducible with seed-based deterministic simulation.*

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | ✅ | — | HuggingFace API key |
| `API_BASE_URL` | ❌ | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | ❌ | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |

---

## ⚖️ Simulating a Judge's Evaluation

To test your submission exactly as a judge would:

### 1. Set Environment Variables
The judges will run your `inference.py` in an environment where these are strictly defined. Create a `.env` file or export them:
```bash
export HF_TOKEN="your_token_here"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
```

### 2. Start the Environment (Server)
In one terminal, start the transit server. This is the "environment" the agent will interact with:
```bash
source .venv/bin/activate
export PYTHONPATH=.
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Run the Inference Script (Agent)
In another terminal, run the agent script. This will talk to the server and execute the evaluation:
```bash
source .venv/bin/activate
export PYTHONPATH=.
python inference.py
```

### 4. Verify Output
Check that the stdout emits **only** structured blocks. Note that:
- The script must follow the `[START]/[STEP]/[END]` protocol strictly.
- Success is determined by a score threshold (usually 0.1+).

---

## 📂 Project Structure
- `inference.py`: Standard entry point for judges to run evaluation.
- `server/`: Core simulation logic, MCP tools, and task definitions.
- `client.py`: Public API client for environment interaction.
- `tests/`: Basic verification suite for environment logic.
- `openenv.yaml`: Metadata for OpenEnv registry.
- `Dockerfile`: Container configuration for Hugging Face Spaces.