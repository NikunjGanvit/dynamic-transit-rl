"""
Inference Script — Dynamic Transit RL Environment
===================================================
Baseline agent that uses an LLM via OpenAI-compatible API to interact
with the transit environment and produce reproducible scores on all tasks.

MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI


# ─── Configuration ────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME = os.getenv("IMAGE_NAME", os.getenv("LOCAL_IMAGE_NAME", "transit-env"))

BENCHMARK = "dynamic_transit_rl"
MAX_STEPS = 20
TEMPERATURE = 0.3
MAX_TOKENS = 500

TASKS = [
    "reduce_overcrowding",   # Easy
    "balance_load",          # Medium
    "demand_spike",          # Hard
    "multi_objective",       # Expert
]


# ─── Logging Functions ────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Truncate action to avoid excessively long lines
    action_short = action[:200].replace("\n", " ") if action else "none"
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─── System Prompt ────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert urban transit operations manager AI agent. You are controlling
a city bus transit system through a simulation environment.

AVAILABLE TOOLS (call exactly ONE per turn):
1. get_system_status() - View all stops, buses, queues, events, metrics
2. get_route_analytics(route_id) - Detailed analytics for route R1/R2/R3/R4
3. get_task_info() - View your current task objective
4. reassign_bus(bus_id, target_route_id) - Move bus to different route
5. dispatch_bus(route_id, bus_type) - Deploy bus: standard(40)/articulated(80)/mini(20)
6. increase_frequency(route_id) - Speed up buses on a route by 30%
7. hold_bus(bus_id, duration) - Hold bus at stop for extra boarding (1-5 ticks)
8. skip_action() - Do nothing this step

RULES:
- You MUST respond with a JSON object: {"tool": "<tool_name>", "args": {<arguments>}}
- Call get_system_status() or get_task_info() first to understand the situation
- Then take action tools to improve the system
- Each action tool advances the simulation by one tick (~5 min)
- Observation tools do NOT advance time
- Think strategically: consider upcoming events and plan ahead
- Balance between gathering information and taking action

RESPONSE FORMAT (strict JSON only, no other text):
{"tool": "get_system_status", "args": {}}
{"tool": "dispatch_bus", "args": {"route_id": "R1", "bus_type": "standard"}}
{"tool": "reassign_bus", "args": {"bus_id": "B07", "target_route_id": "R4"}}
""").strip()


# ─── Agent Logic ──────────────────────────────────────────────

def parse_tool_call(response_text: str) -> Dict[str, Any]:
    """Parse the LLM's response into a tool call."""
    text = response_text.strip()
    
    # Try to extract JSON from the response
    # Handle cases where model wraps in markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "tool" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in the text
    for i in range(len(text)):
        if text[i] == '{':
            for j in range(len(text) - 1, i, -1):
                if text[j] == '}':
                    try:
                        parsed = json.loads(text[i:j+1])
                        if isinstance(parsed, dict) and "tool" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        continue
    
    # Fallback: default action
    return {"tool": "skip_action", "args": {}}


def get_llm_action(
    client: OpenAI,
    system_status: str,
    task_info: str,
    history: List[str],
    step: int,
) -> Dict[str, Any]:
    """Query the LLM for the next action."""
    history_block = "\n".join(history[-6:]) if history else "None"
    
    user_prompt = textwrap.dedent(f"""
    Current Step: {step}/{MAX_STEPS}
    
    Task Info:
    {task_info[:1000]}
    
    System Status:
    {system_status[:3000]}
    
    Recent History:
    {history_block}
    
    What action should you take? Respond with a JSON tool call.
    """).strip()
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_tool_call(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"tool": "skip_action", "args": {}}


# ─── Environment Interaction ─────────────────────────────────

def run_tool_local(env_module, tool_name: str, args: dict) -> str:
    """Run a tool call directly against the local environment."""
    from server.transit_environment import TransitEnvironment
    
    # The environment tools are registered on the MCP server
    # For local usage, we call the engine directly
    engine = env_module._engine
    
    if tool_name == "get_system_status":
        obs = engine.get_observation()
        return json.dumps(obs, indent=2)
    elif tool_name == "get_route_analytics":
        route_id = args.get("route_id", "R1")
        if route_id not in engine.network.routes:
            return json.dumps({"error": f"Route {route_id} not found"})
        route = engine.network.routes[route_id]
        buses = [b.to_dict() for b in engine.buses.values() if b.route_id == route_id]
        stop_queues = {}
        for sid in route.stop_ids:
            stop_queues[sid] = {
                "name": engine.network.stops[sid].name,
                "queue_size": engine.passenger_gen.get_queue_size(sid),
            }
        return json.dumps({"route_id": route_id, "buses": buses, "stops": stop_queues}, indent=2)
    elif tool_name == "get_task_info":
        info = {
            "task": env_module._current_task.to_dict() if env_module._current_task else None,
            "tick": engine.tick,
            "max_ticks": engine.max_ticks,
        }
        return json.dumps(info, indent=2)
    elif tool_name == "reassign_bus":
        result = engine.action_reassign_bus(args.get("bus_id", ""), args.get("target_route_id", ""))
        env_module._advance_and_compute()
        return result
    elif tool_name == "dispatch_bus":
        result = engine.action_dispatch_bus(args.get("route_id", ""), args.get("bus_type", "standard"))
        env_module._advance_and_compute()
        return result
    elif tool_name == "increase_frequency":
        result = engine.action_increase_frequency(args.get("route_id", ""))
        env_module._advance_and_compute()
        return result
    elif tool_name == "hold_bus":
        result = engine.action_hold_bus(args.get("bus_id", ""), args.get("duration", 2))
        env_module._advance_and_compute()
        return result
    elif tool_name == "skip_action":
        result = engine.action_skip()
        env_module._advance_and_compute()
        return result
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


# Action tools that advance the simulation
ACTION_TOOLS = {"reassign_bus", "dispatch_bus", "increase_frequency", "hold_bus", "skip_action"}


def run_task(client: OpenAI, task_name: str) -> float:
    """Run a single task and return the score."""
    from server.transit_environment import TransitEnvironment
    from server.graders import GRADER_REGISTRY
    
    # Create environment locally
    env = TransitEnvironment()
    
    # Reset with task
    from server.tasks import TASK_REGISTRY
    env._current_task_name = task_name
    env._current_task = TASK_REGISTRY[task_name]()
    config = env._current_task.get_config()
    
    from server.simulation.engine import SimulationEngine
    from server.reward import RewardCalculator
    
    env._engine = SimulationEngine(seed=config.get("seed", 42))
    env._reward_calc = RewardCalculator()
    env._rewards = []
    env._last_reward = 0.0
    env._engine.initialize(config)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Get initial info
        task_info = run_tool_local(env, "get_task_info", {})
        system_status = run_tool_local(env, "get_system_status", {})
        
        for step in range(1, MAX_STEPS + 1):
            if env._engine.done:
                break
            
            # Get LLM decision
            tool_call = get_llm_action(client, system_status, task_info, history, step)
            tool_name = tool_call.get("tool", "skip_action")
            tool_args = tool_call.get("args", {})
            
            # Execute tool
            result = run_tool_local(env, tool_name, tool_args)
            
            # Track reward only for action tools (that advance simulation)
            is_action = tool_name in ACTION_TOOLS
            
            if is_action:
                reward = env._last_reward
                rewards.append(reward)
                steps_taken = step
                
                error = env._engine._last_action_error
                log_step(
                    step=steps_taken,
                    action=f"{tool_name}({json.dumps(tool_args)})",
                    reward=reward,
                    done=env._engine.done,
                    error=error,
                )
                
                history.append(
                    f"Step {steps_taken}: {tool_name}({tool_args}) → reward={reward:.2f}"
                )
            else:
                # Observation tools don't advance simulation
                history.append(f"Info: {tool_name}() → got status")
            
            # Update status for next iteration
            if not env._engine.done:
                system_status = run_tool_local(env, "get_system_status", {})
            
            if env._engine.done:
                break
        
        # Compute final score using grader
        if task_name in GRADER_REGISTRY:
            grader = GRADER_REGISTRY[task_name]()
            score = grader.grade(env._engine)
        else:
            score = sum(rewards) / (MAX_STEPS * 1.0) if rewards else 0.0
        
        score = max(0.0, min(1.0, score))
        success = score >= 0.1
    
    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return score


def main() -> None:
    """Run inference on all tasks."""
    if not API_KEY:
        print("[DEBUG] WARNING: No API key found. Set HF_TOKEN or OPENAI_API_KEY.", flush=True)
        print("[DEBUG] Running with dummy responses.", flush=True)
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")
    
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] Tasks: {TASKS}", flush=True)
    print("", flush=True)
    
    scores = {}
    start_time = time.time()
    
    for task_name in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"[DEBUG] Starting task: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)
        
        task_start = time.time()
        score = run_task(client, task_name)
        task_elapsed = time.time() - task_start
        
        scores[task_name] = score
        print(f"[DEBUG] Task {task_name} completed in {task_elapsed:.1f}s, score={score:.2f}", flush=True)
    
    total_elapsed = time.time() - start_time
    
    print(f"\n{'='*60}", flush=True)
    print(f"[DEBUG] All tasks completed in {total_elapsed:.1f}s", flush=True)
    print(f"[DEBUG] Final scores:", flush=True)
    for task_name, score in scores.items():
        print(f"[DEBUG]   {task_name}: {score:.2f}", flush=True)
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"[DEBUG] Average score: {avg_score:.2f}", flush=True)


if __name__ == "__main__":
    main()
