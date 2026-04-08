"""
Inference Script — Dynamic Transit RL Environment
===================================================
Final production version strictly complying with OpenEnv Hackathon requirements.
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI
from client import TransitEnv


# ─── Configuration ────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "dynamic_transit_rl"
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
TEMPERATURE = 0.3
MAX_TOKENS = 500

_tasks_env = os.getenv("TASKS")
if _tasks_env:
    try:
        TASKS = json.loads(_tasks_env.replace("'", '"'))
    except:
        TASKS = [t.strip() for t in _tasks_env.split(",")]
else:
    TASKS = [
        "reduce_overcrowding",
        "balance_load",
        "demand_spike",
        "multi_objective",
    ]


# ─── Logging Functions ────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_short = action[:200].replace("\n", " ") if action else "none"
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── System Prompt ────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert urban transit operations manager AI agent.
    Available Tools:
    - get_system_status()
    - get_route_analytics(route_id)
    - get_task_info()
    - reassign_bus(bus_id, target_route_id)
    - dispatch_bus(route_id, bus_type)
    - increase_frequency(route_id)
    - hold_bus(bus_id, duration)
    - skip_action()

    Rules:
    - Respond strictly with a JSON object: {"tool": "tool_name", "args": {}}
    - Action tools advance time by 5 mins. Observation tools do not.
    - Your goal is to maximize system efficiency and passenger satisfaction.
""").strip()


# ─── Agent Logic ──────────────────────────────────────────────

def parse_tool_call(response_text: str) -> Dict[str, Any]:
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "tool" in parsed:
            return parsed
    except:
        pass
    
    # Fallback to simple scan
    for i in range(len(text)):
        if text[i] == '{':
            for j in range(len(text) - 1, i, -1):
                if text[j] == '}':
                    try:
                        parsed = json.loads(text[i:j+1])
                        if isinstance(parsed, dict) and "tool" in parsed:
                            return parsed
                    except: continue
    return {"tool": "skip_action", "args": {}}


def get_llm_action(client: OpenAI, obs_str: str, history: List[str], step: int) -> Dict[str, Any]:
    history_block = "\n".join(history[-5:]) if history else "None"
    user_prompt = f"Step: {step}/20\n\nStatus:\n{obs_str[:3000]}\n\nHistory:\n{history_block}\n\nAction JSON:"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return parse_tool_call(completion.choices[0].message.content or "")
    except:
        return {"tool": "skip_action", "args": {}}


# ─── Environment Interaction ─────────────────────────────────

ACTION_TOOLS = {"reassign_bus", "dispatch_bus", "increase_frequency", "hold_bus", "skip_action"}

async def run_task(client: OpenAI, task_name: str) -> float:
    async with TransitEnv(base_url="http://localhost:8000") as env:
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        
        rewards = []
        history = []
        steps_taken = 0
        final_score = 0.0
        
        try:
            # Initial state
            res = await env.reset(task_name=task_name)
            system_status = await env.call_tool("get_system_status")
            
            for step in range(1, MAX_STEPS + 1):
                if res.done: break
                
                tool_call = get_llm_action(client, system_status, history, step)
                t_name, t_args = tool_call.get("tool", "skip_action"), tool_call.get("args", {})
                
                res = await env.call_tool(t_name, **t_args)
                
                if t_name in ACTION_TOOLS:
                    reward = res.reward or 0.0
                    rewards.append(reward)
                    steps_taken += 1
                    
                    log_step(step=steps_taken, action=f"{t_name}({json.dumps(t_args)})", 
                             reward=reward, done=res.done, error=None)
                    
                    history.append(f"Step {steps_taken}: {t_name} -> {reward:.2f}")
                
                if res.done: break
                system_status = await env.call_tool("get_system_status")

            final_score = res.metadata.get("final_score", 0.0) if res.metadata else 0.0
            
        except:
            pass
        finally:
            log_end(success=final_score >= 0.1, steps=steps_taken, score=final_score, rewards=rewards)
            return final_score


async def main() -> None:
    # Ensure server is running before starting (judges handle this, but for local we check)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")
    for task in TASKS:
        await run_task(client, task)


if __name__ == "__main__":
    asyncio.run(main())
