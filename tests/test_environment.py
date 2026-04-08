"""
Basic tests for the Transit Environment.
Ensures reset, step, and graduation logic works correctly.
"""

import pytest
import asyncio
from server.transit_environment import TransitEnvironment
from openenv.core.env_server.mcp_types import CallToolAction


def test_environment_init():
    """Test that the environment initializes correctly."""
    env = TransitEnvironment()
    assert env is not None
    assert env._state.step_count == 0


def test_environment_reset():
    """Test that reset returns a valid observation with the initial state."""
    env = TransitEnvironment()
    obs = env.reset(task_name="reduce_overcrowding")
    
    assert obs.done is False
    assert obs.reward == 0.0
    assert "status" in obs.metadata
    assert "observation" in obs.metadata
    assert "stops" in obs.metadata["observation"]
    assert "buses" in obs.metadata["observation"]


def test_environment_step():
    """Test that an action step advances time and increments step_count."""
    env = TransitEnvironment()
    env.reset(task_name="reduce_overcrowding")
    
    # Action tool
    action = CallToolAction(tool_name="skip_action", arguments={})
    obs = env.step(action)
    
    assert env._state.step_count == 1
    
    # In MCP mode, results are serialized in obs.result.content
    import json
    result_content = obs.result.content[0].text
    result_dict = json.loads(result_content)
    assert "action_result" in result_dict


def test_info_tool_no_increment():
    """Test that info-only tools do not increment step_count."""
    env = TransitEnvironment()
    env.reset(task_name="reduce_overcrowding")
    
    # Observation tool
    action = CallToolAction(tool_name="get_system_status", arguments={})
    env.step(action)
    
    assert env._state.step_count == 0


def test_graduation():
    """Test that the environment graduates after max steps."""
    env = TransitEnvironment()
    env.reset(task_name="reduce_overcrowding")
    
    # Run for 20 steps
    action = CallToolAction(tool_name="skip_action", arguments={})
    for _ in range(20):
        obs = env.step(action)
    
    assert env._engine.done is True
    assert "final_score" in obs.metadata
    assert 0.0 <= obs.metadata["final_score"] <= 1.0


if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__])
