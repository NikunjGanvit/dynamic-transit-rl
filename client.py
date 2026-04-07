"""
Transit Environment Client.

Provides the client for connecting to a Transit Environment server.
TransitEnv extends MCPToolClient for MCP tool-calling interactions.
"""

from openenv.core.mcp_client import MCPToolClient


class TransitEnv(MCPToolClient):
    """
    Client for the Transit Environment.
    
    Inherits all functionality from MCPToolClient:
    - list_tools(): Discover available tools
    - call_tool(name, **kwargs): Call a tool by name
    - reset(**kwargs): Reset the environment
    - step(action): Execute an action
    
    Example:
        >>> with TransitEnv(base_url="http://localhost:8000") as env:
        ...     env.reset(task_name="reduce_overcrowding")
        ...     tools = env.list_tools()
        ...     status = env.call_tool("get_system_status")
        ...     result = env.call_tool("dispatch_bus", route_id="R1", bus_type="standard")
    
    Example with Docker:
        >>> env = TransitEnv.from_docker_image("transit-env:latest")
        >>> try:
        ...     env.reset(task_name="demand_spike")
        ...     tools = env.list_tools()
        ... finally:
        ...     env.close()
    """
    pass
