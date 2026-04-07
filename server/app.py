"""
FastAPI application for the Transit Environment.

Creates an HTTP server that exposes the TransitEnvironment
over HTTP and WebSocket endpoints, compatible with MCPToolClient.
"""

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .transit_environment import TransitEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.transit_environment import TransitEnvironment

app = create_app(
    TransitEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="transit_env",
)


def main():
    """
    Entry point for direct execution.
    
    Usage:
        uv run --project . server
        python -m server.app
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
