"""WebSocket client singleton for global access."""

from src.websocket_client import WebSocketClient

_websocket_client: WebSocketClient | None = None


def get_websocket_client() -> WebSocketClient | None:
    """Get the global WebSocket client instance."""
    return _websocket_client


def set_websocket_client(client: WebSocketClient) -> None:
    """Set the global WebSocket client instance."""
    global _websocket_client
    _websocket_client = client
