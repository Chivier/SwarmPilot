"""Chat Completions API routes.

Provides OpenAI-compatible chat completions endpoint with SwarmX extensions.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from src.services.chat import ChatService, FileRefNotFoundError

router = APIRouter(tags=["chat"])

# Service instance (set during app startup)
_chat_service: ChatService | None = None


def set_chat_service(service: ChatService) -> None:
    """Set the chat service instance.

    Args:
        service: ChatService instance to use.
    """
    global _chat_service
    _chat_service = service


def get_chat_service() -> ChatService:
    """Get the chat service instance.

    Returns:
        The ChatService instance.

    Raises:
        RuntimeError: If chat service not initialized.
    """
    if _chat_service is None:
        raise RuntimeError("Chat service not initialized")
    return _chat_service


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """Create a chat completion.

    OpenAI-compatible endpoint for generating chat completions.
    Supports both streaming and non-streaming modes.

    Args:
        request: Chat completion request with model, messages, and options.

    Returns:
        ChatCompletionResponse for non-streaming, StreamingResponse for streaming.

    Raises:
        HTTPException: 400 if file_refs contains invalid file IDs.
    """
    chat = get_chat_service()

    try:
        if request.stream:
            # Streaming response
            return StreamingResponse(
                chat.complete_stream(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Non-streaming response
            return await chat.complete(request)

    except FileRefNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_file_ref",
                "message": str(e),
                "file_id": e.file_id,
            },
        )
