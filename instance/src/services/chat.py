"""Chat service for handling chat completions.

Provides OpenAI-compatible chat completion functionality with SwarmX extensions.
"""

import time
import uuid
from collections.abc import AsyncGenerator

from src.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from src.services.file_storage import FileNotFoundError, FileStorageService
from src.services.inference_manager import InferenceManagerService


class FileRefNotFoundError(Exception):
    """Raised when a referenced file does not exist."""

    def __init__(self, file_id: str):
        """Initialize with file ID.

        Args:
            file_id: The ID of the file that was not found.
        """
        self.file_id = file_id
        super().__init__(f"Referenced file not found: {file_id}")


class ChatService:
    """Service for handling chat completions.

    This service provides OpenAI-compatible chat completion functionality,
    including support for streaming and the SwarmX file_refs extension.

    In production, this would forward requests to an inference server (e.g., vLLM).
    For testing, it generates mock responses.

    Attributes:
        inference_manager: Service for managing inference server.
        file_storage: Service for file storage operations.
    """

    def __init__(
        self,
        inference_manager: InferenceManagerService,
        file_storage: FileStorageService,
    ):
        """Initialize the chat service.

        Args:
            inference_manager: InferenceManagerService instance.
            file_storage: FileStorageService instance.
        """
        self.inference_manager = inference_manager
        self.file_storage = file_storage

    async def _resolve_file_refs(
        self,
        file_refs: list[str] | None,
    ) -> list[str]:
        """Resolve file references to their content.

        Args:
            file_refs: List of file IDs to resolve.

        Returns:
            List of file contents as strings.

        Raises:
            FileRefNotFoundError: If a file_id doesn't exist.
        """
        if not file_refs:
            return []

        contents = []
        for file_id in file_refs:
            try:
                file_path, _info = await self.file_storage.get_file(file_id)
                # Read content from file path
                content = file_path.read_bytes()
                # Decode bytes to string
                try:
                    contents.append(content.decode("utf-8"))
                except UnicodeDecodeError:
                    contents.append(content.decode("latin-1"))
            except FileNotFoundError:
                raise FileRefNotFoundError(file_id)

        return contents

    def _inject_file_context(
        self,
        messages: list[ChatMessage],
        file_contents: list[str],
    ) -> list[ChatMessage]:
        """Inject file contents into messages.

        Prepends file contents to the system message or creates a new
        system message if none exists.

        Args:
            messages: Original messages.
            file_contents: File contents to inject.

        Returns:
            Modified messages with file context.
        """
        if not file_contents:
            return messages

        # Build context string
        context_parts = []
        for i, content in enumerate(file_contents):
            context_parts.append(f"[File {i + 1}]\n{content}")
        file_context = "\n\n".join(context_parts)

        # Create new messages list
        new_messages = []

        # Check if first message is system
        if messages and messages[0].role == "system":
            # Prepend file context to system message
            new_system = ChatMessage(
                role="system",
                content=f"{file_context}\n\n{messages[0].content}",
            )
            new_messages.append(new_system)
            new_messages.extend(messages[1:])
        else:
            # Create new system message with file context
            new_system = ChatMessage(
                role="system",
                content=f"Reference materials:\n{file_context}",
            )
            new_messages.append(new_system)
            new_messages.extend(messages)

        return new_messages

    def _generate_completion_id(self) -> str:
        """Generate a unique completion ID.

        Returns:
            Unique completion ID in OpenAI format.
        """
        return f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Generate a chat completion (non-streaming).

        Args:
            request: Chat completion request.

        Returns:
            Chat completion response.

        Raises:
            FileRefNotFoundError: If a file_ref doesn't exist.
        """
        # Resolve file references
        file_contents = await self._resolve_file_refs(request.file_refs)

        # Inject file context into messages
        messages = self._inject_file_context(request.messages, file_contents)

        # In production, this would forward to inference server
        # For now, generate a mock response
        completion_id = self._generate_completion_id()
        created = int(time.time())

        # Mock response content
        response_content = "Hello! How can I help you today?"

        # Mock token counts
        prompt_tokens = sum(len(m.content.split()) for m in messages) * 4
        completion_tokens = len(response_content.split()) * 4

        return ChatCompletionResponse(
            id=completion_id,
            object="chat.completion",
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_content),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def complete_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming chat completion.

        Args:
            request: Chat completion request.

        Yields:
            SSE-formatted chunks.

        Raises:
            FileRefNotFoundError: If a file_ref doesn't exist.
        """
        # Resolve file references
        file_contents = await self._resolve_file_refs(request.file_refs)

        # Inject file context into messages (used in production for inference server)
        _messages = self._inject_file_context(request.messages, file_contents)

        # Generate IDs and timestamp
        completion_id = self._generate_completion_id()
        created = int(time.time())

        # Mock response - in production would use _messages with inference server
        response_content = "Hello! How can I help you today?"
        words = response_content.split()

        # First chunk with role
        first_chunk = ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionDelta(role="assistant", content=None),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        # Content chunks
        for word in words:
            chunk = ChatCompletionChunk(
                id=completion_id,
                object="chat.completion.chunk",
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionDelta(role=None, content=word + " "),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionDelta(role=None, content=None),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"

        # Done marker
        yield "data: [DONE]\n\n"
