"""HuggingFace Downloader service for model downloads.

Handles downloading models from HuggingFace Hub with progress tracking,
authentication support, and custom endpoint configuration.
"""

import asyncio
import os
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError


class HuggingFaceDownloadError(Exception):
    """Raised when a HuggingFace download fails.

    Attributes:
        repo: The repository that failed to download.
        error_type: Type of error (e.g., 'repository_not_found').
        message: Human-readable error message.
    """

    def __init__(
        self,
        repo: str,
        error_type: str,
        message: str,
    ):
        """Initialize with download error details.

        Args:
            repo: The repository that failed to download.
            error_type: Type of error.
            message: Human-readable error message.
        """
        self.repo = repo
        self.error_type = error_type
        self.message = message
        super().__init__(f"Failed to download {repo}: {message}")


class ProgressTracker:
    """Wrapper class to track download progress via tqdm interface.

    This class implements a tqdm-like interface that forwards progress
    updates to a callback function, allowing integration with the
    task tracking service.

    Attributes:
        callback: Progress callback function.
        total: Total bytes to download.
        completed: Bytes downloaded so far.
    """

    def __init__(
        self,
        callback: Callable[[int, int], None],
        total: int | None = None,
        **kwargs,
    ):
        """Initialize the progress tracker.

        Args:
            callback: Function to call with (completed, total) on updates.
            total: Total bytes to download.
            **kwargs: Additional tqdm-compatible kwargs (ignored).
        """
        self.callback = callback
        self.total = total or 0
        self.completed = 0

    def update(self, n: int = 1) -> None:
        """Update progress by n bytes.

        Args:
            n: Number of bytes completed in this update.
        """
        self.completed += n
        self.callback(self.completed, self.total)

    def close(self) -> None:
        """Close the progress tracker."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()


class HuggingFaceDownloader:
    """Service for downloading models from HuggingFace Hub.

    Provides functionality to download model weights from HuggingFace Hub
    with authentication support, custom endpoints, and progress tracking.

    Attributes:
        default_endpoint: Default HuggingFace Hub URL.
    """

    def __init__(self):
        """Initialize the HuggingFace downloader."""
        self.default_endpoint = "https://huggingface.co"

    def _validate_repo(self, repo: str) -> None:
        """Validate repository format.

        Args:
            repo: Repository identifier (org/model-name format).

        Raises:
            HuggingFaceDownloadError: If repo format is invalid.
        """
        if not repo or "/" not in repo:
            raise HuggingFaceDownloadError(
                repo=repo or "",
                error_type="invalid_repo_format",
                message="Repository must be in 'organization/model-name' format",
            )

    def _create_progress_class(
        self,
        callback: Callable[[int, int], None],
    ) -> type:
        """Create a tqdm-compatible class that forwards to callback.

        Args:
            callback: Progress callback function.

        Returns:
            A class that wraps progress updates.
        """

        class CallbackProgress(ProgressTracker):
            def __init__(self, *args, **kwargs):
                kwargs["callback"] = callback
                super().__init__(**kwargs)

        return CallbackProgress

    async def download_model(
        self,
        task_id: str,
        model_id: str,
        repo: str,
        revision: str,
        endpoint: str | None,
        token: str | None,
        dest_dir: Path,
        progress_callback: Callable[[int, int], None],
    ) -> str:
        """Download a model from HuggingFace Hub.

        Args:
            task_id: Associated task identifier for tracking.
            model_id: Local model identifier.
            repo: HuggingFace repository (org/model-name format).
            revision: Git revision (branch, tag, or commit SHA).
            endpoint: Custom HuggingFace endpoint URL (or None for default).
            token: HuggingFace authentication token (or None for public repos).
            dest_dir: Local directory to download to.
            progress_callback: Function to call with (bytes_completed, bytes_total).

        Returns:
            Path to the downloaded model directory.

        Raises:
            HuggingFaceDownloadError: If download fails.
        """
        # Validate inputs
        self._validate_repo(repo)

        # Set up endpoint
        original_endpoint = os.environ.get("HF_ENDPOINT")
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint

        try:
            # Create progress tracking class
            progress_class = self._create_progress_class(progress_callback)

            # Run download in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=repo,
                    revision=revision,
                    local_dir=str(dest_dir),
                    token=token,
                    tqdm_class=progress_class,
                ),
            )

            return result

        except GatedRepoError:
            raise HuggingFaceDownloadError(
                repo=repo,
                error_type="authentication_required",
                message=(
                    f"Repository '{repo}' requires authentication. "
                    "Please provide a valid HuggingFace token."
                ),
            )

        except RepositoryNotFoundError:
            raise HuggingFaceDownloadError(
                repo=repo,
                error_type="repository_not_found",
                message=f"Repository '{repo}' not found on HuggingFace Hub",
            )

        except Exception as e:
            raise HuggingFaceDownloadError(
                repo=repo,
                error_type="download_failed",
                message=str(e),
            )

        finally:
            # Restore original endpoint
            if endpoint:
                if original_endpoint:
                    os.environ["HF_ENDPOINT"] = original_endpoint
                else:
                    os.environ.pop("HF_ENDPOINT", None)

    async def get_model_size(
        self,
        repo: str,
        revision: str,
        endpoint: str | None,
        token: str | None,
    ) -> int | None:
        """Get the total size of a model on HuggingFace Hub.

        Args:
            repo: HuggingFace repository (org/model-name format).
            revision: Git revision (branch, tag, or commit SHA).
            endpoint: Custom HuggingFace endpoint URL (or None for default).
            token: HuggingFace authentication token (or None for public repos).

        Returns:
            Total size in bytes, or None if not available.
        """
        try:
            api = HfApi(endpoint=endpoint, token=token)

            # Get model info in thread pool
            loop = asyncio.get_event_loop()
            model_info = await loop.run_in_executor(
                None,
                lambda: api.model_info(repo, revision=revision),
            )

            # Calculate total size from siblings
            total_size = 0
            if model_info.siblings:
                for sibling in model_info.siblings:
                    if sibling.size:
                        total_size += sibling.size

            return total_size

        except RepositoryNotFoundError:
            return None

        except Exception:
            return None
