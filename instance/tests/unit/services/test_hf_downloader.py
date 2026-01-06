"""Unit tests for HuggingFace Downloader service.

Tests for EDI-46 - HuggingFace download background task.
Tests follow TDD principle - written before implementation.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def temp_dest_dir(tmp_path):
    """Create a temporary destination directory."""
    dest_dir = tmp_path / "models"
    dest_dir.mkdir()
    return dest_dir


@pytest.fixture
def downloader():
    """Create a HuggingFaceDownloader instance."""
    from src.services.hf_downloader import HuggingFaceDownloader

    return HuggingFaceDownloader()


class TestHuggingFaceDownloaderInit:
    """Tests for HuggingFaceDownloader initialization."""

    def test_init_creates_instance(self, downloader):
        """Test that downloader can be instantiated."""
        assert downloader is not None

    def test_default_endpoint(self, downloader):
        """Test default HuggingFace endpoint is set."""
        assert downloader.default_endpoint == "https://huggingface.co"


class TestDownloadModel:
    """Tests for download_model method."""

    @pytest.mark.asyncio
    async def test_download_model_success(self, downloader, temp_dest_dir):
        """Test successful model download."""
        progress_calls = []

        def progress_callback(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        with patch(
            "src.services.hf_downloader.snapshot_download"
        ) as mock_download:
            mock_download.return_value = str(temp_dest_dir / "test_model")

            # Create mock downloaded files
            model_dir = temp_dest_dir / "test_model"
            model_dir.mkdir(parents=True)
            (model_dir / "model.safetensors").write_bytes(b"fake model data")

            result = await downloader.download_model(
                task_id="task_abc123",
                model_id="model_xyz789",
                repo="test-org/test-model",
                revision="main",
                endpoint=None,
                token=None,
                dest_dir=temp_dest_dir,
                progress_callback=progress_callback,
            )

            # Verify snapshot_download was called
            mock_download.assert_called_once()
            call_kwargs = mock_download.call_args.kwargs
            assert call_kwargs["repo_id"] == "test-org/test-model"
            assert call_kwargs["revision"] == "main"
            assert call_kwargs["local_dir"] == str(temp_dest_dir)

            # Verify result
            assert result == str(temp_dest_dir / "test_model")

    @pytest.mark.asyncio
    async def test_download_model_with_token(self, downloader, temp_dest_dir):
        """Test download with authentication token."""
        with patch(
            "src.services.hf_downloader.snapshot_download"
        ) as mock_download:
            mock_download.return_value = str(temp_dest_dir)

            await downloader.download_model(
                task_id="task_abc123",
                model_id="model_xyz789",
                repo="test-org/private-model",
                revision="main",
                endpoint=None,
                token="hf_secret_token",
                dest_dir=temp_dest_dir,
                progress_callback=lambda c, t: None,
            )

            call_kwargs = mock_download.call_args.kwargs
            assert call_kwargs["token"] == "hf_secret_token"

    @pytest.mark.asyncio
    async def test_download_model_with_custom_endpoint(
        self, downloader, temp_dest_dir
    ):
        """Test download from custom HuggingFace endpoint."""
        with patch(
            "src.services.hf_downloader.snapshot_download"
        ) as mock_download:
            mock_download.return_value = str(temp_dest_dir)

            await downloader.download_model(
                task_id="task_abc123",
                model_id="model_xyz789",
                repo="test-org/test-model",
                revision="v1.0",
                endpoint="https://hf-mirror.com",
                token=None,
                dest_dir=temp_dest_dir,
                progress_callback=lambda c, t: None,
            )

            # Check endpoint was set via environment or passed
            mock_download.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_model_with_revision(
        self, downloader, temp_dest_dir
    ):
        """Test download with specific revision."""
        with patch(
            "src.services.hf_downloader.snapshot_download"
        ) as mock_download:
            mock_download.return_value = str(temp_dest_dir)

            await downloader.download_model(
                task_id="task_abc123",
                model_id="model_xyz789",
                repo="test-org/test-model",
                revision="v2.0.1",
                endpoint=None,
                token=None,
                dest_dir=temp_dest_dir,
                progress_callback=lambda c, t: None,
            )

            call_kwargs = mock_download.call_args.kwargs
            assert call_kwargs["revision"] == "v2.0.1"

    @pytest.mark.asyncio
    async def test_download_model_error_handling(
        self, downloader, temp_dest_dir
    ):
        """Test error handling during download."""
        from src.services.hf_downloader import HuggingFaceDownloadError

        with patch(
            "src.services.hf_downloader.snapshot_download"
        ) as mock_download:
            mock_download.side_effect = Exception("Network error")

            with pytest.raises(HuggingFaceDownloadError) as exc_info:
                await downloader.download_model(
                    task_id="task_abc123",
                    model_id="model_xyz789",
                    repo="test-org/test-model",
                    revision="main",
                    endpoint=None,
                    token=None,
                    dest_dir=temp_dest_dir,
                    progress_callback=lambda c, t: None,
                )

            assert "Network error" in str(exc_info.value)
            assert exc_info.value.repo == "test-org/test-model"

    @pytest.mark.asyncio
    async def test_download_model_repo_not_found(
        self, downloader, temp_dest_dir
    ):
        """Test handling of repository not found error."""
        from huggingface_hub.utils import RepositoryNotFoundError

        from src.services.hf_downloader import HuggingFaceDownloadError

        with patch(
            "src.services.hf_downloader.snapshot_download"
        ) as mock_download:
            mock_download.side_effect = RepositoryNotFoundError(
                "Repository not found"
            )

            with pytest.raises(HuggingFaceDownloadError) as exc_info:
                await downloader.download_model(
                    task_id="task_abc123",
                    model_id="model_xyz789",
                    repo="nonexistent/model",
                    revision="main",
                    endpoint=None,
                    token=None,
                    dest_dir=temp_dest_dir,
                    progress_callback=lambda c, t: None,
                )

            assert exc_info.value.error_type == "repository_not_found"

    @pytest.mark.asyncio
    async def test_download_model_auth_required(
        self, downloader, temp_dest_dir
    ):
        """Test handling of authentication required error."""
        from huggingface_hub.utils import GatedRepoError

        from src.services.hf_downloader import HuggingFaceDownloadError

        with patch(
            "src.services.hf_downloader.snapshot_download"
        ) as mock_download:
            mock_download.side_effect = GatedRepoError(
                "Access denied: gated repository"
            )

            with pytest.raises(HuggingFaceDownloadError) as exc_info:
                await downloader.download_model(
                    task_id="task_abc123",
                    model_id="model_xyz789",
                    repo="private/gated-model",
                    revision="main",
                    endpoint=None,
                    token=None,
                    dest_dir=temp_dest_dir,
                    progress_callback=lambda c, t: None,
                )

            assert exc_info.value.error_type == "authentication_required"


class TestProgressCallback:
    """Tests for progress callback functionality."""

    @pytest.mark.asyncio
    async def test_progress_callback_called(self, downloader, temp_dest_dir):
        """Test progress callback is invoked during download."""
        progress_calls = []

        def progress_callback(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        with patch(
            "src.services.hf_downloader.snapshot_download"
        ) as mock_download:
            # Simulate progress updates
            def simulate_download(*args, **kwargs):
                # Get the tqdm_class and call it with progress
                tqdm_class = kwargs.get("tqdm_class")
                if tqdm_class:
                    # Create a mock progress bar instance
                    pbar = tqdm_class(total=1000)
                    pbar.update(500)
                    pbar.update(500)
                    pbar.close()
                return str(temp_dest_dir)

            mock_download.side_effect = simulate_download

            await downloader.download_model(
                task_id="task_abc123",
                model_id="model_xyz789",
                repo="test-org/test-model",
                revision="main",
                endpoint=None,
                token=None,
                dest_dir=temp_dest_dir,
                progress_callback=progress_callback,
            )

            # Progress callback should have been called
            assert len(progress_calls) > 0


class TestValidation:
    """Tests for input validation."""

    @pytest.mark.asyncio
    async def test_invalid_repo_format(self, downloader, temp_dest_dir):
        """Test validation of repository format."""
        from src.services.hf_downloader import HuggingFaceDownloadError

        with pytest.raises(HuggingFaceDownloadError) as exc_info:
            await downloader.download_model(
                task_id="task_abc123",
                model_id="model_xyz789",
                repo="invalid-repo-no-slash",  # Invalid format
                revision="main",
                endpoint=None,
                token=None,
                dest_dir=temp_dest_dir,
                progress_callback=lambda c, t: None,
            )

        assert exc_info.value.error_type == "invalid_repo_format"

    @pytest.mark.asyncio
    async def test_empty_repo(self, downloader, temp_dest_dir):
        """Test validation of empty repository."""
        from src.services.hf_downloader import HuggingFaceDownloadError

        with pytest.raises(HuggingFaceDownloadError) as exc_info:
            await downloader.download_model(
                task_id="task_abc123",
                model_id="model_xyz789",
                repo="",
                revision="main",
                endpoint=None,
                token=None,
                dest_dir=temp_dest_dir,
                progress_callback=lambda c, t: None,
            )

        assert exc_info.value.error_type == "invalid_repo_format"


class TestCalculateSize:
    """Tests for calculating download size."""

    @pytest.mark.asyncio
    async def test_get_model_size(self, downloader):
        """Test getting model size from HuggingFace."""
        with patch(
            "src.services.hf_downloader.HfApi"
        ) as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api

            # Mock model info with siblings
            mock_info = MagicMock()
            mock_sibling1 = MagicMock()
            mock_sibling1.size = 1000000000  # 1GB
            mock_sibling2 = MagicMock()
            mock_sibling2.size = 500000000  # 500MB
            mock_info.siblings = [mock_sibling1, mock_sibling2]
            mock_api.model_info.return_value = mock_info

            size = await downloader.get_model_size(
                repo="test-org/test-model",
                revision="main",
                endpoint=None,
                token=None,
            )

            assert size == 1500000000  # 1.5GB total

    @pytest.mark.asyncio
    async def test_get_model_size_not_found(self, downloader):
        """Test getting size for non-existent model."""
        from huggingface_hub.utils import RepositoryNotFoundError

        with patch(
            "src.services.hf_downloader.HfApi"
        ) as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api
            mock_api.model_info.side_effect = RepositoryNotFoundError(
                "Not found"
            )

            size = await downloader.get_model_size(
                repo="nonexistent/model",
                revision="main",
                endpoint=None,
                token=None,
            )

            assert size is None
