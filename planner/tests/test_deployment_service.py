"""Tests for deployment service layer."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from src.deployment_service import ModelMapper, InstanceDeployer


class TestModelMapper:
    """Tests for ModelMapper class."""

    def test_create_mapping_unique_models(self):
        """Test mapping creation with unique model names."""
        model_names = ["model_a", "model_b", "model_c"]
        mapping = ModelMapper.create_mapping(model_names)

        assert mapping == {
            "model_a": 0,
            "model_b": 1,
            "model_c": 2
        }

    def test_create_mapping_with_duplicates(self):
        """Test mapping creation with duplicate model names."""
        model_names = ["model_a", "model_b", "model_a", "model_c", "model_b"]
        mapping = ModelMapper.create_mapping(model_names)

        assert mapping == {
            "model_a": 0,
            "model_b": 1,
            "model_c": 2
        }

    def test_map_names_to_ids(self):
        """Test mapping model names to IDs."""
        mapping = {"model_0": 0, "model_1": 1, "model_2": 2}
        names = ["model_0", "model_1", "model_2", "model_1"]
        ids = ModelMapper.map_names_to_ids(names, mapping)

        assert ids == [0, 1, 2, 1]

    def test_map_names_to_ids_unknown_name(self):
        """Test mapping fails for unknown model name."""
        mapping = {"model_0": 0, "model_1": 1}
        names = ["model_0", "model_unknown"]

        with pytest.raises(ValueError) as exc_info:
            ModelMapper.map_names_to_ids(names, mapping)
        assert "model_unknown" in str(exc_info.value)

    def test_map_ids_to_names(self):
        """Test mapping model IDs to names."""
        reverse_mapping = {0: "model_a", 1: "model_b", 2: "model_c"}
        ids = [0, 1, 2, 1]
        names = ModelMapper.map_ids_to_names(ids, reverse_mapping)

        assert names == ["model_a", "model_b", "model_c", "model_b"]

    def test_map_ids_to_names_unknown_id(self):
        """Test mapping fails for unknown model ID."""
        reverse_mapping = {0: "model_a", 1: "model_b"}
        ids = [0, 5]

        with pytest.raises(ValueError) as exc_info:
            ModelMapper.map_ids_to_names(ids, reverse_mapping)
        assert "Model ID 5" in str(exc_info.value)


class TestInstanceDeployer:
    """Tests for InstanceDeployer class."""

    @pytest.mark.asyncio
    async def test_get_instance_info_success(self, mock_instance_responses):
        """Test successful instance info retrieval."""
        deployer = InstanceDeployer(timeout=30)

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_instance_responses["info"]
            mock_response.raise_for_status = MagicMock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            info = await deployer.get_instance_info("http://test:8080")

            assert info is not None
            assert info["instance_id"] == "test-instance"
            assert info["current_model"]["model_id"] == "model_0"
            mock_get.assert_called_once_with("http://test:8080/info")

    @pytest.mark.asyncio
    async def test_get_instance_info_failure(self):
        """Test instance info retrieval failure."""
        deployer = InstanceDeployer(timeout=30)

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(side_effect=httpx.RequestError("Connection failed"))
            mock_client.return_value.__aenter__.return_value.get = mock_get

            info = await deployer.get_instance_info("http://test:8080")

            assert info is None

    @pytest.mark.asyncio
    async def test_deploy_model_same_as_current(self, mock_instance_responses):
        """Test deployment when target model is already running."""
        deployer = InstanceDeployer(timeout=30)

        with patch("httpx.AsyncClient") as mock_client:
            # Mock /info response
            mock_info_response = MagicMock()
            mock_info_response.json.return_value = mock_instance_responses["info"]
            mock_info_response.raise_for_status = MagicMock()

            mock_get = AsyncMock(return_value=mock_info_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            status = await deployer.deploy_model(
                endpoint="http://test:8080",
                target_model="model_0",  # Same as current
                instance_index=0,
                previous_model="model_0"
            )

            # Should skip deployment
            assert status.success is True
            assert status.target_model == "model_0"
            assert status.previous_model == "model_0"
            # Only GET /info should be called, not stop or start
            assert mock_get.call_count == 1

    @pytest.mark.asyncio
    async def test_deploy_model_switch_required(self, mock_instance_responses):
        """Test deployment when model switch is required."""
        deployer = InstanceDeployer(timeout=30)

        with patch("httpx.AsyncClient") as mock_client:
            client_mock = mock_client.return_value.__aenter__.return_value

            # Mock /info response
            mock_info_response = MagicMock()
            mock_info_response.json.return_value = mock_instance_responses["info"]
            mock_info_response.raise_for_status = MagicMock()

            # Mock /model/stop response
            mock_stop_response = MagicMock()
            mock_stop_response.json.return_value = mock_instance_responses["stop"]
            mock_stop_response.raise_for_status = MagicMock()

            # Mock /model/start response
            mock_start_response = MagicMock()
            mock_start_response.json.return_value = mock_instance_responses["start"]
            mock_start_response.raise_for_status = MagicMock()

            client_mock.get = AsyncMock(side_effect=[mock_info_response, mock_stop_response])
            client_mock.post = AsyncMock(return_value=mock_start_response)

            status = await deployer.deploy_model(
                endpoint="http://test:8080",
                target_model="model_1",  # Different from current model_0
                instance_index=0,
                previous_model="model_0"
            )

            assert status.success is True
            assert status.target_model == "model_1"
            assert status.previous_model == "model_0"

            # Should call /info, /model/stop, /model/start
            assert client_mock.get.call_count == 2
            client_mock.post.assert_called_once_with(
                "http://test:8080/model/start",
                json={"model_id": "model_1", "parameters": {}}
            )

    @pytest.mark.asyncio
    async def test_deploy_model_http_error(self):
        """Test deployment failure due to HTTP error."""
        deployer = InstanceDeployer(timeout=30)

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"

            mock_get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Server error",
                    request=MagicMock(),
                    response=mock_response
                )
            )
            mock_client.return_value.__aenter__.return_value.get = mock_get

            status = await deployer.deploy_model(
                endpoint="http://test:8080",
                target_model="model_1",
                instance_index=0,
                previous_model="model_0"
            )

            assert status.success is False
            assert "HTTP 500" in status.error_message
            assert status.target_model == "model_1"

    @pytest.mark.asyncio
    async def test_deploy_to_instances_parallel(self, mock_instance_responses):
        """Test parallel deployment to multiple instances."""
        deployer = InstanceDeployer(timeout=30)

        endpoints = [
            "http://instance-1:8080",
            "http://instance-2:8080",
            "http://instance-3:8080"
        ]
        target_models = ["model_0", "model_1", "model_2"]
        previous_models = ["model_0", "model_0", "model_0"]

        with patch("httpx.AsyncClient") as mock_client:
            # Mock responses for all instances
            mock_info_response = MagicMock()
            mock_info_response.json.return_value = mock_instance_responses["info"]
            mock_info_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_info_response
            )

            statuses = await deployer.deploy_to_instances(
                endpoints=endpoints,
                target_models=target_models,
                previous_models=previous_models
            )

            assert len(statuses) == 3
            # First instance should skip (same model)
            assert statuses[0].success is True
            assert statuses[0].instance_index == 0
            assert statuses[1].instance_index == 1
            assert statuses[2].instance_index == 2
