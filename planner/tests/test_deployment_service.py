"""Tests for deployment service layer."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from src.deployment_service import ModelMapper, InstanceDeployer, InstanceMigrator
from src.available_instance_store import AvailableInstance, AvailableInstanceStore


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
            client_mock = mock_client.return_value.__aenter__.return_value

            # Mock responses for all instances
            mock_info_response = MagicMock()
            mock_info_response.json.return_value = mock_instance_responses["info"]
            mock_info_response.raise_for_status = MagicMock()

            # Mock stop response
            mock_stop_response = MagicMock()
            mock_stop_response.json.return_value = mock_instance_responses["stop"]
            mock_stop_response.raise_for_status = MagicMock()

            # Mock start response
            mock_start_response = MagicMock()
            mock_start_response.json.return_value = mock_instance_responses["start"]
            mock_start_response.raise_for_status = MagicMock()

            # Setup get to return info and stop responses
            client_mock.get = AsyncMock(side_effect=[
                mock_info_response,  # instance-1 info
                mock_info_response,  # instance-2 info
                mock_stop_response,  # instance-2 stop
                mock_info_response,  # instance-3 info
                mock_stop_response,  # instance-3 stop
            ])

            # Setup post to return start response
            client_mock.post = AsyncMock(return_value=mock_start_response)

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

    @pytest.mark.asyncio
    async def test_deploy_model_with_scheduler_url(self, mock_instance_responses):
        """Test deployment with scheduler_url parameter."""
        deployer = InstanceDeployer(timeout=30, scheduler_url="http://scheduler:8100")

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
                target_model="model_1",
                instance_index=0,
                previous_model="model_0"
            )

            assert status.success is True
            assert status.target_model == "model_1"
            assert status.previous_model == "model_0"

            # Should call /model/start with scheduler_url in payload
            client_mock.post.assert_called_once_with(
                "http://test:8080/model/start",
                json={
                    "model_id": "model_1",
                    "parameters": {},
                    "scheduler_url": "http://scheduler:8100"
                }
            )


class TestInstanceMigrator:
    """Tests for InstanceMigrator class."""

    @pytest.mark.asyncio
    async def test_migration_model_success(self, mock_instance_responses, mock_migration_info_responses):
        """Test successful migration from one instance to another."""
        migrator = InstanceMigrator(
            timeout=30,
            scheduler_url="http://scheduler:8100"
        )

        with patch("httpx.AsyncClient") as mock_client:
            client_mock = mock_client.return_value.__aenter__.return_value

            # Mock /info responses for original and target
            mock_original_info = MagicMock()
            mock_original_info.json.return_value = mock_migration_info_responses["original_info"]
            mock_original_info.raise_for_status = MagicMock()

            mock_target_info = MagicMock()
            mock_target_info.json.return_value = mock_migration_info_responses["target_info"]
            mock_target_info.raise_for_status = MagicMock()

            # Mock deregister response
            mock_deregister_response = MagicMock()
            mock_deregister_response.json.return_value = mock_instance_responses["deregister"]
            mock_deregister_response.raise_for_status = MagicMock()

            # Mock register response
            mock_register_response = MagicMock()
            mock_register_response.json.return_value = mock_instance_responses["register"]
            mock_register_response.raise_for_status = MagicMock()

            # Setup get to return info responses
            client_mock.get = AsyncMock(side_effect=[
                mock_original_info,
                mock_target_info
            ])

            # Setup post to return register response
            client_mock.post = AsyncMock(return_value=mock_register_response)

            # Mock the deregister_model method
            with patch.object(migrator, 'deregister_model', new_callable=AsyncMock) as mock_deregister:
                mock_deregister.return_value = mock_deregister_response

                status = await migrator.migration_model(
                    original_endpoint="http://original:8080",
                    target_endpoint="http://target:8080",
                    instance_index=0
                )

                assert status.success is True
                assert status.previous_model == "model_0"
                assert status.target_model == "model_1"
                assert status.endpoint == "http://original:8080"
                assert status.deployment_time > 0

                # Verify deregister was called
                mock_deregister.assert_called_once_with("http://original:8080")

                # Verify register was called
                client_mock.post.assert_called_once()
                call_args = client_mock.post.call_args
                assert "model/register" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_migration_model_skip_same_model(self, mock_migration_info_responses):
        """Test migration is skipped when both instances have the same model."""
        migrator = InstanceMigrator(
            timeout=30,
            scheduler_url="http://scheduler:8100"
        )

        # Make both instances have the same model
        same_model_responses = {
            "original_info": mock_migration_info_responses["original_info"],
            "target_info": {
                "success": True,
                "instance": {
                    "instance_id": "target-instance",
                    "status": "running",
                    "current_model": {
                        "model_id": "model_0",  # Same as original
                        "started_at": "2025-10-31T10:00:00Z",
                        "parameters": {}
                    }
                }
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            client_mock = mock_client.return_value.__aenter__.return_value

            mock_original_info = MagicMock()
            mock_original_info.json.return_value = same_model_responses["original_info"]
            mock_original_info.raise_for_status = MagicMock()

            mock_target_info = MagicMock()
            mock_target_info.json.return_value = same_model_responses["target_info"]
            mock_target_info.raise_for_status = MagicMock()

            client_mock.get = AsyncMock(side_effect=[
                mock_original_info,
                mock_target_info
            ])

            status = await migrator.migration_model(
                original_endpoint="http://original:8080",
                target_endpoint="http://target:8080",
                instance_index=0
            )

            # Should succeed but skip the actual migration
            assert status.success is True
            assert status.previous_model == "model_0"
            assert status.target_model == "model_0"

            # Should only call get for info, no post calls
            assert client_mock.get.call_count == 2
            client_mock.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_migration_instances_parallel(self, mock_instance_responses, mock_migration_info_responses):
        """Test parallel migration to multiple instances."""
        migrator = InstanceMigrator(
            timeout=30,
            scheduler_url="http://scheduler:8100"
        )

        original_endpoints = [
            "http://original-1:8080",
            "http://original-2:8080"
        ]
        target_endpoints = [
            "http://target-1:8080",
            "http://target-2:8080"
        ]

        # Mock migration_model to return successful status
        with patch.object(migrator, 'migration_model', new_callable=AsyncMock) as mock_migration:
            from src.models import MigrationStatus
            mock_migration.return_value = MigrationStatus(
                endpoint="http://original:8080",
                target_model="model_1",
                previous_model="model_0",
                success=True,
                error_message=None,
                deployment_time=1.0
            )

            statuses = await migrator.migration_instances(
                original_endpoints=original_endpoints,
                target_endpoints=target_endpoints
            )

            assert len(statuses) == 2
            assert all(s.success for s in statuses)

            # Verify migration_model was called for each pair
            assert mock_migration.call_count == 2

    @pytest.mark.asyncio
    async def test_deregister_model_success(self, mock_instance_responses):
        """Test successful deregistration from scheduler."""
        migrator = InstanceMigrator(
            timeout=30,
            scheduler_url="http://scheduler:8100"
        )

        with patch("httpx.AsyncClient") as mock_client:
            client_mock = mock_client.return_value.__aenter__.return_value

            mock_deregister_response = MagicMock()
            mock_deregister_response.json.return_value = mock_instance_responses["deregister"]
            mock_deregister_response.raise_for_status = MagicMock()

            client_mock.post = AsyncMock(return_value=mock_deregister_response)

            result = await migrator.deregister_model("http://test:8080")

            assert result["success"] is True
            client_mock.post.assert_called_once_with(
                "http://test:8080/model/deregister"
            )


class TestAvailableInstanceStore:
    """Tests for AvailableInstanceStore class."""

    @pytest.mark.asyncio
    async def test_add_available_instance(self):
        """Test adding an instance to the store."""
        store = AvailableInstanceStore()

        instance = AvailableInstance(
            model_id="model_0",
            endpoint="http://instance-1:8080"
        )

        await store.add_available_instance(instance)

        instances = await store.get_available_instances_by_model_id("model_0")
        assert len(instances) == 1
        assert instances[0].endpoint == "http://instance-1:8080"

    @pytest.mark.asyncio
    async def test_add_multiple_instances_same_model(self):
        """Test adding multiple instances for the same model."""
        store = AvailableInstanceStore()

        instance1 = AvailableInstance(
            model_id="model_0",
            endpoint="http://instance-1:8080"
        )
        instance2 = AvailableInstance(
            model_id="model_0",
            endpoint="http://instance-2:8080"
        )

        await store.add_available_instance(instance1)
        await store.add_available_instance(instance2)

        instances = await store.get_available_instances_by_model_id("model_0")
        assert len(instances) == 2

    @pytest.mark.asyncio
    async def test_fetch_one_available_instance(self):
        """Test fetching one instance from the store."""
        store = AvailableInstanceStore()

        instance1 = AvailableInstance(
            model_id="model_0",
            endpoint="http://instance-1:8080"
        )
        instance2 = AvailableInstance(
            model_id="model_0",
            endpoint="http://instance-2:8080"
        )

        await store.add_available_instance(instance1)
        await store.add_available_instance(instance2)

        # Fetch first instance
        fetched = await store.fetch_one_available_instance("model_0")
        assert fetched is not None
        assert fetched.endpoint == "http://instance-1:8080"

        # Verify only one remains
        remaining = await store.get_available_instances_by_model_id("model_0")
        assert len(remaining) == 1
        assert remaining[0].endpoint == "http://instance-2:8080"

    @pytest.mark.asyncio
    async def test_fetch_one_available_instance_empty(self):
        """Test fetching from empty store returns None."""
        store = AvailableInstanceStore()

        fetched = await store.fetch_one_available_instance("model_0")
        assert fetched is None

    @pytest.mark.asyncio
    async def test_fetch_one_available_instance_unknown_model(self):
        """Test fetching unknown model returns None."""
        store = AvailableInstanceStore()

        instance = AvailableInstance(
            model_id="model_0",
            endpoint="http://instance-1:8080"
        )
        await store.add_available_instance(instance)

        fetched = await store.fetch_one_available_instance("model_unknown")
        assert fetched is None

    @pytest.mark.asyncio
    async def test_remove_available_instance(self):
        """Test removing an instance from the store."""
        store = AvailableInstanceStore()

        instance1 = AvailableInstance(
            model_id="model_0",
            endpoint="http://instance-1:8080"
        )
        instance2 = AvailableInstance(
            model_id="model_0",
            endpoint="http://instance-2:8080"
        )

        await store.add_available_instance(instance1)
        await store.add_available_instance(instance2)

        # Remove the first instance
        await store.remove_available_instance(instance1)

        instances = await store.get_available_instances_by_model_id("model_0")
        assert len(instances) == 1
        assert instances[0].endpoint == "http://instance-2:8080"

    @pytest.mark.asyncio
    async def test_get_all_available_instances(self):
        """Test getting all instances across all models."""
        store = AvailableInstanceStore()

        instance1 = AvailableInstance(
            model_id="model_0",
            endpoint="http://instance-1:8080"
        )
        instance2 = AvailableInstance(
            model_id="model_1",
            endpoint="http://instance-2:8080"
        )
        instance3 = AvailableInstance(
            model_id="model_0",
            endpoint="http://instance-3:8080"
        )

        await store.add_available_instance(instance1)
        await store.add_available_instance(instance2)
        await store.add_available_instance(instance3)

        all_instances = await store.get_available_instances()
        assert len(all_instances) == 3

    @pytest.mark.asyncio
    async def test_remove_instance_from_unknown_model(self):
        """Test removing instance from unknown model doesn't raise error."""
        store = AvailableInstanceStore()

        instance = AvailableInstance(
            model_id="model_unknown",
            endpoint="http://instance-1:8080"
        )

        # Should not raise any error
        await store.remove_available_instance(instance)
