"""
Unit tests for Node class.

Tests cover:
- Initialization
- Scheduler module finding
- Configuration file creation
- Process management
- Instance registration
- Error handling
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Import after adding to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from node import Node, PossibleFanout
from clients.instance_client import InstanceClient


class TestPossibleFanout:
    """Test suite for PossibleFanout model."""

    def test_possible_fanout_init(self):
        """Test PossibleFanout initialization."""
        fanout = PossibleFanout(model_id="gpt-4", min_fanout=1, max_fanout=5)

        assert fanout.model_id == "gpt-4"
        assert fanout.min_fanout == 1
        assert fanout.max_fanout == 5


class TestNodeInitialization:
    """Test suite for Node initialization."""

    def test_node_init_defaults(self):
        """Test Node initialization with defaults."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")

        assert node.model_id == "gpt-4"
        assert node.predictor_url == "http://localhost:8001"
        assert node.scheduler_host == "localhost"
        assert node.scheduler_port == 8100
        assert node._started is False
        assert node.instance_list == []

    def test_node_init_custom_params(self):
        """Test Node initialization with custom parameters."""
        node = Node(
            model_id="llama-2",
            predictor_url="http://predictor:8001",
            scheduler_host="0.0.0.0",
            scheduler_port=9000,
        )

        assert node.model_id == "llama-2"
        assert node.predictor_url == "http://predictor:8001"
        assert node.scheduler_host == "0.0.0.0"
        assert node.scheduler_port == 9000

    def test_node_init_custom_scheduler_path(self):
        """Test Node initialization with custom scheduler path."""
        custom_path = "/custom/scheduler"
        node = Node(
            model_id="gpt-4",
            predictor_url="http://localhost:8001",
            scheduler_module_path=custom_path,
        )

        assert node.scheduler_module_path == Path(custom_path)


class TestNodeSchedulerModuleFinding:
    """Test suite for scheduler module path finding."""

    def test_find_scheduler_module_relative_path(self):
        """Test finding scheduler module via relative path."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")

        # Should find scheduler module relative to this file
        # graph/src/node.py -> scheduler/
        expected_path = Path(__file__).parent.parent.parent.parent / "scheduler"

        # Check if the found path matches expected structure
        assert node.scheduler_module_path.name == "scheduler"

    def test_find_scheduler_module_env_variable(self):
        """Test finding scheduler module via environment variable."""
        custom_path = "/tmp/custom_scheduler"

        # Mock to make relative paths not exist, so env var is used
        def mock_exists(path_instance):
            return str(path_instance) == custom_path

        with patch.dict("os.environ", {"SCHEDULER_MODULE_PATH": custom_path}):
            with patch.object(Path, "exists", mock_exists):
                node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")
                assert str(node.scheduler_module_path) == custom_path


class TestNodeConfigCreation:
    """Test suite for configuration file creation."""

    def test_create_scheduler_config(self):
        """Test scheduler configuration file creation."""
        node = Node(
            model_id="gpt-4",
            predictor_url="http://localhost:8001",
            scheduler_host="localhost",
            scheduler_port=8100,
        )

        node._create_scheduler_config()

        # Check config file created
        assert node._scheduler_config_file is not None
        assert node._scheduler_config_file.exists()

        # Read and verify config
        import tomllib

        with open(node._scheduler_config_file, "rb") as f:
            config = tomllib.load(f)

        assert config["server"]["host"] == "localhost"
        assert config["server"]["port"] == 8100
        assert config["predictor"]["url"] == "http://localhost:8001"
        assert config["predictor"]["timeout"] == 30
        assert config["scheduling"]["strategy"] == "min_time"

        # Cleanup
        node._scheduler_config_file.unlink()

    def test_create_scheduler_config_missing_tomli_w(self):
        """Test error when tomli_w not installed."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")

        with patch.dict("sys.modules", {"tomli_w": None}):
            with pytest.raises(ImportError, match="tomli_w is required"):
                node._create_scheduler_config()


class TestNodeStartStop:
    """Test suite for node start/stop functionality."""

    @pytest.mark.asyncio
    async def test_start_already_started(self):
        """Test starting an already started node."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")
        node._started = True

        with pytest.raises(RuntimeError, match="already started"):
            await node.start()

    @pytest.mark.asyncio
    async def test_start_scheduler_module_not_found(self):
        """Test starting with missing scheduler module."""
        node = Node(
            model_id="gpt-4",
            predictor_url="http://localhost:8001",
            scheduler_module_path="/nonexistent/path",
        )

        with pytest.raises(RuntimeError, match="Scheduler module not found"):
            await node.start()

    @pytest.mark.asyncio
    async def test_stop_not_started(self):
        """Test stopping a node that wasn't started."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")

        # Should not raise error
        await node.stop()

    @pytest.mark.asyncio
    async def test_stop_cleans_up_process(self):
        """Test that stop cleans up scheduler process."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")

        # Mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Running
        mock_process.wait.return_value = None
        node._scheduler_process = mock_process
        node._started = True

        # Create temp config file
        node._scheduler_config_file = Path(tempfile.gettempdir()) / "test_config.toml"
        node._scheduler_config_file.write_text("test")

        await node.stop()

        # Verify cleanup
        mock_process.terminate.assert_called_once()
        assert node._scheduler_process is None
        assert not node._scheduler_config_file.exists()
        assert node._started is False

    @pytest.mark.asyncio
    async def test_stop_force_kill_on_timeout(self):
        """Test force kill when graceful shutdown times out."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")

        # Mock process that doesn't terminate
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = [
            __import__("subprocess").TimeoutExpired("cmd", 10),
            None,
        ]
        node._scheduler_process = mock_process
        node._started = True

        await node.stop()

        # Verify force kill
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()


class TestNodeInstanceRegistration:
    """Test suite for instance registration."""

    @pytest.mark.asyncio
    async def test_register_instance_not_started(self):
        """Test registering instance to non-started node."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")
        instance = InstanceClient(base_url="http://localhost:5001")

        with pytest.raises(RuntimeError, match="not started"):
            await node.register_instance(instance)

    @pytest.mark.asyncio
    async def test_register_instance_success(self):
        """Test successful instance registration."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")
        node._started = True
        node.scheduler = MagicMock()
        node.scheduler.base_url = "http://localhost:8100"

        instance = InstanceClient(base_url="http://localhost:5001")

        with patch.object(instance, "start_model", new_callable=AsyncMock) as mock_start:
            await node.register_instance(instance)

            # Verify start_model was called
            mock_start.assert_called_once_with(
                model_id="gpt-4", scheduler_url="http://localhost:8100"
            )

            # Verify instance added to list
            assert instance in node.instance_list
            assert len(node.instance_list) == 1


class TestNodeExecution:
    """Test suite for task execution."""

    @pytest.mark.asyncio
    async def test_exec_not_started(self):
        """Test execution on non-started node."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")

        with pytest.raises(RuntimeError, match="not started"):
            await node.exec({"prompt": "test"})

    @pytest.mark.asyncio
    async def test_exec_no_instances(self):
        """Test execution without registered instances."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")
        node._started = True

        with pytest.raises(RuntimeError, match="No instances registered"):
            await node.exec({"prompt": "test"})

    @pytest.mark.asyncio
    async def test_exec_success(self):
        """Test successful task execution."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")
        node._started = True
        node.instance_list = [MagicMock()]  # At least one instance

        # Mock scheduler
        mock_scheduler = AsyncMock()
        mock_scheduler.submit_task = AsyncMock(
            return_value={"success": True, "task_id": "task-123"}
        )
        mock_scheduler.get_task_info = AsyncMock(
            return_value={"task": {"status": "completed", "result": {"output": "test"}}}
        )
        mock_scheduler.__aenter__ = AsyncMock(return_value=mock_scheduler)
        mock_scheduler.__aexit__ = AsyncMock(return_value=None)

        node.scheduler = mock_scheduler

        result = await node.exec({"prompt": "test"})

        assert result == {"output": "test"}

    @pytest.mark.asyncio
    async def test_exec_task_failed(self):
        """Test execution when task fails."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")
        node._started = True
        node.instance_list = [MagicMock()]

        # Mock scheduler returning failed task
        mock_scheduler = AsyncMock()
        mock_scheduler.submit_task = AsyncMock(return_value={"success": True})
        mock_scheduler.get_task_info = AsyncMock(
            return_value={
                "task": {"status": "failed", "error": "Model error"}
            }
        )
        mock_scheduler.__aenter__ = AsyncMock(return_value=mock_scheduler)
        mock_scheduler.__aexit__ = AsyncMock(return_value=None)

        node.scheduler = mock_scheduler

        with pytest.raises(RuntimeError, match="Task .* failed: Model error"):
            await node.exec({"prompt": "test"})


class TestNodeStatusCheck:
    """Test suite for node status checking."""

    def test_is_running_not_started(self):
        """Test is_running when node not started."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")

        assert node.is_running() is False

    def test_is_running_no_process(self):
        """Test is_running when no process exists."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")
        node._started = True
        node._scheduler_process = None

        assert node.is_running() is False

    def test_is_running_process_alive(self):
        """Test is_running when process is alive."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")
        node._started = True

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Still running
        node._scheduler_process = mock_process

        assert node.is_running() is True

    def test_is_running_process_dead(self):
        """Test is_running when process has exited."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")
        node._started = True

        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Exited
        node._scheduler_process = mock_process

        assert node.is_running() is False


class TestNodeContextManager:
    """Test suite for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using node as async context manager."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")

        with patch.object(node, "start", new_callable=AsyncMock) as mock_start:
            with patch.object(node, "stop", new_callable=AsyncMock) as mock_stop:
                async with node as n:
                    assert n is node
                    mock_start.assert_called_once()

                mock_stop.assert_called_once()


class TestNodeCoverage:
    """Additional tests to improve coverage."""

    def test_find_scheduler_module_default_path(self):
        """Test finding scheduler module with default fallback path."""
        # Mock all paths to not exist except default
        def mock_exists(path_instance):
            return False

        with patch.object(Path, "exists", mock_exists):
            node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")

            # Should return the default calculated path
            assert node.scheduler_module_path.name == "scheduler"

    @pytest.mark.asyncio
    async def test_wait_for_scheduler_timeout(self):
        """Test _wait_for_scheduler_ready timeout."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")
        node._scheduler_process = MagicMock()
        node._scheduler_process.poll.return_value = None  # Still running

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection failed")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_cls.return_value = mock_client

            with pytest.raises(TimeoutError, match="Scheduler not ready"):
                await node._wait_for_scheduler_ready(
                    "http://localhost:8100", timeout=0.1, check_interval=0.05
                )

    @pytest.mark.asyncio
    async def test_wait_for_scheduler_process_died(self):
        """Test _wait_for_scheduler_ready when process dies."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")

        # Mock process that died
        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Exit code 1
        mock_process.returncode = 1
        mock_stderr = MagicMock()
        mock_stderr.read.return_value.decode.return_value = "Error starting"
        mock_stdout = MagicMock()
        mock_stdout.read.return_value.decode.return_value = "Output"
        mock_process.stderr = mock_stderr
        mock_process.stdout = mock_stdout
        node._scheduler_process = mock_process

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection failed")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_cls.return_value = mock_client

            with pytest.raises(TimeoutError, match="Process exited with code"):
                await node._wait_for_scheduler_ready(
                    "http://localhost:8100", timeout=0.1, check_interval=0.05
                )

    @pytest.mark.asyncio
    async def test_exec_polling_loop(self):
        """Test exec method polling for task completion."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")
        node._started = True
        node.instance_list = [MagicMock()]

        # Mock scheduler with multiple status responses
        mock_scheduler = AsyncMock()
        mock_scheduler.submit_task = AsyncMock(return_value={"success": True})

        # First call: pending, second call: in_progress, third call: completed
        call_count = [0]

        async def mock_get_task_info(task_id):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"task": {"status": "pending"}}
            elif call_count[0] == 2:
                return {"task": {"status": "in_progress"}}
            else:
                return {"task": {"status": "completed", "result": {"output": "done"}}}

        mock_scheduler.get_task_info = mock_get_task_info
        mock_scheduler.__aenter__ = AsyncMock(return_value=mock_scheduler)
        mock_scheduler.__aexit__ = AsyncMock(return_value=None)

        node.scheduler = mock_scheduler

        result = await node.exec({"prompt": "test"})

        assert result == {"output": "done"}
        assert call_count[0] >= 3

    def test_destructor_cleanup(self):
        """Test that destructor cleans up process."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Running
        mock_process.wait.return_value = None
        node._scheduler_process = mock_process

        # Trigger destructor
        node.__del__()

        mock_process.terminate.assert_called_once()

    def test_destructor_process_already_stopped(self):
        """Test destructor when process already stopped."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")

        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Already stopped
        node._scheduler_process = mock_process

        # Should not crash
        node.__del__()

    def test_destructor_no_process(self):
        """Test destructor when no process exists."""
        node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")
        node._scheduler_process = None

        # Should not crash
        node.__del__()
