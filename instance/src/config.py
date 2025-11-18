"""
Configuration management for Instance Service
"""

import os
import uuid
from pathlib import Path
from typing import Optional


class Config:
    """Instance service configuration"""

    def __init__(self):
        # Instance configuration
        self.instance_id: str = os.getenv("INSTANCE_ID", str(uuid.uuid4()))
        self.instance_port: int = int(os.getenv("INSTANCE_PORT", "8000"))

        # Model container port (instance_port + 1000)
        self.model_port: int = self.instance_port + 1000

        # Paths
        self.base_dir: Path = Path(__file__).parent.parent
        self.dockers_dir: Path = self.base_dir / "dockers"
        self.registry_path: Path = self.dockers_dir / "model_registry.yaml"

        # Docker configuration
        self.docker_network: str = os.getenv("DOCKER_NETWORK", "instance_network")
        self.container_name_prefix: str = f"model_{self.instance_id}"

        # Logging
        self.log_level: str = os.getenv("INSTANCE_LOG_LEVEL", "INFO")
        self.log_dir: str = os.getenv("INSTANCE_LOG_DIR", "logs")
        self.enable_json_logs: bool = (
            os.getenv("INSTANCE_ENABLE_JSON_LOGS", "false").lower() == "true"
        )

        # Task queue
        self.max_queue_size: int = int(os.getenv("MAX_QUEUE_SIZE", "100"))

        # Manager configuration
        self.use_docker: bool = (
            os.getenv("INSTANCE_USE_DOCKER", "false").lower() == "true"
        )

        # Health check
        self.health_check_interval: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "10"))
        self.health_check_timeout: int = int(os.getenv("HEALTH_CHECK_TIMEOUT", "30"))

        # Platform information overrides
        # These allow users to specify platform info instead of auto-detection
        self.platform_software_name: Optional[str] = os.getenv(
            "INSTANCE_PLATFORM_SOFTWARE_NAME"
        )
        self.platform_software_version: Optional[str] = os.getenv(
            "INSTANCE_PLATFORM_SOFTWARE_VERSION"
        )
        self.platform_hardware_name: Optional[str] = os.getenv(
            "INSTANCE_PLATFORM_HARDWARE_NAME"
        )

    def get_model_directory(self, directory_name: str) -> Path:
        """Get full path to model directory"""
        return self.dockers_dir / directory_name

    def get_model_container_name(self, model_id: str) -> str:
        """Generate container name for a model"""
        # Replace special characters with underscores
        safe_model_id = model_id.replace("/", "_").replace(":", "_")
        return f"{self.container_name_prefix}_{safe_model_id}"


# Global config instance
config = Config()
