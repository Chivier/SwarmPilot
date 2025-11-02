"""Configuration management for predictor service.

Supports configuration from multiple sources with priority:
1. CLI arguments (highest priority)
2. Environment variables (PREDICTOR_* prefix)
3. Configuration file (predictor.toml)
4. Default values (lowest priority)
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PredictorConfig(BaseSettings):
    """Configuration for the predictor service."""

    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind the server to"
    )
    port: int = Field(
        default=8000,
        description="Port to bind the server to"
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development"
    )
    workers: int = Field(
        default=1,
        description="Number of worker processes (production)"
    )

    # Storage settings
    storage_dir: str = Field(
        default="models",
        description="Directory to store trained models"
    )

    # Logging settings
    log_level: str = Field(
        default="info",
        description="Logging level (debug, info, warning, error, critical)"
    )

    # Application metadata
    app_name: str = Field(
        default="Runtime Predictor Service",
        description="Application name"
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version"
    )

    model_config = SettingsConfigDict(
        env_prefix="PREDICTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def from_toml(cls, config_path: Optional[Path] = None) -> "PredictorConfig":
        """Load configuration from TOML file.

        Args:
            config_path: Path to configuration file. If None, searches for
                        predictor.toml in current directory and parent directories.

        Returns:
            PredictorConfig instance with values from TOML file and environment.
        """
        if config_path is None:
            # Search for predictor.toml in current and parent directories
            current = Path.cwd()
            for directory in [current, *current.parents]:
                candidate = directory / "predictor.toml"
                if candidate.exists():
                    config_path = candidate
                    break

        if config_path and config_path.exists():
            # Python 3.11+ has tomllib built-in, otherwise use tomli
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    # If neither available, skip config file loading
                    return cls()

            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)

            # Extract predictor section if it exists
            predictor_config = config_data.get("predictor", config_data)
            return cls(**predictor_config)

        # No config file found, use defaults and environment variables
        return cls()

    def get_storage_path(self) -> Path:
        """Get the storage directory as a Path object.

        Returns:
            Path to the storage directory.
        """
        return Path(self.storage_dir)

    def ensure_storage_dir(self) -> Path:
        """Ensure storage directory exists and return its path.

        Returns:
            Path to the storage directory.
        """
        storage_path = self.get_storage_path()
        storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "host": self.host,
            "port": self.port,
            "reload": self.reload,
            "workers": self.workers,
            "storage_dir": self.storage_dir,
            "log_level": self.log_level,
            "app_name": self.app_name,
            "app_version": self.app_version,
        }


# Global configuration instance
_config: Optional[PredictorConfig] = None


def get_config() -> PredictorConfig:
    """Get the global configuration instance.

    Returns:
        The global PredictorConfig instance.
    """
    global _config
    if _config is None:
        _config = PredictorConfig()
    return _config


def set_config(config: PredictorConfig) -> None:
    """Set the global configuration instance.

    Args:
        config: PredictorConfig instance to set as global.
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to None."""
    global _config
    _config = None
