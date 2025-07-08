"""
Configuration management system for the autonomous agent framework.
"""

from collections.abc import Callable
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .exceptions import ConfigError, ConfigNotFoundError, ConfigValidationError


class ConfigSchema:
    """
    Configuration schema definition and validation.

    Defines the expected structure and types for configuration data.
    """

    def __init__(self) -> None:
        """Initialize the configuration schema."""
        self.agent_manager = {
            "max_agents": int,
            "heartbeat_interval": float,
            "communication_timeout": float,
            "retry_attempts": int
        }

        self.logging = {
            "level": str,
            "format": str,
            "handlers": dict
        }

        self.communication = {
            "message_broker": {
                "queue_size": int,
                "timeout": float
            }
        }

        self.agents = dict

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary representation."""
        return {
            "agent_manager": self.agent_manager,
            "logging": self.logging,
            "communication": self.communication,
            "agents": self.agents
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigSchema":
        """Create schema from dictionary."""
        schema = cls()

        if "agent_manager" in data:
            schema.agent_manager = data["agent_manager"]
        if "logging" in data:
            schema.logging = data["logging"]
        if "communication" in data:
            schema.communication = data["communication"]
        if "agents" in data:
            schema.agents = data["agents"]

        return schema


class ConfigValidator:
    """
    Validator for configuration data.

    Provides comprehensive validation of configuration structure,
    types, and values according to the defined schema.
    """

    def __init__(self, schema: Optional[ConfigSchema] = None):
        """
        Initialize the validator.

        Args:
            schema: Optional custom schema (uses default if not provided)
        """
        self.schema = schema or ConfigSchema()
        self.required_keys = ["agent_manager", "logging", "communication", "agents"]
        self.valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def validate(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration data.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ConfigValidationError: If validation fails
        """
        # Check required top-level keys
        self._validate_required_keys(config)

        # Validate each section
        self._validate_agent_manager(config.get("agent_manager", {}))
        self._validate_logging(config.get("logging", {}))
        self._validate_communication(config.get("communication", {}))
        self._validate_agents(config.get("agents", {}))

    def _validate_required_keys(self, config: Dict[str, Any]) -> None:
        """Validate that all required top-level keys are present."""
        missing_keys = [key for key in self.required_keys if key not in config]

        if missing_keys:
            raise ConfigValidationError(
                f"Missing required configuration sections: {missing_keys}",
                context={"missing_keys": missing_keys, "available_keys": list(config.keys())}
            )

    def _validate_agent_manager(self, config: Dict[str, Any]) -> None:
        """Validate agent manager configuration."""
        if "max_agents" in config:
            if not isinstance(config["max_agents"], int) or config["max_agents"] <= 0:
                raise ConfigValidationError("max_agents must be a positive integer")

        if "heartbeat_interval" in config:
            if not isinstance(config["heartbeat_interval"], (int, float)) or config["heartbeat_interval"] <= 0:
                raise ConfigValidationError("heartbeat_interval must be a positive number")

        if "communication_timeout" in config:
            if not isinstance(config["communication_timeout"], (int, float)) or config["communication_timeout"] <= 0:
                raise ConfigValidationError("communication_timeout must be a positive number")

        if "retry_attempts" in config:
            if not isinstance(config["retry_attempts"], int) or config["retry_attempts"] < 0:
                raise ConfigValidationError("retry_attempts must be a non-negative integer")

    def _validate_logging(self, config: Dict[str, Any]) -> None:
        """Validate logging configuration."""
        if "level" in config:
            if not isinstance(config["level"], str) or config["level"] not in self.valid_log_levels:
                raise ConfigValidationError(
                    f"logging level must be one of: {self.valid_log_levels}",
                    context={"provided_level": config.get("level"), "valid_levels": self.valid_log_levels}
                )

        if "format" in config and not isinstance(config["format"], str):
            raise ConfigValidationError("logging format must be a string")

        if "handlers" in config and not isinstance(config["handlers"], dict):
            raise ConfigValidationError("logging handlers must be a dictionary")

    def _validate_communication(self, config: Dict[str, Any]) -> None:
        """Validate communication configuration."""
        if "message_broker" in config:
            broker_config = config["message_broker"]

            if not isinstance(broker_config, dict):
                raise ConfigValidationError("message_broker must be a dictionary")

            if "queue_size" in broker_config:
                if not isinstance(broker_config["queue_size"], int) or broker_config["queue_size"] <= 0:
                    raise ConfigValidationError("message_broker queue_size must be a positive integer")

            if "timeout" in broker_config:
                if not isinstance(broker_config["timeout"], (int, float)) or broker_config["timeout"] <= 0:
                    raise ConfigValidationError("message_broker timeout must be a positive number")

    def _validate_agents(self, config: Dict[str, Any]) -> None:
        """Validate agents configuration."""
        if not isinstance(config, dict):
            raise ConfigValidationError("agents configuration must be a dictionary")

        for agent_id, agent_config in config.items():
            self._validate_agent_config(agent_id, agent_config)

    def _validate_agent_config(self, agent_id: str, config: Dict[str, Any]) -> None:
        """Validate individual agent configuration."""
        if not isinstance(config, dict):
            raise ConfigValidationError(f"Agent {agent_id} configuration must be a dictionary")

        required_agent_fields = ["agent_type", "enabled"]
        for field in required_agent_fields:
            if field not in config:
                raise ConfigValidationError(
                    f"Agent {agent_id} missing required field: {field}",
                    context={"agent_id": agent_id, "missing_field": field}
                )

        if not isinstance(config["agent_type"], str):
            raise ConfigValidationError(f"Agent {agent_id} agent_type must be a string")

        if not isinstance(config["enabled"], bool):
            raise ConfigValidationError(f"Agent {agent_id} enabled must be a boolean")

        if "priority" in config and not isinstance(config["priority"], int):
            raise ConfigValidationError(f"Agent {agent_id} priority must be an integer")

        if "config" in config and not isinstance(config["config"], dict):
            raise ConfigValidationError(f"Agent {agent_id} config must be a dictionary")


class ConfigLoader:
    """
    Configuration loader supporting multiple formats.

    Provides loading from files, dictionaries, and environment variables
    with support for JSON and YAML formats.
    """

    def __init__(self) -> None:
        """Initialize the configuration loader."""
        self.supported_formats = ['.json', '.yaml', '.yml']

    def load_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a file.

        Args:
            file_path: Path to the configuration file

        Returns:
            Configuration dictionary

        Raises:
            ConfigNotFoundError: If file is not found
            ConfigError: If file format is unsupported or parsing fails
        """
        path = Path(file_path)

        if not path.exists():
            raise ConfigNotFoundError(
                f"Configuration file not found: {file_path}",
                context={"file_path": str(file_path)}
            )

        if path.suffix not in self.supported_formats:
            raise ConfigError(
                f"Unsupported file format: {path.suffix}. Supported formats: {self.supported_formats}",
                context={"file_path": str(file_path), "suffix": path.suffix}
            )

        try:
            with open(path, encoding='utf-8') as f:
                if path.suffix == '.json':
                    return json.load(f)  # type: ignore[return-value]
                elif path.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}  # type: ignore[return-value]
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ConfigError(f"Failed to parse configuration file: {e}", cause=e)
        except Exception as e:
            raise ConfigError(f"Failed to read configuration file: {e}", cause=e)

        # This should never be reached, but adding for type safety
        return {}

    def load_from_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load configuration from a dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Configuration dictionary (deep copy)
        """
        return deepcopy(config_dict)

    def load_from_environment(self, prefix: str = "AGENT_") -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        Args:
            prefix: Prefix for environment variable names

        Returns:
            Configuration dictionary built from environment variables
        """
        config = {
            "agent_manager": {},
            "logging": {},
            "communication": {"message_broker": {}},
            "agents": {}
        }

        env_mapping = {
            f"{prefix}MANAGER_MAX_AGENTS": ("agent_manager", "max_agents", int),
            f"{prefix}MANAGER_HEARTBEAT_INTERVAL": ("agent_manager", "heartbeat_interval", float),
            f"{prefix}LOGGING_LEVEL": ("logging", "level", str),
            f"{prefix}LOGGING_FORMAT": ("logging", "format", str),
            f"{prefix}COMMUNICATION_MESSAGE_BROKER_QUEUE_SIZE": ("communication", "message_broker", "queue_size", int),
            f"{prefix}COMMUNICATION_MESSAGE_BROKER_TIMEOUT": ("communication", "message_broker", "timeout", float),
        }

        for env_var, path_info in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert value to appropriate type
                    if len(path_info) == 3:
                        section, key, value_type = path_info
                        config[section][key] = value_type(value)
                    elif len(path_info) == 4:
                        section, subsection, key, value_type = path_info
                        config[section][subsection][key] = value_type(value)
                except (ValueError, TypeError) as e:
                    raise ConfigError(f"Invalid value for {env_var}: {value}", cause=e)

        return config

    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.

        Args:
            base_config: Base configuration
            override_config: Configuration to merge (takes precedence)

        Returns:
            Merged configuration dictionary
        """
        merged = deepcopy(base_config)
        self._deep_merge(merged, override_config)
        return merged

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge two dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = deepcopy(value)


class ConfigManager:
    """
    Centralized configuration management system.

    Provides comprehensive configuration loading, validation, access,
    and change management for the autonomous agent system.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the configuration manager.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.loader = ConfigLoader()
        self.validator = ConfigValidator()
        self.config: Optional[Dict[str, Any]] = None
        self.config_file_path: Optional[Path] = None
        self.change_callbacks: Dict[str, List[Callable]] = {}

    def load_config(self, source: Union[str, Path, Dict[str, Any]]) -> None:
        """
        Load configuration from various sources.

        Args:
            source: Configuration source (file path or dictionary)

        Raises:
            ConfigError: If loading or validation fails
        """
        try:
            if isinstance(source, (str, Path)):
                self.config_file_path = Path(source)
                self.config = self.loader.load_from_file(source)
                self.logger.info(f"Loaded configuration from file: {source}")
            elif isinstance(source, dict):
                self.config = self.loader.load_from_dict(source)
                self.logger.info("Loaded configuration from dictionary")
            else:
                raise ConfigError(f"Unsupported configuration source type: {type(source)}")

            # Validate the loaded configuration
            self.validator.validate(self.config)
            self.logger.info("Configuration validation passed")

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def reload_config(self) -> None:
        """
        Reload configuration from the original file.

        Raises:
            ConfigError: If no file path is set or reload fails
        """
        if self.config_file_path is None:
            raise ConfigError("No configuration file path set for reload")

        self.load_config(self.config_file_path)
        self.logger.info("Configuration reloaded successfully")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "agent_manager.max_agents")
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        if self.config is None:
            return default

        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "agent_manager.max_agents")
            value: Value to set
        """
        if self.config is None:
            self.config = {}

        keys = key.split('.')
        current = self.config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Store old value for callbacks
        old_value = current.get(keys[-1])

        # Set the value
        current[keys[-1]] = value

        # Trigger change callbacks
        self._trigger_change_callbacks(key, old_value, value)

        self.logger.debug(f"Set configuration {key} = {value}")

    def get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Agent configuration dictionary

        Raises:
            ConfigNotFoundError: If agent configuration is not found
        """
        agent_config = self.get(f"agents.{agent_id}")

        if agent_config is None:
            raise ConfigNotFoundError(
                f"Configuration for agent {agent_id} not found",
                context={"agent_id": agent_id}
            )

        return agent_config

    def list_agents(self) -> List[str]:
        """
        List all configured agent IDs.

        Returns:
            List of agent IDs
        """
        agents_config = self.get("agents", {})
        return list(agents_config.keys())

    def list_enabled_agents(self) -> List[str]:
        """
        List all enabled agent IDs.

        Returns:
            List of enabled agent IDs
        """
        enabled_agents = []
        agents_config = self.get("agents", {})

        for agent_id, config in agents_config.items():
            if config.get("enabled", False):
                enabled_agents.append(agent_id)

        return enabled_agents

    def save_config(self, file_path: Union[str, Path]) -> None:
        """
        Save current configuration to a file.

        Args:
            file_path: Path to save the configuration

        Raises:
            ConfigError: If saving fails
        """
        if self.config is None:
            raise ConfigError("No configuration loaded to save")

        path = Path(file_path)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                if path.suffix == '.json':
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                elif path.suffix in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                else:
                    # Default to JSON
                    json.dump(self.config, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Configuration saved to: {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise ConfigError(f"Failed to save configuration to {file_path}", cause=e)

    def validate_config(self) -> None:
        """
        Validate the current configuration.

        Raises:
            ConfigValidationError: If validation fails
        """
        if self.config is None:
            raise ConfigError("No configuration loaded to validate")

        self.validator.validate(self.config)

    def register_change_callback(self, key: str, callback: Callable[[str, Any, Any], None]) -> None:
        """
        Register a callback for configuration changes.

        Args:
            key: Configuration key to monitor
            callback: Callback function (key, old_value, new_value)
        """
        if key not in self.change_callbacks:
            self.change_callbacks[key] = []

        self.change_callbacks[key].append(callback)
        self.logger.debug(f"Registered change callback for key: {key}")

    def unregister_change_callback(self, key: str, callback: Callable) -> None:
        """
        Unregister a change callback.

        Args:
            key: Configuration key
            callback: Callback function to remove
        """
        if key in self.change_callbacks:
            try:
                self.change_callbacks[key].remove(callback)
                if not self.change_callbacks[key]:
                    del self.change_callbacks[key]
                self.logger.debug(f"Unregistered change callback for key: {key}")
            except ValueError:
                pass  # Callback not found

    def apply_environment_overrides(self, prefix: str = "AGENT_") -> None:
        """
        Apply environment variable overrides to current configuration.

        Args:
            prefix: Prefix for environment variable names
        """
        env_config = self.loader.load_from_environment(prefix)

        if self.config is None:
            self.config = env_config
        else:
            self.config = self.loader.merge_configs(self.config, env_config)

        self.logger.info("Applied environment variable overrides")

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.

        Returns:
            Configuration summary dictionary
        """
        if self.config is None:
            return {"status": "No configuration loaded"}

        summary = {
            "status": "Configuration loaded",
            "agent_count": len(self.get("agents", {})),
            "enabled_agents": len(self.list_enabled_agents()),
            "max_agents": self.get("agent_manager.max_agents"),
            "log_level": self.get("logging.level"),
            "config_file": str(self.config_file_path) if self.config_file_path else None
        }

        return summary

    def _trigger_change_callbacks(self, key: str, old_value: Any, new_value: Any) -> None:
        """Trigger change callbacks for a configuration key."""
        callbacks = self.change_callbacks.get(key, [])

        for callback in callbacks:
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                self.logger.error(f"Error in change callback for {key}: {e}")
