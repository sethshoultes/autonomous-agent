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

        self.ollama = {
            "host": str,
            "port": int,
            "timeout": float,
            "default_model": str,
            "max_context_length": int,
            "stream_enabled": bool,
            "retry_attempts": int,
            "retry_delay": float,
            "temperature": float,
            "top_p": float,
            "top_k": int,
            "models": dict
        }

        self.gmail = {
            "credentials_path": str,
            "scopes": list,
            "user_email": str,
            "batch_size": int,
            "rate_limit_per_minute": int,
            "max_retries": int,
            "retry_delay": float,
            "classification": {
                "enabled": bool,
                "spam_threshold": float,
                "importance_threshold": float,
                "categories": list,
                "keywords": dict
            },
            "auto_response": {
                "enabled": bool,
                "response_delay": int,
                "max_responses_per_day": int,
                "templates": dict,
                "trigger_patterns": dict
            },
            "archiving": {
                "enabled": bool,
                "archive_after_days": int,
                "auto_label": bool,
                "label_rules": list,
                "smart_folders": dict
            }
        }

        self.agents = dict

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary representation."""
        return {
            "agent_manager": self.agent_manager,
            "logging": self.logging,
            "communication": self.communication,
            "ollama": self.ollama,
            "gmail": self.gmail,
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
        if "ollama" in data:
            schema.ollama = data["ollama"]
        if "gmail" in data:
            schema.gmail = data["gmail"]
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
        self.optional_keys = ["ollama", "gmail"]
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
        
        # Validate optional sections
        if "ollama" in config:
            self._validate_ollama(config.get("ollama", {}))
        
        if "gmail" in config:
            self._validate_gmail(config.get("gmail", {}))
        
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

    def _validate_ollama(self, config: Dict[str, Any]) -> None:
        """Validate Ollama configuration."""
        if not isinstance(config, dict):
            raise ConfigValidationError("ollama configuration must be a dictionary")

        # Validate host
        if "host" in config and not isinstance(config["host"], str):
            raise ConfigValidationError("ollama host must be a string")

        # Validate port
        if "port" in config:
            if not isinstance(config["port"], int) or not (1 <= config["port"] <= 65535):
                raise ConfigValidationError("ollama port must be an integer between 1 and 65535")

        # Validate timeout
        if "timeout" in config:
            if not isinstance(config["timeout"], (int, float)) or config["timeout"] <= 0:
                raise ConfigValidationError("ollama timeout must be a positive number")

        # Validate default_model
        if "default_model" in config and not isinstance(config["default_model"], str):
            raise ConfigValidationError("ollama default_model must be a string")

        # Validate max_context_length
        if "max_context_length" in config:
            if not isinstance(config["max_context_length"], int) or config["max_context_length"] <= 0:
                raise ConfigValidationError("ollama max_context_length must be a positive integer")

        # Validate stream_enabled
        if "stream_enabled" in config and not isinstance(config["stream_enabled"], bool):
            raise ConfigValidationError("ollama stream_enabled must be a boolean")

        # Validate retry_attempts
        if "retry_attempts" in config:
            if not isinstance(config["retry_attempts"], int) or config["retry_attempts"] < 0:
                raise ConfigValidationError("ollama retry_attempts must be a non-negative integer")

        # Validate retry_delay
        if "retry_delay" in config:
            if not isinstance(config["retry_delay"], (int, float)) or config["retry_delay"] < 0:
                raise ConfigValidationError("ollama retry_delay must be a non-negative number")

        # Validate temperature
        if "temperature" in config:
            if not isinstance(config["temperature"], (int, float)) or not (0 <= config["temperature"] <= 2):
                raise ConfigValidationError("ollama temperature must be a number between 0 and 2")

        # Validate top_p
        if "top_p" in config:
            if not isinstance(config["top_p"], (int, float)) or not (0 <= config["top_p"] <= 1):
                raise ConfigValidationError("ollama top_p must be a number between 0 and 1")

        # Validate top_k
        if "top_k" in config:
            if not isinstance(config["top_k"], int) or config["top_k"] <= 0:
                raise ConfigValidationError("ollama top_k must be a positive integer")

        # Validate models
        if "models" in config and not isinstance(config["models"], dict):
            raise ConfigValidationError("ollama models must be a dictionary")

    def _validate_gmail(self, config: Dict[str, Any]) -> None:
        """Validate Gmail configuration."""
        if not isinstance(config, dict):
            raise ConfigValidationError("gmail configuration must be a dictionary")

        # Validate credentials_path
        if "credentials_path" in config and not isinstance(config["credentials_path"], str):
            raise ConfigValidationError("gmail credentials_path must be a string")

        # Validate scopes
        if "scopes" in config:
            if not isinstance(config["scopes"], list):
                raise ConfigValidationError("gmail scopes must be a list")
            for scope in config["scopes"]:
                if not isinstance(scope, str):
                    raise ConfigValidationError("gmail scopes must contain only strings")

        # Validate user_email
        if "user_email" in config and not isinstance(config["user_email"], str):
            raise ConfigValidationError("gmail user_email must be a string")

        # Validate batch_size
        if "batch_size" in config:
            if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
                raise ConfigValidationError("gmail batch_size must be a positive integer")

        # Validate rate_limit_per_minute
        if "rate_limit_per_minute" in config:
            if not isinstance(config["rate_limit_per_minute"], int) or config["rate_limit_per_minute"] <= 0:
                raise ConfigValidationError("gmail rate_limit_per_minute must be a positive integer")

        # Validate max_retries
        if "max_retries" in config:
            if not isinstance(config["max_retries"], int) or config["max_retries"] < 0:
                raise ConfigValidationError("gmail max_retries must be a non-negative integer")

        # Validate retry_delay
        if "retry_delay" in config:
            if not isinstance(config["retry_delay"], (int, float)) or config["retry_delay"] < 0:
                raise ConfigValidationError("gmail retry_delay must be a non-negative number")

        # Validate classification section
        if "classification" in config:
            self._validate_gmail_classification(config["classification"])

        # Validate auto_response section
        if "auto_response" in config:
            self._validate_gmail_auto_response(config["auto_response"])

        # Validate archiving section
        if "archiving" in config:
            self._validate_gmail_archiving(config["archiving"])

    def _validate_gmail_classification(self, config: Dict[str, Any]) -> None:
        """Validate Gmail classification configuration."""
        if not isinstance(config, dict):
            raise ConfigValidationError("gmail classification configuration must be a dictionary")

        # Validate enabled
        if "enabled" in config and not isinstance(config["enabled"], bool):
            raise ConfigValidationError("gmail classification enabled must be a boolean")

        # Validate spam_threshold
        if "spam_threshold" in config:
            if not isinstance(config["spam_threshold"], (int, float)) or not (0 <= config["spam_threshold"] <= 1):
                raise ConfigValidationError("gmail classification spam_threshold must be a number between 0 and 1")

        # Validate importance_threshold
        if "importance_threshold" in config:
            if not isinstance(config["importance_threshold"], (int, float)) or not (0 <= config["importance_threshold"] <= 1):
                raise ConfigValidationError("gmail classification importance_threshold must be a number between 0 and 1")

        # Validate categories
        if "categories" in config:
            if not isinstance(config["categories"], list):
                raise ConfigValidationError("gmail classification categories must be a list")
            for category in config["categories"]:
                if not isinstance(category, str):
                    raise ConfigValidationError("gmail classification categories must contain only strings")

        # Validate keywords
        if "keywords" in config:
            if not isinstance(config["keywords"], dict):
                raise ConfigValidationError("gmail classification keywords must be a dictionary")
            for category, keywords in config["keywords"].items():
                if not isinstance(keywords, list):
                    raise ConfigValidationError(f"gmail classification keywords for {category} must be a list")
                for keyword in keywords:
                    if not isinstance(keyword, str):
                        raise ConfigValidationError(f"gmail classification keywords for {category} must contain only strings")

    def _validate_gmail_auto_response(self, config: Dict[str, Any]) -> None:
        """Validate Gmail auto response configuration."""
        if not isinstance(config, dict):
            raise ConfigValidationError("gmail auto_response configuration must be a dictionary")

        # Validate enabled
        if "enabled" in config and not isinstance(config["enabled"], bool):
            raise ConfigValidationError("gmail auto_response enabled must be a boolean")

        # Validate response_delay
        if "response_delay" in config:
            if not isinstance(config["response_delay"], int) or config["response_delay"] < 0:
                raise ConfigValidationError("gmail auto_response response_delay must be a non-negative integer")

        # Validate max_responses_per_day
        if "max_responses_per_day" in config:
            if not isinstance(config["max_responses_per_day"], int) or config["max_responses_per_day"] <= 0:
                raise ConfigValidationError("gmail auto_response max_responses_per_day must be a positive integer")

        # Validate templates
        if "templates" in config:
            if not isinstance(config["templates"], dict):
                raise ConfigValidationError("gmail auto_response templates must be a dictionary")
            for template_name, template_content in config["templates"].items():
                if not isinstance(template_content, str):
                    raise ConfigValidationError(f"gmail auto_response template {template_name} must be a string")

        # Validate trigger_patterns
        if "trigger_patterns" in config:
            if not isinstance(config["trigger_patterns"], dict):
                raise ConfigValidationError("gmail auto_response trigger_patterns must be a dictionary")
            for pattern_name, patterns in config["trigger_patterns"].items():
                if not isinstance(patterns, list):
                    raise ConfigValidationError(f"gmail auto_response trigger_patterns for {pattern_name} must be a list")
                for pattern in patterns:
                    if not isinstance(pattern, str):
                        raise ConfigValidationError(f"gmail auto_response trigger_patterns for {pattern_name} must contain only strings")

    def _validate_gmail_archiving(self, config: Dict[str, Any]) -> None:
        """Validate Gmail archiving configuration."""
        if not isinstance(config, dict):
            raise ConfigValidationError("gmail archiving configuration must be a dictionary")

        # Validate enabled
        if "enabled" in config and not isinstance(config["enabled"], bool):
            raise ConfigValidationError("gmail archiving enabled must be a boolean")

        # Validate archive_after_days
        if "archive_after_days" in config:
            if not isinstance(config["archive_after_days"], int) or config["archive_after_days"] <= 0:
                raise ConfigValidationError("gmail archiving archive_after_days must be a positive integer")

        # Validate auto_label
        if "auto_label" in config and not isinstance(config["auto_label"], bool):
            raise ConfigValidationError("gmail archiving auto_label must be a boolean")

        # Validate label_rules
        if "label_rules" in config:
            if not isinstance(config["label_rules"], list):
                raise ConfigValidationError("gmail archiving label_rules must be a list")
            for rule in config["label_rules"]:
                if not isinstance(rule, dict):
                    raise ConfigValidationError("gmail archiving label_rules must contain dictionaries")
                if "pattern" not in rule or "label" not in rule:
                    raise ConfigValidationError("gmail archiving label_rules must contain 'pattern' and 'label' fields")
                if not isinstance(rule["pattern"], str) or not isinstance(rule["label"], str):
                    raise ConfigValidationError("gmail archiving label_rules pattern and label must be strings")

        # Validate smart_folders
        if "smart_folders" in config:
            if not isinstance(config["smart_folders"], dict):
                raise ConfigValidationError("gmail archiving smart_folders must be a dictionary")
            for folder_name, keywords in config["smart_folders"].items():
                if not isinstance(keywords, list):
                    raise ConfigValidationError(f"gmail archiving smart_folders for {folder_name} must be a list")
                for keyword in keywords:
                    if not isinstance(keyword, str):
                        raise ConfigValidationError(f"gmail archiving smart_folders for {folder_name} must contain only strings")

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
            "ollama": {},
            "agents": {}
        }

        env_mapping = {
            f"{prefix}MANAGER_MAX_AGENTS": ("agent_manager", "max_agents", int),
            f"{prefix}MANAGER_HEARTBEAT_INTERVAL": ("agent_manager", "heartbeat_interval", float),
            f"{prefix}LOGGING_LEVEL": ("logging", "level", str),
            f"{prefix}LOGGING_FORMAT": ("logging", "format", str),
            f"{prefix}COMMUNICATION_MESSAGE_BROKER_QUEUE_SIZE": ("communication", "message_broker", "queue_size", int),
            f"{prefix}COMMUNICATION_MESSAGE_BROKER_TIMEOUT": ("communication", "message_broker", "timeout", float),
            f"{prefix}OLLAMA_HOST": ("ollama", "host", str),
            f"{prefix}OLLAMA_PORT": ("ollama", "port", int),
            f"{prefix}OLLAMA_TIMEOUT": ("ollama", "timeout", float),
            f"{prefix}OLLAMA_DEFAULT_MODEL": ("ollama", "default_model", str),
            f"{prefix}OLLAMA_MAX_CONTEXT_LENGTH": ("ollama", "max_context_length", int),
            f"{prefix}OLLAMA_STREAM_ENABLED": ("ollama", "stream_enabled", lambda x: x.lower() in ('true', '1', 'yes')),
            f"{prefix}OLLAMA_RETRY_ATTEMPTS": ("ollama", "retry_attempts", int),
            f"{prefix}OLLAMA_RETRY_DELAY": ("ollama", "retry_delay", float),
            f"{prefix}OLLAMA_TEMPERATURE": ("ollama", "temperature", float),
            f"{prefix}OLLAMA_TOP_P": ("ollama", "top_p", float),
            f"{prefix}OLLAMA_TOP_K": ("ollama", "top_k", int),
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
