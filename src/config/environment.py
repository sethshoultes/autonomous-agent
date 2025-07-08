"""
Environment-specific configuration management for the Autonomous Agent System.

This module handles loading and merging of environment-specific configurations
including development, staging, and production settings.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

from .manager import ConfigManager, ConfigLoader
from .exceptions import ConfigError, ConfigNotFoundError


class Environment(Enum):
    """Supported environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class ConfigProfile:
    """Configuration profile for specific environment."""
    name: str
    environment: Environment
    config_path: Path
    base_config_path: Optional[Path] = None
    secrets_path: Optional[Path] = None
    env_file_path: Optional[Path] = None
    overrides: Dict[str, Any] = field(default_factory=dict)


class EnvironmentSettings(BaseSettings):
    """Environment-specific settings with validation."""
    
    # Environment identification
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Application settings
    app_name: str = Field(default="Autonomous Agent System", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Database settings
    database_url: str = Field(default="postgresql://agent:secure_password@localhost:5432/autonomous_agent", env="DATABASE_URL")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    
    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Security settings - MUST be set via environment variable in production
    jwt_secret: str = Field(env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    
    # External service settings
    ollama_url: str = Field(default="http://localhost:11434", env="OLLAMA_URL")
    gmail_credentials_path: Optional[str] = Field(default=None, env="GMAIL_CREDENTIALS_PATH")
    github_token: Optional[str] = Field(default=None, env="GITHUB_TOKEN")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    
    # Performance settings
    max_request_size: int = Field(default=10485760, env="MAX_REQUEST_SIZE")  # 10MB
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    worker_processes: int = Field(default=4, env="WORKER_PROCESSES")
    
    # Feature flags
    enable_monitoring: bool = Field(default=True, env="ENABLE_MONITORING")
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    enable_cors: bool = Field(default=True, env="ENABLE_CORS")
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment value."""
        valid_environments = [env.value for env in Environment]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @field_validator('port')
    @classmethod
    def validate_port(cls, v):
        """Validate port number."""
        if not (1 <= v <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @field_validator('jwt_secret')
    @classmethod
    def validate_jwt_secret(cls, v):
        """Validate JWT secret security."""
        if not v:
            raise ValueError("JWT secret is required")
        if len(v) < 32:
            raise ValueError("JWT secret must be at least 32 characters long")
        if v == "your-secret-key-change-me":
            raise ValueError("JWT secret cannot use default value - set JWT_SECRET environment variable")
        return v
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


class EnvironmentConfigManager:
    """Enhanced configuration manager with environment-specific support."""
    
    def __init__(self, 
                 config_dir: Union[str, Path] = "config",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the environment configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config_dir = Path(config_dir)
        self.base_manager = ConfigManager(logger)
        self.loader = ConfigLoader()
        
        # Current configuration state
        self.current_environment: Optional[Environment] = None
        self.current_profile: Optional[ConfigProfile] = None
        self.settings: Optional[EnvironmentSettings] = None
        self.merged_config: Optional[Dict[str, Any]] = None
        
        # Available profiles
        self.profiles: Dict[str, ConfigProfile] = {}
        
        # Load available profiles
        self._discover_profiles()
    
    def _discover_profiles(self) -> None:
        """Discover available configuration profiles."""
        if not self.config_dir.exists():
            self.logger.warning(f"Configuration directory not found: {self.config_dir}")
            return
        
        # Base configuration
        base_config_path = self.config_dir / "base.yml"
        if not base_config_path.exists():
            base_config_path = self.config_dir / "base.yaml"
        
        # Environment-specific configurations
        for env in Environment:
            env_dir = self.config_dir / env.value
            if env_dir.exists():
                config_file = env_dir / "app.yml"
                if not config_file.exists():
                    config_file = env_dir / "app.yaml"
                
                if config_file.exists():
                    profile = ConfigProfile(
                        name=env.value,
                        environment=env,
                        config_path=config_file,
                        base_config_path=base_config_path if base_config_path.exists() else None,
                        secrets_path=self.config_dir.parent / "secrets",
                        env_file_path=self.config_dir.parent / f".env.{env.value}"
                    )
                    self.profiles[env.value] = profile
                    self.logger.debug(f"Discovered profile: {env.value}")
    
    def load_environment_config(self, environment: Union[str, Environment]) -> None:
        """
        Load configuration for a specific environment.
        
        Args:
            environment: Environment name or enum
            
        Raises:
            ConfigError: If environment configuration cannot be loaded
        """
        if isinstance(environment, str):
            try:
                env_enum = Environment(environment)
            except ValueError:
                raise ConfigError(f"Unknown environment: {environment}")
        else:
            env_enum = environment
        
        self.current_environment = env_enum
        env_name = env_enum.value
        
        # Load environment settings
        env_file_path = self.config_dir.parent / f".env.{env_name}"
        if env_file_path.exists():
            self.settings = EnvironmentSettings(_env_file=str(env_file_path))
        else:
            self.settings = EnvironmentSettings()
        
        # Get profile
        profile = self.profiles.get(env_name)
        if not profile:
            raise ConfigError(f"No configuration profile found for environment: {env_name}")
        
        self.current_profile = profile
        
        # Load and merge configurations
        self.merged_config = self._load_merged_config(profile)
        
        # Apply environment variable overrides
        self._apply_environment_overrides()
        
        # Load into base manager
        self.base_manager.load_config(self.merged_config)
        
        self.logger.info(f"Loaded configuration for environment: {env_name}")
    
    def _load_merged_config(self, profile: ConfigProfile) -> Dict[str, Any]:
        """Load and merge configuration from multiple sources."""
        merged_config = {}
        
        # Load base configuration
        if profile.base_config_path and profile.base_config_path.exists():
            try:
                base_config = self.loader.load_from_file(profile.base_config_path)
                merged_config = self.loader.merge_configs(merged_config, base_config)
                self.logger.debug(f"Loaded base config from: {profile.base_config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load base config: {e}")
                raise ConfigError(f"Failed to load base configuration: {e}")
        
        # Load environment-specific configuration
        try:
            env_config = self.loader.load_from_file(profile.config_path)
            merged_config = self.loader.merge_configs(merged_config, env_config)
            self.logger.debug(f"Loaded environment config from: {profile.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load environment config: {e}")
            raise ConfigError(f"Failed to load environment configuration: {e}")
        
        # Apply profile overrides
        if profile.overrides:
            merged_config = self.loader.merge_configs(merged_config, profile.overrides)
            self.logger.debug("Applied profile overrides")
        
        return merged_config
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides."""
        if not self.settings or not self.merged_config:
            return
        
        # Map environment settings to config structure
        env_overrides = {
            "app": {
                "name": self.settings.app_name,
                "version": self.settings.app_version,
                "host": self.settings.host,
                "port": self.settings.port,
                "debug": self.settings.debug
            },
            "database": {
                "url": self.settings.database_url,
                "echo": self.settings.database_echo,
                "pool_size": self.settings.database_pool_size
            },
            "redis": {
                "url": self.settings.redis_url,
                "password": self.settings.redis_password
            },
            "security": {
                "jwt_secret": self.settings.jwt_secret,
                "jwt_algorithm": self.settings.jwt_algorithm
            },
            "external_services": {
                "ollama": {
                    "url": self.settings.ollama_url
                },
                "gmail": {
                    "credentials_path": self.settings.gmail_credentials_path
                },
                "github": {
                    "token": self.settings.github_token
                }
            },
            "logging": {
                "level": self.settings.log_level,
                "format": self.settings.log_format
            },
            "performance": {
                "max_request_size": self.settings.max_request_size,
                "request_timeout": self.settings.request_timeout,
                "worker_processes": self.settings.worker_processes
            },
            "features": {
                "monitoring_enabled": self.settings.enable_monitoring,
                "rate_limiting_enabled": self.settings.enable_rate_limiting,
                "cors_enabled": self.settings.enable_cors
            }
        }
        
        # Remove None values
        env_overrides = self._remove_none_values(env_overrides)
        
        # Merge with existing config
        self.merged_config = self.loader.merge_configs(self.merged_config, env_overrides)
        self.logger.debug("Applied environment variable overrides")
    
    def _remove_none_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values from configuration dictionary."""
        cleaned = {}
        for key, value in config.items():
            if value is not None:
                if isinstance(value, dict):
                    cleaned_value = self._remove_none_values(value)
                    if cleaned_value:  # Only add non-empty dictionaries
                        cleaned[key] = cleaned_value
                else:
                    cleaned[key] = value
        return cleaned
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the current environment."""
        return {
            "environment": self.current_environment.value if self.current_environment else None,
            "profile": self.current_profile.name if self.current_profile else None,
            "config_dir": str(self.config_dir),
            "available_profiles": list(self.profiles.keys()),
            "settings": self.settings.dict() if self.settings else None
        }
    
    def get_secrets_config(self) -> Dict[str, Any]:
        """Get secrets configuration for the current environment."""
        if not self.current_profile or not self.current_profile.secrets_path:
            return {}
        
        secrets_path = self.current_profile.secrets_path
        if not secrets_path.exists():
            self.logger.warning(f"Secrets directory not found: {secrets_path}")
            return {}
        
        secrets_config = {}
        
        # Load secrets from files
        secret_files = {
            "postgres_password": secrets_path / "postgres_password.txt",
            "redis_password": secrets_path / "redis_password.txt",
            "jwt_secret": secrets_path / "jwt_secret.txt",
            "github_token": secrets_path / "github_token.txt",
            "gmail_credentials": secrets_path / "gmail_credentials.json"
        }
        
        for secret_name, secret_file in secret_files.items():
            if secret_file.exists():
                try:
                    if secret_file.suffix == '.json':
                        with open(secret_file, 'r') as f:
                            secrets_config[secret_name] = f.read().strip()
                    else:
                        with open(secret_file, 'r') as f:
                            secrets_config[secret_name] = f.read().strip()
                except Exception as e:
                    self.logger.error(f"Failed to read secret {secret_name}: {e}")
        
        return secrets_config
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the current configuration."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if not self.merged_config:
            validation_results["valid"] = False
            validation_results["errors"].append("No configuration loaded")
            return validation_results
        
        # Validate required sections
        required_sections = ["app", "database", "redis", "security", "logging"]
        for section in required_sections:
            if section not in self.merged_config:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Missing required section: {section}")
        
        # Validate database configuration
        if "database" in self.merged_config:
            db_config = self.merged_config["database"]
            if not db_config.get("url"):
                validation_results["valid"] = False
                validation_results["errors"].append("Database URL is required")
        
        # Validate security configuration
        if "security" in self.merged_config:
            security_config = self.merged_config["security"]
            if not security_config.get("jwt_secret"):
                validation_results["valid"] = False
                validation_results["errors"].append("JWT secret is required")
            elif security_config["jwt_secret"] == "your-secret-key-change-me":
                validation_results["warnings"].append("JWT secret is using default value")
        
        # Validate external services
        if "external_services" in self.merged_config:
            services = self.merged_config["external_services"]
            if "ollama" in services and not services["ollama"].get("url"):
                validation_results["warnings"].append("Ollama URL not configured")
            if "gmail" in services and not services["gmail"].get("credentials_path"):
                validation_results["warnings"].append("Gmail credentials not configured")
            if "github" in services and not services["github"].get("token"):
                validation_results["warnings"].append("GitHub token not configured")
        
        return validation_results
    
    def get_config_for_service(self, service_name: str) -> Dict[str, Any]:
        """Get configuration for a specific service."""
        if not self.merged_config:
            return {}
        
        service_config = {}
        
        # Get service-specific configuration
        if service_name in self.merged_config:
            service_config.update(self.merged_config[service_name])
        
        # Get from external services
        external_services = self.merged_config.get("external_services", {})
        if service_name in external_services:
            service_config.update(external_services[service_name])
        
        # Add common configuration
        common_config = {
            "app_name": self.merged_config.get("app", {}).get("name"),
            "app_version": self.merged_config.get("app", {}).get("version"),
            "environment": self.current_environment.value if self.current_environment else None,
            "debug": self.merged_config.get("app", {}).get("debug", False),
            "log_level": self.merged_config.get("logging", {}).get("level", "INFO")
        }
        
        service_config.update(common_config)
        
        return service_config
    
    def create_profile(self, 
                      name: str, 
                      environment: Environment,
                      base_config: Optional[Dict[str, Any]] = None) -> ConfigProfile:
        """Create a new configuration profile."""
        env_dir = self.config_dir / environment.value
        env_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = env_dir / "app.yml"
        
        # Create default configuration if none provided
        if base_config is None:
            base_config = {
                "app": {
                    "debug": environment == Environment.DEVELOPMENT,
                    "environment": environment.value
                },
                "logging": {
                    "level": "DEBUG" if environment == Environment.DEVELOPMENT else "INFO"
                }
            }
        
        # Save configuration
        with open(config_file, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False)
        
        # Create profile
        profile = ConfigProfile(
            name=name,
            environment=environment,
            config_path=config_file,
            base_config_path=self.config_dir / "base.yml",
            secrets_path=self.config_dir.parent / "secrets",
            env_file_path=self.config_dir.parent / f".env.{environment.value}"
        )
        
        self.profiles[name] = profile
        self.logger.info(f"Created configuration profile: {name}")
        
        return profile
    
    # Delegate methods to base manager
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.base_manager.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        return self.base_manager.set(key, value)
    
    def get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """Get agent configuration."""
        return self.base_manager.get_agent_config(agent_id)
    
    def list_agents(self) -> List[str]:
        """List all agents."""
        return self.base_manager.list_agents()
    
    def list_enabled_agents(self) -> List[str]:
        """List enabled agents."""
        return self.base_manager.list_enabled_agents()


def get_current_environment() -> Environment:
    """Get the current environment from environment variables."""
    env_name = os.getenv("ENVIRONMENT", "development").lower()
    try:
        return Environment(env_name)
    except ValueError:
        return Environment.DEVELOPMENT


def create_environment_config(config_dir: Union[str, Path] = "config") -> EnvironmentConfigManager:
    """Create an environment configuration manager."""
    return EnvironmentConfigManager(config_dir)