"""
Tests for the configuration management system.
"""

import pytest
import tempfile
import os
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.config.manager import ConfigManager, ConfigValidator, ConfigLoader, ConfigSchema
from src.config.exceptions import ConfigError, ConfigValidationError, ConfigNotFoundError


class TestConfigSchema:
    """Test ConfigSchema class."""
    
    def test_schema_initialization(self):
        """Test schema initialization."""
        schema = ConfigSchema()
        
        assert schema.agent_manager is not None
        assert schema.logging is not None
        assert schema.communication is not None
        assert schema.agents is not None
    
    def test_schema_to_dict(self):
        """Test converting schema to dict."""
        schema = ConfigSchema()
        schema_dict = schema.to_dict()
        
        assert "agent_manager" in schema_dict
        assert "logging" in schema_dict
        assert "communication" in schema_dict
        assert "agents" in schema_dict
    
    def test_schema_from_dict(self):
        """Test creating schema from dict."""
        schema_data = {
            "agent_manager": {
                "max_agents": 10,
                "heartbeat_interval": 30
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "communication": {
                "message_broker": {
                    "queue_size": 1000,
                    "timeout": 10
                }
            },
            "agents": {}
        }
        
        schema = ConfigSchema.from_dict(schema_data)
        
        assert schema.agent_manager["max_agents"] == 10
        assert schema.logging["level"] == "INFO"
        assert schema.communication["message_broker"]["queue_size"] == 1000


class TestConfigValidator:
    """Test ConfigValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a config validator for testing."""
        return ConfigValidator()
    
    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.schema is not None
        assert validator.required_keys is not None
    
    def test_validate_config_structure(self, validator):
        """Test validating config structure."""
        valid_config = {
            "agent_manager": {
                "max_agents": 10,
                "heartbeat_interval": 30
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "communication": {
                "message_broker": {
                    "queue_size": 1000,
                    "timeout": 10
                }
            },
            "agents": {}
        }
        
        # Should not raise an exception
        validator.validate(valid_config)
    
    def test_validate_missing_required_keys(self, validator):
        """Test validation with missing required keys."""
        invalid_config = {
            "agent_manager": {
                "max_agents": 10
            }
            # Missing logging, communication, agents
        }
        
        with pytest.raises(ConfigValidationError):
            validator.validate(invalid_config)
    
    def test_validate_invalid_data_types(self, validator):
        """Test validation with invalid data types."""
        invalid_config = {
            "agent_manager": {
                "max_agents": "not_a_number",  # Should be int
                "heartbeat_interval": 30
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "communication": {
                "message_broker": {
                    "queue_size": 1000,
                    "timeout": 10
                }
            },
            "agents": {}
        }
        
        with pytest.raises(ConfigValidationError):
            validator.validate(invalid_config)
    
    def test_validate_invalid_values(self, validator):
        """Test validation with invalid values."""
        invalid_config = {
            "agent_manager": {
                "max_agents": -1,  # Should be positive
                "heartbeat_interval": 30
            },
            "logging": {
                "level": "INVALID_LEVEL",  # Should be valid log level
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "communication": {
                "message_broker": {
                    "queue_size": 1000,
                    "timeout": 10
                }
            },
            "agents": {}
        }
        
        with pytest.raises(ConfigValidationError):
            validator.validate(invalid_config)
    
    def test_validate_agent_configurations(self, validator):
        """Test validating agent configurations."""
        config_with_agents = {
            "agent_manager": {
                "max_agents": 10,
                "heartbeat_interval": 30
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "communication": {
                "message_broker": {
                    "queue_size": 1000,
                    "timeout": 10
                }
            },
            "agents": {
                "gmail_agent": {
                    "agent_type": "GmailAgent",
                    "enabled": True,
                    "priority": 1,
                    "config": {
                        "email": "test@example.com",
                        "check_interval": 300
                    }
                },
                "research_agent": {
                    "agent_type": "ResearchAgent",
                    "enabled": True,
                    "priority": 2,
                    "config": {
                        "sources": ["rss", "web"],
                        "update_interval": 3600
                    }
                }
            }
        }
        
        # Should not raise an exception
        validator.validate(config_with_agents)
    
    def test_validate_invalid_agent_config(self, validator):
        """Test validating invalid agent configuration."""
        invalid_agent_config = {
            "agent_manager": {
                "max_agents": 10,
                "heartbeat_interval": 30
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "communication": {
                "message_broker": {
                    "queue_size": 1000,
                    "timeout": 10
                }
            },
            "agents": {
                "invalid_agent": {
                    # Missing required fields
                    "config": {}
                }
            }
        }
        
        with pytest.raises(ConfigValidationError):
            validator.validate(invalid_agent_config)


class TestConfigLoader:
    """Test ConfigLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create a config loader for testing."""
        return ConfigLoader()
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        config_data = {
            "agent_manager": {
                "max_agents": 10,
                "heartbeat_interval": 30
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "communication": {
                "message_broker": {
                    "queue_size": 1000,
                    "timeout": 10
                }
            },
            "agents": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        os.unlink(temp_file)
    
    @pytest.fixture
    def temp_yaml_config_file(self):
        """Create a temporary YAML config file for testing."""
        config_data = {
            "agent_manager": {
                "max_agents": 10,
                "heartbeat_interval": 30
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "communication": {
                "message_broker": {
                    "queue_size": 1000,
                    "timeout": 10
                }
            },
            "agents": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        os.unlink(temp_file)
    
    def test_loader_initialization(self, loader):
        """Test loader initialization."""
        assert loader.supported_formats == ['.json', '.yaml', '.yml']
    
    def test_load_json_config(self, loader, temp_config_file):
        """Test loading JSON config file."""
        config = loader.load_from_file(temp_config_file)
        
        assert config["agent_manager"]["max_agents"] == 10
        assert config["logging"]["level"] == "INFO"
        assert config["communication"]["message_broker"]["queue_size"] == 1000
    
    def test_load_yaml_config(self, loader, temp_yaml_config_file):
        """Test loading YAML config file."""
        config = loader.load_from_file(temp_yaml_config_file)
        
        assert config["agent_manager"]["max_agents"] == 10
        assert config["logging"]["level"] == "INFO"
        assert config["communication"]["message_broker"]["queue_size"] == 1000
    
    def test_load_nonexistent_file(self, loader):
        """Test loading non-existent file."""
        with pytest.raises(ConfigNotFoundError):
            loader.load_from_file("/nonexistent/config.json")
    
    def test_load_invalid_format(self, loader):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("not a config file")
            temp_file = f.name
        
        try:
            with pytest.raises(ConfigError):
                loader.load_from_file(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_load_invalid_json(self, loader):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name
        
        try:
            with pytest.raises(ConfigError):
                loader.load_from_file(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_load_from_dict(self, loader):
        """Test loading config from dictionary."""
        config_dict = {
            "agent_manager": {
                "max_agents": 10,
                "heartbeat_interval": 30
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "communication": {
                "message_broker": {
                    "queue_size": 1000,
                    "timeout": 10
                }
            },
            "agents": {}
        }
        
        config = loader.load_from_dict(config_dict)
        
        assert config == config_dict
    
    def test_load_from_environment(self, loader):
        """Test loading config from environment variables."""
        env_vars = {
            "AGENT_MANAGER_MAX_AGENTS": "20",
            "AGENT_LOGGING_LEVEL": "DEBUG",
            "AGENT_COMMUNICATION_MESSAGE_BROKER_QUEUE_SIZE": "2000"
        }
        
        with patch.dict(os.environ, env_vars):
            config = loader.load_from_environment()
        
        assert config["agent_manager"]["max_agents"] == 20
        assert config["logging"]["level"] == "DEBUG"
        assert config["communication"]["message_broker"]["queue_size"] == 2000
    
    def test_merge_configs(self, loader):
        """Test merging multiple configurations."""
        base_config = {
            "agent_manager": {
                "max_agents": 10,
                "heartbeat_interval": 30
            },
            "logging": {
                "level": "INFO"
            }
        }
        
        override_config = {
            "agent_manager": {
                "max_agents": 20  # Override
            },
            "logging": {
                "format": "%(asctime)s - %(message)s"  # Add new field
            },
            "communication": {  # Add new section
                "message_broker": {
                    "queue_size": 1000
                }
            }
        }
        
        merged_config = loader.merge_configs(base_config, override_config)
        
        assert merged_config["agent_manager"]["max_agents"] == 20  # Overridden
        assert merged_config["agent_manager"]["heartbeat_interval"] == 30  # Preserved
        assert merged_config["logging"]["level"] == "INFO"  # Preserved
        assert merged_config["logging"]["format"] == "%(asctime)s - %(message)s"  # Added
        assert merged_config["communication"]["message_broker"]["queue_size"] == 1000  # Added


class TestConfigManager:
    """Test ConfigManager class."""
    
    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock()
    
    @pytest.fixture
    def manager(self, mock_logger):
        """Create a config manager for testing."""
        return ConfigManager(logger=mock_logger)
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        config_data = {
            "agent_manager": {
                "max_agents": 10,
                "heartbeat_interval": 30
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "communication": {
                "message_broker": {
                    "queue_size": 1000,
                    "timeout": 10
                }
            },
            "agents": {
                "gmail_agent": {
                    "agent_type": "GmailAgent",
                    "enabled": True,
                    "priority": 1,
                    "config": {
                        "email": "test@example.com",
                        "check_interval": 300
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        os.unlink(temp_file)
    
    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager.logger is not None
        assert isinstance(manager.loader, ConfigLoader)
        assert isinstance(manager.validator, ConfigValidator)
        assert manager.config is None
    
    def test_load_config_from_file(self, manager, temp_config_file):
        """Test loading config from file."""
        manager.load_config(temp_config_file)
        
        assert manager.config is not None
        assert manager.config["agent_manager"]["max_agents"] == 10
        assert manager.config["logging"]["level"] == "INFO"
        assert "gmail_agent" in manager.config["agents"]
    
    def test_load_config_from_dict(self, manager):
        """Test loading config from dictionary."""
        config_dict = {
            "agent_manager": {
                "max_agents": 10,
                "heartbeat_interval": 30
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "communication": {
                "message_broker": {
                    "queue_size": 1000,
                    "timeout": 10
                }
            },
            "agents": {}
        }
        
        manager.load_config(config_dict)
        
        assert manager.config is not None
        assert manager.config["agent_manager"]["max_agents"] == 10
    
    def test_load_invalid_config(self, manager):
        """Test loading invalid config."""
        invalid_config = {
            "agent_manager": {
                "max_agents": "not_a_number"
            }
        }
        
        with pytest.raises(ConfigValidationError):
            manager.load_config(invalid_config)
    
    def test_get_config_value(self, manager, temp_config_file):
        """Test getting config values."""
        manager.load_config(temp_config_file)
        
        # Test getting nested values
        assert manager.get("agent_manager.max_agents") == 10
        assert manager.get("logging.level") == "INFO"
        assert manager.get("agents.gmail_agent.enabled") is True
        
        # Test getting with default value
        assert manager.get("nonexistent.key", "default") == "default"
    
    def test_set_config_value(self, manager, temp_config_file):
        """Test setting config values."""
        manager.load_config(temp_config_file)
        
        # Test setting nested values
        manager.set("agent_manager.max_agents", 20)
        assert manager.get("agent_manager.max_agents") == 20
        
        # Test setting new values
        manager.set("new_section.new_key", "new_value")
        assert manager.get("new_section.new_key") == "new_value"
    
    def test_get_agent_config(self, manager, temp_config_file):
        """Test getting agent configuration."""
        manager.load_config(temp_config_file)
        
        agent_config = manager.get_agent_config("gmail_agent")
        
        assert agent_config is not None
        assert agent_config["agent_type"] == "GmailAgent"
        assert agent_config["enabled"] is True
        assert agent_config["config"]["email"] == "test@example.com"
    
    def test_get_nonexistent_agent_config(self, manager, temp_config_file):
        """Test getting non-existent agent configuration."""
        manager.load_config(temp_config_file)
        
        with pytest.raises(ConfigNotFoundError):
            manager.get_agent_config("nonexistent_agent")
    
    def test_list_agents(self, manager, temp_config_file):
        """Test listing configured agents."""
        manager.load_config(temp_config_file)
        
        agents = manager.list_agents()
        
        assert len(agents) == 1
        assert "gmail_agent" in agents
    
    def test_list_enabled_agents(self, manager, temp_config_file):
        """Test listing enabled agents."""
        manager.load_config(temp_config_file)
        
        # Add a disabled agent
        manager.set("agents.disabled_agent.enabled", False)
        manager.set("agents.disabled_agent.agent_type", "DisabledAgent")
        
        enabled_agents = manager.list_enabled_agents()
        
        assert len(enabled_agents) == 1
        assert "gmail_agent" in enabled_agents
        assert "disabled_agent" not in enabled_agents
    
    def test_save_config(self, manager, temp_config_file):
        """Test saving config to file."""
        manager.load_config(temp_config_file)
        
        # Modify config
        manager.set("agent_manager.max_agents", 20)
        
        # Save to new file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            save_file = f.name
        
        try:
            manager.save_config(save_file)
            
            # Load the saved config and verify
            with open(save_file, 'r') as f:
                saved_config = json.load(f)
            
            assert saved_config["agent_manager"]["max_agents"] == 20
        finally:
            os.unlink(save_file)
    
    def test_reload_config(self, manager, temp_config_file):
        """Test reloading config."""
        manager.load_config(temp_config_file)
        
        # Modify config in memory
        manager.set("agent_manager.max_agents", 20)
        assert manager.get("agent_manager.max_agents") == 20
        
        # Reload from file
        manager.reload_config()
        
        # Should be back to original value
        assert manager.get("agent_manager.max_agents") == 10
    
    def test_validate_current_config(self, manager, temp_config_file):
        """Test validating current config."""
        manager.load_config(temp_config_file)
        
        # Should not raise an exception
        manager.validate_config()
        
        # Corrupt the config
        manager.config["agent_manager"]["max_agents"] = "not_a_number"
        
        # Should raise an exception
        with pytest.raises(ConfigValidationError):
            manager.validate_config()
    
    def test_config_change_callbacks(self, manager, temp_config_file):
        """Test config change callbacks."""
        manager.load_config(temp_config_file)
        
        # Register a callback
        callback_calls = []
        
        def test_callback(key, old_value, new_value):
            callback_calls.append((key, old_value, new_value))
        
        manager.register_change_callback("agent_manager.max_agents", test_callback)
        
        # Change the value
        manager.set("agent_manager.max_agents", 20)
        
        # Verify callback was called
        assert len(callback_calls) == 1
        assert callback_calls[0] == ("agent_manager.max_agents", 10, 20)
    
    def test_config_environment_override(self, manager, temp_config_file):
        """Test environment variable override."""
        manager.load_config(temp_config_file)
        
        # Set environment variable
        with patch.dict(os.environ, {"AGENT_MANAGER_MAX_AGENTS": "30"}):
            manager.apply_environment_overrides()
        
        assert manager.get("agent_manager.max_agents") == 30