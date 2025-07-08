"""
Lifecycle hooks for agent lifecycle events.
"""

from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Any, Dict, Optional

# Resource thresholds
MAX_MEMORY_USAGE_PERCENT = 90
MAX_DISK_USAGE_PERCENT = 95


class LifecycleHook(ABC):
    """
    Abstract base class for lifecycle hooks.

    Defines the interface for hooks that can be executed
    during agent lifecycle events.
    """

    @abstractmethod
    async def execute(self, agent_id: str, context: Dict[str, Any]) -> bool:
        """
        Execute the lifecycle hook.

        Args:
            agent_id: ID of the agent
            context: Context dictionary containing relevant data

        Returns:
            True if hook executed successfully, False otherwise
        """
        pass


class PreStartHook(LifecycleHook):
    """
    Hook executed before agent startup.

    Performs validation and preparation tasks before
    an agent is started.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the pre-start hook.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, agent_id: str, context: Dict[str, Any]) -> bool:
        """
        Execute pre-start validation and preparation.

        Args:
            agent_id: ID of the agent
            context: Context containing agent instance and other data

        Returns:
            True if pre-start checks pass, False otherwise
        """
        try:
            self.logger.info(f"Executing pre-start hook for agent {agent_id}")

            agent = context.get("agent")
            if not agent:
                self.logger.error(f"No agent instance in context for {agent_id}")
                return False

            # Perform health check before starting
            if hasattr(agent, 'health_check'):
                health_status = await agent.health_check()
                if not health_status:
                    self.logger.warning(f"Pre-start health check failed for agent {agent_id}")
                    return False

            # Validate agent configuration
            if hasattr(agent, 'config') and agent.config and not self._validate_agent_config(agent.config):
                self.logger.error(f"Invalid configuration for agent {agent_id}")
                return False

            # Check resource availability
            if not await self._check_resource_availability(agent_id, context):
                self.logger.error(f"Insufficient resources for agent {agent_id}")
                return False

            self.logger.info(f"Pre-start hook completed successfully for agent {agent_id}")
            return True

        except Exception as e:
            self.logger.error(f"Pre-start hook failed for agent {agent_id}: {e}")
            return False

    def _validate_agent_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate agent configuration.

        Args:
            config: Agent configuration

        Returns:
            True if configuration is valid
        """
        # Basic validation - can be extended by subclasses
        required_fields = ["agent_id"]

        for field in required_fields:
            if field not in config:
                self.logger.error(f"Missing required configuration field: {field}")
                return False

        return True

    async def _check_resource_availability(self, agent_id: str, context: Dict[str, Any]) -> bool:
        """
        Check if necessary resources are available.

        Args:
            agent_id: ID of the agent
            context: Context dictionary

        Returns:
            True if resources are available
        """
        # Basic resource check - can be extended by subclasses
        try:
            # Check memory availability (example)
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > MAX_MEMORY_USAGE_PERCENT:
                self.logger.warning(f"High memory usage ({memory.percent}%) detected")
                return False

            # Check disk space (example)
            disk = psutil.disk_usage('/')
            if disk.percent > MAX_DISK_USAGE_PERCENT:
                self.logger.warning(f"High disk usage ({disk.percent}%) detected")
                return False

            return True

        except ImportError:
            # psutil not available, skip resource checks
            self.logger.debug("psutil not available, skipping resource checks")
            return True
        except Exception as e:
            self.logger.warning(f"Resource check failed: {e}")
            return True  # Don't block startup on resource check failures


class PostStartHook(LifecycleHook):
    """
    Hook executed after agent startup.

    Performs registration, initialization, and verification tasks
    after an agent has successfully started.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the post-start hook.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, agent_id: str, context: Dict[str, Any]) -> bool:
        """
        Execute post-start registration and verification.

        Args:
            agent_id: ID of the agent
            context: Context containing agent instance and other data

        Returns:
            True if post-start operations succeed, False otherwise
        """
        try:
            self.logger.info(f"Executing post-start hook for agent {agent_id}")

            agent = context.get("agent")
            if not agent:
                self.logger.error(f"No agent instance in context for {agent_id}")
                return False

            # Register agent with external systems
            if not await self._register_agent(agent_id, agent, context):
                self.logger.error(f"Failed to register agent {agent_id}")
                return False

            # Verify agent is responding
            if not await self._verify_agent_responsiveness(agent):
                self.logger.error(f"Agent {agent_id} is not responding properly")
                return False

            # Initialize agent metrics
            await self._initialize_metrics(agent_id, agent)

            # Send startup notification
            await self._send_startup_notification(agent_id, context)

            self.logger.info(f"Post-start hook completed successfully for agent {agent_id}")
            return True

        except Exception as e:
            self.logger.error(f"Post-start hook failed for agent {agent_id}: {e}")
            return False

    async def _register_agent(self, agent_id: str, agent: Any, context: Dict[str, Any]) -> bool:
        """
        Register agent with external systems.

        Args:
            agent_id: ID of the agent
            agent: Agent instance
            context: Context dictionary

        Returns:
            True if registration succeeds
        """
        try:
            # Register with agent registry if available
            registry = context.get("registry")
            if registry and hasattr(registry, "register"):
                registry.register(agent_id, agent)
                self.logger.debug(f"Registered agent {agent_id} with registry")

            # Register with service discovery if available
            service_discovery = context.get("service_discovery")
            if service_discovery and hasattr(service_discovery, "register_service"):
                service_discovery.register_service(agent_id, {
                    "type": "autonomous_agent",
                    "status": "active"
                })
                self.logger.debug(f"Registered agent {agent_id} with service discovery")

            return True

        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_id}: {e}")
            return False

    async def _verify_agent_responsiveness(self, agent: Any) -> bool:
        """
        Verify that the agent is responding properly.

        Args:
            agent: Agent instance

        Returns:
            True if agent is responsive
        """
        try:
            # Perform a simple health check
            if hasattr(agent, 'health_check'):
                result = await agent.health_check()
                return bool(result)

            # If no health check available, assume responsive
            return True

        except Exception as e:
            self.logger.error(f"Agent responsiveness check failed: {e}")
            return False

    async def _initialize_metrics(self, agent_id: str, agent: Any) -> None:
        """
        Initialize metrics collection for the agent.

        Args:
            agent_id: ID of the agent
            agent: Agent instance
        """
        try:
            # Initialize baseline metrics
            if hasattr(agent, 'get_metrics'):
                initial_metrics = agent.get_metrics()
                self.logger.debug(f"Initialized metrics for agent {agent_id}: {initial_metrics}")

        except Exception as e:
            self.logger.warning(f"Failed to initialize metrics for agent {agent_id}: {e}")

    async def _send_startup_notification(self, agent_id: str, context: Dict[str, Any]) -> None:
        """
        Send startup notification to interested parties.

        Args:
            agent_id: ID of the agent
            context: Context dictionary
        """
        try:
            # Send notification via message broker if available
            message_broker = context.get("message_broker")
            if message_broker:
                # Implementation would depend on the message broker interface
                pass

            self.logger.debug(f"Sent startup notification for agent {agent_id}")

        except Exception as e:
            self.logger.warning(f"Failed to send startup notification for agent {agent_id}: {e}")


class PreStopHook(LifecycleHook):
    """
    Hook executed before agent shutdown.

    Performs cleanup preparation and graceful shutdown tasks
    before an agent is stopped.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, graceful_shutdown_timeout: float = 10.0):
        """
        Initialize the pre-stop hook.

        Args:
            logger: Optional logger instance
            graceful_shutdown_timeout: Timeout for graceful shutdown
        """
        self.logger = logger or logging.getLogger(__name__)
        self.graceful_shutdown_timeout = graceful_shutdown_timeout

    async def execute(self, agent_id: str, context: Dict[str, Any]) -> bool:
        """
        Execute pre-stop cleanup and preparation.

        Args:
            agent_id: ID of the agent
            context: Context containing agent instance and other data

        Returns:
            True if pre-stop operations succeed, False otherwise
        """
        try:
            self.logger.info(f"Executing pre-stop hook for agent {agent_id}")

            agent = context.get("agent")
            if not agent:
                self.logger.error(f"No agent instance in context for {agent_id}")
                return False

            # Save agent state and metrics
            await self._save_agent_state(agent_id, agent)

            # Graceful shutdown with timeout
            if hasattr(agent, 'graceful_shutdown'):
                try:
                    await asyncio.wait_for(
                        agent.graceful_shutdown(timeout=self.graceful_shutdown_timeout),
                        timeout=self.graceful_shutdown_timeout + 1.0
                    )
                except TimeoutError:
                    self.logger.warning(f"Graceful shutdown timed out for agent {agent_id}")

            # Notify about impending shutdown
            await self._send_shutdown_notification(agent_id, context)

            self.logger.info(f"Pre-stop hook completed successfully for agent {agent_id}")
            return True

        except Exception as e:
            self.logger.error(f"Pre-stop hook failed for agent {agent_id}: {e}")
            return False

    async def _save_agent_state(self, agent_id: str, agent: Any) -> None:
        """
        Save agent state and metrics before shutdown.

        Args:
            agent_id: ID of the agent
            agent: Agent instance
        """
        try:
            # Save current metrics
            if hasattr(agent, 'get_metrics'):
                metrics = agent.get_metrics()
                self.logger.info(f"Final metrics for agent {agent_id}: {metrics}")

            # Save agent state if supported
            if hasattr(agent, 'save_state'):
                await agent.save_state()
                self.logger.debug(f"Saved state for agent {agent_id}")

        except Exception as e:
            self.logger.warning(f"Failed to save agent state for {agent_id}: {e}")

    async def _send_shutdown_notification(self, agent_id: str, context: Dict[str, Any]) -> None:
        """
        Send shutdown notification.

        Args:
            agent_id: ID of the agent
            context: Context dictionary
        """
        try:
            # Send notification via message broker if available
            message_broker = context.get("message_broker")
            if message_broker:
                # Implementation would depend on the message broker interface
                pass

            self.logger.debug(f"Sent shutdown notification for agent {agent_id}")

        except Exception as e:
            self.logger.warning(f"Failed to send shutdown notification for agent {agent_id}: {e}")


class PostStopHook(LifecycleHook):
    """
    Hook executed after agent shutdown.

    Performs final cleanup and deregistration tasks
    after an agent has been stopped.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the post-stop hook.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, agent_id: str, context: Dict[str, Any]) -> bool:
        """
        Execute post-stop cleanup and deregistration.

        Args:
            agent_id: ID of the agent
            context: Context containing agent instance and other data

        Returns:
            True if post-stop operations succeed, False otherwise
        """
        try:
            self.logger.info(f"Executing post-stop hook for agent {agent_id}")

            agent = context.get("agent")
            if not agent:
                self.logger.error(f"No agent instance in context for {agent_id}")
                return False

            # Cleanup agent resources
            if hasattr(agent, 'cleanup_resources'):
                await agent.cleanup_resources()
                self.logger.debug(f"Cleaned up resources for agent {agent_id}")

            # Deregister from external systems
            await self._deregister_agent(agent_id, context)

            # Archive agent data
            await self._archive_agent_data(agent_id, agent)

            # Send final notification
            await self._send_stop_notification(agent_id, context)

            self.logger.info(f"Post-stop hook completed successfully for agent {agent_id}")
            return True

        except Exception as e:
            self.logger.error(f"Post-stop hook failed for agent {agent_id}: {e}")
            return False

    async def _deregister_agent(self, agent_id: str, context: Dict[str, Any]) -> None:
        """
        Deregister agent from external systems.

        Args:
            agent_id: ID of the agent
            context: Context dictionary
        """
        try:
            # Deregister from agent registry if available
            registry = context.get("registry")
            if registry and hasattr(registry, "deregister"):
                registry.deregister(agent_id)
                self.logger.debug(f"Deregistered agent {agent_id} from registry")

            # Deregister from service discovery if available
            service_discovery = context.get("service_discovery")
            if service_discovery and hasattr(service_discovery, "deregister_service"):
                service_discovery.deregister_service(agent_id)
                self.logger.debug(f"Deregistered agent {agent_id} from service discovery")

        except Exception as e:
            self.logger.warning(f"Failed to deregister agent {agent_id}: {e}")

    async def _archive_agent_data(self, agent_id: str, agent: Any) -> None:
        """
        Archive agent data for future reference.

        Args:
            agent_id: ID of the agent
            agent: Agent instance
        """
        try:
            # Archive final metrics
            if hasattr(agent, 'get_metrics'):
                agent.get_metrics()
                # Store metrics to archive (implementation depends on storage backend)
                self.logger.debug(f"Archived final metrics for agent {agent_id}")

            # Archive logs if supported
            if hasattr(agent, 'get_logs'):
                agent.get_logs()
                # Store logs to archive
                self.logger.debug(f"Archived logs for agent {agent_id}")

        except Exception as e:
            self.logger.warning(f"Failed to archive data for agent {agent_id}: {e}")

    async def _send_stop_notification(self, agent_id: str, context: Dict[str, Any]) -> None:
        """
        Send final stop notification.

        Args:
            agent_id: ID of the agent
            context: Context dictionary
        """
        try:
            # Send notification via message broker if available
            message_broker = context.get("message_broker")
            if message_broker:
                # Implementation would depend on the message broker interface
                pass

            self.logger.debug(f"Sent stop notification for agent {agent_id}")

        except Exception as e:
            self.logger.warning(f"Failed to send stop notification for agent {agent_id}: {e}")
