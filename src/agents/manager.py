"""
Agent Manager and Registry for coordinating autonomous agents.
"""

import asyncio
import contextlib
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

from .base import AgentMessage, AgentState, BaseAgent
from .exceptions import AgentManagerError, AgentNotFoundError, AgentRegistrationError


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_id: str
    agent_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "config": self.config,
            "enabled": self.enabled,
            "priority": self.priority
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            config=data.get("config", {}),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 1)
        )


class AgentRegistry:
    """
    Registry for managing agent instances and configurations.

    Provides centralized storage and retrieval of agents following
    the Single Responsibility Principle.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self._lock = asyncio.Lock()

    def register_agent(self, agent: BaseAgent, config: AgentConfig) -> None:
        """
        Register an agent with the registry.

        Args:
            agent: Agent instance to register
            config: Agent configuration

        Raises:
            AgentRegistrationError: If agent is already registered
        """
        if agent.agent_id in self.agents:
            raise AgentRegistrationError(
                f"Agent {agent.agent_id} is already registered",
                context={"agent_id": agent.agent_id}
            )

        self.agents[agent.agent_id] = agent
        self.agent_configs[agent.agent_id] = config

    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the registry.

        Args:
            agent_id: ID of the agent to unregister

        Raises:
            AgentNotFoundError: If agent is not found
        """
        if agent_id not in self.agents:
            raise AgentNotFoundError(
                f"Agent {agent_id} not found",
                context={"agent_id": agent_id}
            )

        del self.agents[agent_id]
        del self.agent_configs[agent_id]

    def get_agent(self, agent_id: str) -> BaseAgent:
        """
        Get an agent by ID.

        Args:
            agent_id: ID of the agent to retrieve

        Returns:
            Agent instance

        Raises:
            AgentNotFoundError: If agent is not found
        """
        if agent_id not in self.agents:
            raise AgentNotFoundError(
                f"Agent {agent_id} not found",
                context={"agent_id": agent_id}
            )

        return self.agents[agent_id]

    def get_agent_config(self, agent_id: str) -> AgentConfig:
        """
        Get agent configuration by ID.

        Args:
            agent_id: ID of the agent

        Returns:
            Agent configuration

        Raises:
            AgentNotFoundError: If agent is not found
        """
        if agent_id not in self.agent_configs:
            raise AgentNotFoundError(
                f"Agent config for {agent_id} not found",
                context={"agent_id": agent_id}
            )

        return self.agent_configs[agent_id]

    def list_agents(self) -> List[BaseAgent]:
        """
        List all registered agents.

        Returns:
            List of agent instances
        """
        return list(self.agents.values())

    def list_agent_configs(self) -> List[AgentConfig]:
        """
        List all agent configurations.

        Returns:
            List of agent configurations
        """
        return list(self.agent_configs.values())

    def list_agent_ids(self) -> List[str]:
        """
        List all registered agent IDs.

        Returns:
            List of agent IDs
        """
        return list(self.agents.keys())

    def get_agents_by_state(self, state: AgentState) -> List[BaseAgent]:
        """
        Get agents by state.

        Args:
            state: Agent state to filter by

        Returns:
            List of agents in the specified state
        """
        return [agent for agent in self.agents.values() if agent.state == state]

    def get_enabled_agents(self) -> List[BaseAgent]:
        """
        Get all enabled agents.

        Returns:
            List of enabled agents
        """
        enabled_agent_ids = [
            config.agent_id for config in self.agent_configs.values()
            if config.enabled
        ]
        return [self.agents[agent_id] for agent_id in enabled_agent_ids if agent_id in self.agents]


class AgentManager:
    """
    Manager for coordinating multiple autonomous agents.

    Provides centralized management, communication coordination,
    and lifecycle management for all agents in the system.
    Follows the Dependency Inversion Principle with injected dependencies.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        message_broker: Any  # Avoid circular import
    ):
        """
        Initialize the agent manager.

        Args:
            config: Manager configuration
            logger: Logger instance
            message_broker: Message broker for communication
        """
        self.config = config
        self.logger = logger
        self.message_broker = message_broker
        self.registry = AgentRegistry()
        self.is_running = False

        # Manager configuration
        self.max_agents = config.get("max_agents", 100)
        self.heartbeat_interval = config.get("heartbeat_interval", 30)
        self.communication_timeout = config.get("communication_timeout", 10)
        self.retry_attempts = config.get("retry_attempts", 3)

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the agent manager."""
        if self.is_running:
            self.logger.warning("Agent manager is already running")
            return

        self.logger.info("Starting agent manager")
        self.is_running = True

        # Start heartbeat monitoring
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self.logger.info("Agent manager started successfully")

    async def stop(self) -> None:
        """Stop the agent manager."""
        if not self.is_running:
            self.logger.warning("Agent manager is not running")
            return

        self.logger.info("Stopping agent manager")
        self.is_running = False

        # Signal shutdown
        self._shutdown_event.set()

        # Stop all agents
        await self.stop_all_agents()

        # Cancel heartbeat task
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

        self.logger.info("Agent manager stopped successfully")

    async def register_agent(self, agent: BaseAgent, config: AgentConfig) -> None:
        """
        Register an agent with the manager.

        Args:
            agent: Agent instance to register
            config: Agent configuration

        Raises:
            AgentManagerError: If registration fails
        """
        try:
            if len(self.registry.agents) >= self.max_agents:
                raise AgentManagerError(
                    f"Maximum number of agents ({self.max_agents}) reached",
                    context={"max_agents": self.max_agents, "current_count": len(self.registry.agents)}
                )

            self.registry.register_agent(agent, config)
            self.logger.info(f"Registered agent {agent.agent_id}")

        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            raise AgentManagerError("Agent registration failed", cause=e) from e

    async def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the manager.

        Args:
            agent_id: ID of the agent to unregister

        Raises:
            AgentManagerError: If unregistration fails
        """
        try:
            agent = self.registry.get_agent(agent_id)

            # Stop the agent if it's running
            if agent.state == AgentState.ACTIVE:
                await agent.stop()

            self.registry.unregister_agent(agent_id)
            self.logger.info(f"Unregistered agent {agent_id}")

        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {e}")
            raise AgentManagerError("Agent unregistration failed", cause=e) from e

    async def start_agent(self, agent_id: str) -> None:
        """
        Start a specific agent.

        Args:
            agent_id: ID of the agent to start

        Raises:
            AgentNotFoundError: If agent is not found
            AgentManagerError: If start fails
        """
        try:
            agent = self.registry.get_agent(agent_id)
            await agent.start()
            self.logger.info(f"Started agent {agent_id}")

        except AgentNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to start agent {agent_id}: {e}")
            raise AgentManagerError(f"Failed to start agent {agent_id}", cause=e) from e

    async def stop_agent(self, agent_id: str) -> None:
        """
        Stop a specific agent.

        Args:
            agent_id: ID of the agent to stop

        Raises:
            AgentNotFoundError: If agent is not found
            AgentManagerError: If stop fails
        """
        try:
            agent = self.registry.get_agent(agent_id)
            await agent.stop()
            self.logger.info(f"Stopped agent {agent_id}")

        except AgentNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to stop agent {agent_id}: {e}")
            raise AgentManagerError(f"Failed to stop agent {agent_id}", cause=e) from e

    async def restart_agent(self, agent_id: str) -> None:
        """
        Restart a specific agent.

        Args:
            agent_id: ID of the agent to restart
        """
        await self.stop_agent(agent_id)
        await self.start_agent(agent_id)

    async def send_message(self, message: AgentMessage) -> None:
        """
        Send a message to an agent.

        Args:
            message: Message to send

        Raises:
            AgentNotFoundError: If recipient agent is not found
            AgentManagerError: If message sending fails
        """
        try:
            recipient_agent = self.registry.get_agent(message.recipient)

            if recipient_agent.state != AgentState.ACTIVE:
                raise AgentManagerError(
                    f"Recipient agent {message.recipient} is not active",
                    context={"recipient": message.recipient, "state": recipient_agent.state.value}
                )

            await recipient_agent.handle_message(message)
            self.logger.debug(f"Sent message {message.id} to {message.recipient}")

        except AgentNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to send message {message.id}: {e}")
            raise AgentManagerError("Message sending failed", cause=e) from e

    async def broadcast_message(self, message: AgentMessage) -> None:
        """
        Broadcast a message to all active agents.

        Args:
            message: Message to broadcast
        """
        active_agents = self.registry.get_agents_by_state(AgentState.ACTIVE)

        tasks = []
        for agent in active_agents:
            # Create a copy of the message for each recipient
            agent_message = AgentMessage(
                id=message.id,
                sender=message.sender,
                recipient=agent.agent_id,
                message_type=message.message_type,
                payload=message.payload,
                timestamp=message.timestamp
            )
            tasks.append(agent.handle_message(agent_message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self.logger.info(f"Broadcasted message {message.id} to {len(tasks)} agents")

    async def execute_task(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task on a specific agent.

        Args:
            agent_id: ID of the agent to execute task on
            task: Task to execute

        Returns:
            Task execution result

        Raises:
            AgentNotFoundError: If agent is not found
            AgentManagerError: If task execution fails
        """
        try:
            agent = self.registry.get_agent(agent_id)
            result = await agent.execute_task(task)
            self.logger.debug(f"Executed task on agent {agent_id}")
            return result

        except AgentNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to execute task on agent {agent_id}: {e}")
            raise AgentManagerError("Task execution failed", cause=e) from e

    async def health_check_all_agents(self) -> Dict[str, bool]:
        """
        Perform health check on all agents.

        Returns:
            Dictionary mapping agent IDs to health status
        """
        health_status = {}
        agents = self.registry.list_agents()

        tasks = {}
        for agent in agents:
            tasks[agent.agent_id] = asyncio.create_task(agent.health_check())

        for agent_id, task in tasks.items():
            try:
                health_status[agent_id] = await task
            except Exception as e:
                self.logger.error(f"Health check failed for agent {agent_id}: {e}")
                health_status[agent_id] = False

        return health_status

    async def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Agent metrics

        Raises:
            AgentNotFoundError: If agent is not found
        """
        agent = self.registry.get_agent(agent_id)
        return agent.get_metrics()

    async def get_all_agent_metrics(self) -> Dict[str, dict[str, Any]]:
        """
        Get metrics for all agents.

        Returns:
            Dictionary mapping agent IDs to their metrics
        """
        metrics = {}
        agents = self.registry.list_agents()

        for agent in agents:
            try:
                metrics[agent.agent_id] = agent.get_metrics()
            except Exception as e:
                self.logger.error(f"Failed to get metrics for agent {agent.agent_id}: {e}")
                metrics[agent.agent_id] = {"error": str(e)}

        return metrics

    async def start_all_agents(self) -> None:
        """Start all enabled agents."""
        enabled_agents = self.registry.get_enabled_agents()

        tasks = []
        for agent in enabled_agents:
            if agent.state == AgentState.INACTIVE:
                tasks.append(self._safe_start_agent(agent))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = sum(1 for result in results if result is True)
            failure_count = len(results) - success_count

            self.logger.info(f"Started {success_count} agents, {failure_count} failures")

    async def stop_all_agents(self) -> None:
        """Stop all active agents."""
        active_agents = self.registry.get_agents_by_state(AgentState.ACTIVE)

        tasks = []
        for agent in active_agents:
            tasks.append(self._safe_stop_agent(agent))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = sum(1 for result in results if result is True)
            failure_count = len(results) - success_count

            self.logger.info(f"Stopped {success_count} agents, {failure_count} failures")

    def get_manager_status(self) -> Dict[str, Any]:
        """
        Get manager status and statistics.

        Returns:
            Manager status information
        """
        agents = self.registry.list_agents()
        agent_states = {}

        for state in AgentState:
            agent_states[state.value] = len([a for a in agents if a.state == state])

        return {
            "is_running": self.is_running,
            "total_agents": len(agents),
            "agent_states": agent_states,
            "max_agents": self.max_agents,
            "heartbeat_interval": self.heartbeat_interval
        }

    async def _heartbeat_loop(self) -> None:
        """Internal heartbeat monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_heartbeat_check()
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.heartbeat_interval
                )
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(1)

    async def _perform_heartbeat_check(self) -> None:
        """Perform heartbeat check on all agents."""
        health_status = await self.health_check_all_agents()

        for agent_id, is_healthy in health_status.items():
            if not is_healthy:
                self.logger.warning(f"Agent {agent_id} failed health check")
                # Could implement recovery logic here

    async def _safe_start_agent(self, agent: BaseAgent) -> bool:
        """Safely start an agent with error handling."""
        try:
            await agent.start()
            return True
        except Exception as e:
            self.logger.error(f"Failed to start agent {agent.agent_id}: {e}")
            return False

    async def _safe_stop_agent(self, agent: BaseAgent) -> bool:
        """Safely stop an agent with error handling."""
        try:
            await agent.stop()
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop agent {agent.agent_id}: {e}")
            return False
