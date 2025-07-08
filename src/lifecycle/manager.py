"""
Lifecycle manager for coordinating agent lifecycle operations.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional

from ..agents.base import BaseAgent
from ..agents.exceptions import AgentError, AgentStateError
from .hooks import LifecycleHook
from .monitor import LifecycleMonitor


class LifecycleState(Enum):
    """Enumeration of lifecycle states."""
    CREATED = "created"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DESTROYED = "destroyed"


@dataclass
class LifecycleEvent:
    """Lifecycle event data structure."""
    agent_id: str
    event_type: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "agent_id": self.agent_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "data": self.data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LifecycleEvent":
        """Create event from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            event_type=data["event_type"],
            timestamp=data.get("timestamp", time.time()),
            data=data.get("data", {})
        )


class LifecycleManager:
    """
    Manager for agent lifecycle operations.

    Provides comprehensive lifecycle management including hooks,
    monitoring, and state transitions for all agents in the system.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        monitor: Optional[LifecycleMonitor] = None
    ):
        """
        Initialize the lifecycle manager.

        Args:
            config: Lifecycle configuration
            logger: Logger instance
            monitor: Optional lifecycle monitor
        """
        self.config = config.get("lifecycle", {})
        self.logger = logger
        self.monitor = monitor or LifecycleMonitor(logger)

        # Lifecycle hooks by event type
        self.hooks: Dict[str, List[LifecycleHook]] = defaultdict(list)

        # Agent state tracking
        self.agent_states: Dict[str, LifecycleState] = {}
        self.registered_agents: Dict[str, BaseAgent] = {}

        # Event history
        self.event_history: List[LifecycleEvent] = []
        self.max_event_history = self.config.get("max_event_history", 1000)

        # Configuration
        self.graceful_shutdown_timeout = self.config.get("graceful_shutdown_timeout", 10.0)
        self.health_check_interval = self.config.get("health_check_interval", 30.0)
        self.performance_monitor_interval = self.config.get("performance_monitor_interval", 60.0)

    def add_hook(self, event_type: str, hook: LifecycleHook) -> None:
        """
        Add a lifecycle hook for a specific event type.

        Args:
            event_type: Type of lifecycle event
            hook: Hook to execute
        """
        self.hooks[event_type].append(hook)
        self.logger.debug(f"Added {type(hook).__name__} hook for {event_type}")

    def remove_hook(self, event_type: str, hook: LifecycleHook) -> None:
        """
        Remove a lifecycle hook.

        Args:
            event_type: Type of lifecycle event
            hook: Hook to remove
        """
        if event_type in self.hooks:
            try:
                self.hooks[event_type].remove(hook)
                self.logger.debug(f"Removed {type(hook).__name__} hook for {event_type}")
            except ValueError:
                pass  # Hook not found

    async def start_agent(self, agent: BaseAgent) -> None:
        """
        Start an agent with full lifecycle management.

        Args:
            agent: Agent to start

        Raises:
            AgentError: If startup fails
            AgentStateError: If agent is in invalid state
        """
        agent_id = agent.agent_id

        try:
            # Check current state
            current_state = self.agent_states.get(agent_id, LifecycleState.CREATED)
            if current_state not in [LifecycleState.CREATED, LifecycleState.STOPPED]:
                raise AgentStateError(
                    f"Cannot start agent {agent_id} from state {current_state.value}",
                    context={"agent_id": agent_id, "current_state": current_state.value}
                )

            self.logger.info(f"Starting lifecycle for agent {agent_id}")

            # Update state
            self.agent_states[agent_id] = LifecycleState.STARTING
            self.registered_agents[agent_id] = agent

            # Log event
            self.log_event(agent_id, "lifecycle_start_initiated")

            # Execute pre-start hooks
            await self._execute_hooks("pre_start", agent_id, {"agent": agent})

            # Start the agent
            await agent.start()

            # Update state
            self.agent_states[agent_id] = LifecycleState.RUNNING

            # Execute post-start hooks
            await self._execute_hooks("post_start", agent_id, {"agent": agent})

            # Start monitoring
            await self.monitor.start_monitoring(agent)

            # Log event
            self.log_event(agent_id, "lifecycle_start_completed")

            self.logger.info(f"Successfully started lifecycle for agent {agent_id}")

        except Exception as e:
            # Update state to error
            self.agent_states[agent_id] = LifecycleState.ERROR

            # Log event
            self.log_event(agent_id, "lifecycle_start_failed", {"error": str(e)})

            self.logger.error(f"Failed to start lifecycle for agent {agent_id}: {e}")
            raise AgentError(f"Failed to start agent {agent_id}", cause=e) from e

    async def stop_agent(self, agent_id: str) -> None:
        """
        Stop an agent with full lifecycle management.

        Args:
            agent_id: ID of agent to stop

        Raises:
            AgentError: If shutdown fails
            AgentStateError: If agent is in invalid state
        """
        try:
            # Check current state
            current_state = self.agent_states.get(agent_id, LifecycleState.CREATED)
            if current_state != LifecycleState.RUNNING:
                raise AgentStateError(
                    f"Cannot stop agent {agent_id} from state {current_state.value}",
                    context={"agent_id": agent_id, "current_state": current_state.value}
                )

            agent = self.registered_agents.get(agent_id)
            if not agent:
                raise AgentError(f"Agent {agent_id} not found in registry")

            self.logger.info(f"Stopping lifecycle for agent {agent_id}")

            # Update state
            self.agent_states[agent_id] = LifecycleState.STOPPING

            # Log event
            self.log_event(agent_id, "lifecycle_stop_initiated")

            # Execute pre-stop hooks
            await self._execute_hooks("pre_stop", agent_id, {"agent": agent})

            # Stop monitoring
            await self.monitor.stop_monitoring(agent_id)

            # Stop the agent
            await agent.stop()

            # Update state
            self.agent_states[agent_id] = LifecycleState.STOPPED

            # Execute post-stop hooks
            await self._execute_hooks("post_stop", agent_id, {"agent": agent})

            # Log event
            self.log_event(agent_id, "lifecycle_stop_completed")

            self.logger.info(f"Successfully stopped lifecycle for agent {agent_id}")

        except Exception as e:
            # Update state to error
            self.agent_states[agent_id] = LifecycleState.ERROR

            # Log event
            self.log_event(agent_id, "lifecycle_stop_failed", {"error": str(e)})

            self.logger.error(f"Failed to stop lifecycle for agent {agent_id}: {e}")
            raise AgentError(f"Failed to stop agent {agent_id}", cause=e) from e

    async def restart_agent(self, agent_id: str) -> None:
        """
        Restart an agent.

        Args:
            agent_id: ID of agent to restart
        """
        self.logger.info(f"Restarting agent {agent_id}")

        # Log event
        self.log_event(agent_id, "lifecycle_restart_initiated")

        # Stop and start the agent
        await self.stop_agent(agent_id)

        agent = self.registered_agents.get(agent_id)
        if agent:
            await self.start_agent(agent)

        # Log event
        self.log_event(agent_id, "lifecycle_restart_completed")

    def get_agent_state(self, agent_id: str) -> Optional[LifecycleState]:
        """
        Get the current lifecycle state of an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Current lifecycle state or None if agent not found
        """
        return self.agent_states.get(agent_id)

    def get_all_agent_states(self) -> Dict[str, LifecycleState]:
        """
        Get the lifecycle states of all agents.

        Returns:
            Dictionary mapping agent IDs to their lifecycle states
        """
        return dict(self.agent_states)

    async def graceful_shutdown_all(self) -> None:
        """Gracefully shutdown all running agents."""
        running_agents = [
            agent_id for agent_id, state in self.agent_states.items()
            if state == LifecycleState.RUNNING
        ]

        if not running_agents:
            self.logger.info("No running agents to shutdown")
            return

        self.logger.info(f"Gracefully shutting down {len(running_agents)} agents")

        # Create shutdown tasks for all agents
        shutdown_tasks = []
        for agent_id in running_agents:
            task = asyncio.create_task(self._safe_stop_agent(agent_id))
            shutdown_tasks.append(task)

        # Wait for all shutdowns with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*shutdown_tasks, return_exceptions=True),
                timeout=self.graceful_shutdown_timeout
            )
        except TimeoutError:
            self.logger.warning("Graceful shutdown timed out, some agents may not have stopped cleanly")

        self.logger.info("Graceful shutdown completed")

    async def emergency_shutdown(self, timeout: float = 5.0) -> None:
        """
        Emergency shutdown of all agents.

        Args:
            timeout: Maximum time to wait for shutdowns
        """
        self.logger.warning("Initiating emergency shutdown")

        # Get all agents that need to be stopped
        agents_to_stop = [
            agent_id for agent_id, state in self.agent_states.items()
            if state in [LifecycleState.RUNNING, LifecycleState.STARTING]
        ]

        # Create emergency stop tasks
        stop_tasks = []
        for agent_id in agents_to_stop:
            task = asyncio.create_task(self._emergency_stop_agent(agent_id))
            stop_tasks.append(task)

        # Wait with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*stop_tasks, return_exceptions=True),
                timeout=timeout
            )
        except TimeoutError:
            self.logger.error("Emergency shutdown timed out")

        # Force all agents to stopped state
        for agent_id in agents_to_stop:
            self.agent_states[agent_id] = LifecycleState.STOPPED

        self.logger.warning("Emergency shutdown completed")

    def log_event(self, agent_id: str, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a lifecycle event.

        Args:
            agent_id: ID of the agent
            event_type: Type of event
            data: Optional event data
        """
        event = LifecycleEvent(
            agent_id=agent_id,
            event_type=event_type,
            data=data or {}
        )

        # Add to history
        self.event_history.append(event)

        # Trim history if needed
        if len(self.event_history) > self.max_event_history:
            self.event_history = self.event_history[-self.max_event_history:]

        # Log the event
        self.logger.info(f"Lifecycle event: {event_type} for agent {agent_id}", extra={
            "agent_id": agent_id,
            "event_type": event_type,
            "event_data": data
        })

    def get_event_history(
        self,
        agent_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[LifecycleEvent]:
        """
        Get filtered event history.

        Args:
            agent_id: Optional agent ID filter
            event_type: Optional event type filter
            limit: Optional limit on number of events

        Returns:
            List of matching events
        """
        events = self.event_history

        # Apply filters
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Apply limit
        if limit:
            events = events[-limit:]

        return events

    async def _execute_hooks(self, event_type: str, agent_id: str, context: Dict[str, Any]) -> None:
        """Execute all hooks for a specific event type."""
        hooks = self.hooks.get(event_type, [])

        for hook in hooks:
            try:
                self.logger.debug(f"Executing {type(hook).__name__} hook for agent {agent_id}")

                success = await hook.execute(agent_id, context)

                if not success:
                    raise AgentError(f"Hook {type(hook).__name__} failed for agent {agent_id}")

            except Exception as e:
                self.logger.error(f"Hook {type(hook).__name__} failed for agent {agent_id}: {e}")
                raise

    async def _safe_stop_agent(self, agent_id: str) -> bool:
        """Safely stop an agent with error handling."""
        try:
            await self.stop_agent(agent_id)
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop agent {agent_id}: {e}")
            return False

    async def _emergency_stop_agent(self, agent_id: str) -> None:
        """Emergency stop an agent."""
        try:
            agent = self.registered_agents.get(agent_id)
            if agent:
                # Skip hooks and monitoring for emergency stop
                await agent.stop()

            self.agent_states[agent_id] = LifecycleState.STOPPED

        except Exception as e:
            self.logger.error(f"Emergency stop failed for agent {agent_id}: {e}")
            self.agent_states[agent_id] = LifecycleState.ERROR

    def get_manager_status(self) -> Dict[str, Any]:
        """
        Get lifecycle manager status.

        Returns:
            Dictionary containing manager status
        """
        state_counts = {}
        for state in LifecycleState:
            state_counts[state.value] = sum(
                1 for s in self.agent_states.values() if s == state
            )

        return {
            "total_agents": len(self.agent_states),
            "state_distribution": state_counts,
            "event_history_size": len(self.event_history),
            "registered_hooks": {
                event_type: len(hooks) for event_type, hooks in self.hooks.items()
            },
            "monitor_status": self.monitor.get_all_agent_status() if self.monitor else {}
        }
