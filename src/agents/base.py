"""
Base classes and interfaces for the autonomous agent system.
"""

from abc import ABC, abstractmethod
import asyncio
import contextlib
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional, Union
import uuid

from .exceptions import AgentCommunicationError, AgentError, AgentStateError


class AgentState(Enum):
    """Enumeration of possible agent states."""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    id: str
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 5  # Lower number = higher priority

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "priority": self.priority
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        return cls(
            id=data["id"],
            sender=data["sender"],
            recipient=data["recipient"],
            message_type=data["message_type"],
            payload=data["payload"],
            timestamp=data.get("timestamp", time.time()),
            priority=data.get("priority", 5)
        )


class AgentInterface(ABC):
    """Abstract interface defining the contract for all agents."""

    @abstractmethod
    async def start(self) -> None:
        """Start the agent."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the agent."""
        pass

    @abstractmethod
    async def send_message(self, recipient: str, message_type: str, payload: Dict[str, Any]) -> None:
        """Send a message to another agent."""
        pass

    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> None:
        """Handle an incoming message."""
        pass

    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task and return the result."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Perform a health check and return status."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        pass


class BaseAgent(AgentInterface):
    """
    Abstract base class for all agents in the system.

    Provides common functionality and enforces the agent interface.
    Follows SOLID principles with dependency injection and clean interfaces.
    """

    def __init__(
        self,
        agent_id: str,
        config: Dict[str, Any],
        logger: logging.Logger,
        message_broker: Any  # Avoid circular import
    ):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent
            config: Configuration dictionary
            logger: Logger instance for the agent
            message_broker: Message broker for communication
        """
        self.agent_id = agent_id
        self.config = config
        self.logger = logger
        self.message_broker = message_broker
        self.state = AgentState.INACTIVE
        self.start_time: Optional[float] = None

        # Initialize metrics
        self.metrics = {
            "messages_processed": 0,
            "tasks_completed": 0,
            "errors": 0,
            "uptime": 0
        }

        # Task queue for async processing
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Performance optimizations
        self._message_cache: Dict[str, Any] = {}
        self._cache_max_size = 100
        self._cache_ttl = 300  # 5 minutes

    async def start(self) -> None:
        """
        Start the agent.

        Raises:
            AgentStateError: If agent is not in INACTIVE state
        """
        if self.state != AgentState.INACTIVE:
            raise AgentStateError(
                f"Cannot start agent {self.agent_id} from state {self.state.value}",
                context={"agent_id": self.agent_id, "current_state": self.state.value}
            )

        try:
            self.state = AgentState.STARTING
            self.start_time = time.time()

            self.logger.info(f"Starting agent {self.agent_id}")

            # Start message processing task
            self._processing_task = asyncio.create_task(self._message_processing_loop())

            # Perform agent-specific initialization
            await self._initialize()

            self.state = AgentState.ACTIVE
            self.logger.info(f"Agent {self.agent_id} started successfully")

        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Failed to start agent {self.agent_id}: {e}")
            raise AgentError(f"Failed to start agent {self.agent_id}", cause=e) from e

    async def stop(self) -> None:
        """
        Stop the agent.

        Raises:
            AgentStateError: If agent is not in ACTIVE state
        """
        if self.state not in [AgentState.ACTIVE, AgentState.ERROR]:
            raise AgentStateError(
                f"Cannot stop agent {self.agent_id} from state {self.state.value}",
                context={"agent_id": self.agent_id, "current_state": self.state.value}
            )

        try:
            self.state = AgentState.STOPPING
            self.logger.info(f"Stopping agent {self.agent_id}")

            # Signal shutdown
            self._shutdown_event.set()

            # Cancel processing task with proper cleanup
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
                try:
                    await asyncio.wait_for(self._processing_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass  # Expected when cancelling
                except Exception as e:
                    self.logger.warning(f"Error during task cleanup: {e}")

            # Clear task queue to prevent memory leaks
            while not self._task_queue.empty():
                try:
                    self._task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Clear caches to prevent memory leaks
            self._message_cache.clear()

            # Perform agent-specific cleanup
            await self._cleanup()

            # Disconnect from message broker with timeout
            if self.message_broker:
                try:
                    await asyncio.wait_for(self.message_broker.disconnect(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Message broker disconnect timed out")
                except Exception as e:
                    self.logger.warning(f"Error disconnecting message broker: {e}")
                finally:
                    self.message_broker = None

            self.state = AgentState.INACTIVE
            self.logger.info(f"Agent {self.agent_id} stopped successfully")

        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Failed to stop agent {self.agent_id}: {e}")
            raise AgentError(f"Failed to stop agent {self.agent_id}", cause=e) from e

    async def send_message(self, recipient: str, message_type: str, payload: Dict[str, Any]) -> None:
        """
        Send a message to another agent.

        Args:
            recipient: ID of the recipient agent
            message_type: Type of the message
            payload: Message payload
        """
        if self.state != AgentState.ACTIVE:
            raise AgentStateError(f"Agent {self.agent_id} is not active")

        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            payload=payload
        )

        try:
            await self.message_broker.publish(message)
            self.logger.debug(f"Sent message {message.id} to {recipient}")
        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Failed to send message to {recipient}: {e}")
            raise AgentCommunicationError(f"Failed to send message to {recipient}", cause=e) from e

    async def handle_message(self, message: AgentMessage) -> None:
        """
        Handle an incoming message.

        Args:
            message: Incoming message to handle
        """
        try:
            self.logger.debug(f"Handling message {message.id} from {message.sender}")

            # Process the message using agent-specific logic
            response = await self._process_message(message)

            # Send response if one is generated
            if response:
                await self.message_broker.publish(response)

            self.metrics["messages_processed"] += 1

        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Error handling message {message.id}: {e}")
            # Don't re-raise to avoid breaking the message processing loop

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task and return the result.

        Args:
            task: Task to execute

        Returns:
            Task execution result
        """
        if self.state != AgentState.ACTIVE:
            raise AgentStateError(f"Agent {self.agent_id} is not active")

        try:
            self.logger.debug(f"Executing task for agent {self.agent_id}")

            # Execute agent-specific task logic
            result = await self._execute_task(task)

            self.metrics["tasks_completed"] += 1
            return result

        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Error executing task: {e}")
            raise AgentError("Task execution failed", cause=e) from e

    async def health_check(self) -> bool:
        """
        Perform a health check and return status.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Perform agent-specific health check
            return await self._health_check()
        except Exception as e:
            self.logger.error(f"Health check failed for agent {self.agent_id}: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent metrics.

        Returns:
            Dictionary containing agent metrics
        """
        current_time = time.time()
        uptime = current_time - self.start_time if self.start_time else 0

        return {
            **self.metrics,
            "uptime": uptime,
            "state": self.state.value,
            "agent_id": self.agent_id
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.stop()
        return False  # Don't suppress exceptions

    async def _message_processing_loop(self) -> None:
        """Internal message processing loop."""
        while not self._shutdown_event.is_set():
            try:
                # Get message from queue with timeout
                try:
                    # Simple implementation to avoid task leakage in tests
                    # Check shutdown first
                    if self._shutdown_event.is_set():
                        break
                    
                    # Try to get a message with a short timeout
                    try:
                        message = await asyncio.wait_for(self._task_queue.get(), timeout=0.1)
                        await self.handle_message(message)
                    except asyncio.TimeoutError:
                        continue  # No message available, check shutdown again
                    
                except asyncio.TimeoutError:
                    continue  # Just check the shutdown event again

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying

    # Abstract methods that must be implemented by concrete agents

    @abstractmethod
    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message.

        Args:
            message: Message to process

        Returns:
            Optional response message
        """
        pass

    @abstractmethod
    async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task.

        Args:
            task: Task to execute

        Returns:
            Task result
        """
        pass

    @abstractmethod
    async def _health_check(self) -> bool:
        """
        Perform agent-specific health check.

        Returns:
            True if healthy, False otherwise
        """
        pass

    # Optional hook methods for agent lifecycle

    async def _initialize(self) -> None:
        """Initialize agent-specific resources. Override if needed."""
        pass

    async def _cleanup(self) -> None:
        """Cleanup agent-specific resources. Override if needed."""
        pass

    def _cache_get(self, key: str) -> Any:
        """Get value from cache with TTL check."""
        if key in self._message_cache:
            value, timestamp = self._message_cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return value
            else:
                del self._message_cache[key]
        return None

    def _cache_set(self, key: str, value: Any) -> None:
        """Set value in cache with TTL and size management."""
        # Cleanup old entries if cache is full
        if len(self._message_cache) >= self._cache_max_size:
            # Remove oldest entries
            oldest_keys = sorted(
                self._message_cache.keys(),
                key=lambda k: self._message_cache[k][1]
            )[:10]  # Remove 10 oldest entries
            for old_key in oldest_keys:
                del self._message_cache[old_key]
        
        self._message_cache[key] = (value, time.time())
