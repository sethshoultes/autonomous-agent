"""
Message broker implementation for agent communication.
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable
import contextlib
import heapq
import logging
import time
from typing import Any, Dict, List, Optional

from ..agents.base import AgentMessage
from ..agents.exceptions import CommunicationError


class MessageQueue:
    """
    Asynchronous message queue with priority support.

    Provides thread-safe message queuing with optional priority ordering.
    """

    def __init__(self, max_size: int = 1000, priority_queue: bool = False) -> None:
        """
        Initialize the message queue.

        Args:
            max_size: Maximum number of messages in queue
            priority_queue: Whether to use priority ordering
        """
        self.max_size = max_size
        self.priority_queue = priority_queue
        self._queue: list = []
        self._condition = asyncio.Condition()
        self._closed = False

    @property
    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        return len(self._queue) >= self.max_size

    async def put(self, message: AgentMessage, timeout: Optional[float] = None) -> None:
        """
        Put a message in the queue.

        Args:
            message: Message to add
            timeout: Optional timeout in seconds

        Raises:
            CommunicationError: If queue is full or timeout occurs
        """
        async with self._condition:
            if self._closed:
                raise CommunicationError("Queue is closed")

            # Wait for space if queue is full
            if timeout is not None:
                deadline = time.time() + timeout

            while self.is_full:
                if self._closed:
                    raise CommunicationError("Queue is closed")

                if timeout is not None:
                    remaining_time = deadline - time.time()
                    if remaining_time <= 0:
                        raise CommunicationError("Put operation timed out")

                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=remaining_time)
                    except TimeoutError:
                        raise CommunicationError("Put operation timed out")
                else:
                    await self._condition.wait()

            # Add message to queue
            if self.priority_queue:
                # Use priority (lower number = higher priority)
                priority = getattr(message, 'priority', 5)
                heapq.heappush(self._queue, (priority, time.time(), message))
            else:
                self._queue.append(message)

            # Notify waiters
            self._condition.notify()

    async def get(self, timeout: Optional[float] = None) -> AgentMessage:
        """
        Get a message from the queue.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Message from queue

        Raises:
            CommunicationError: If queue is empty or timeout occurs
        """
        async with self._condition:
            # Wait for message if queue is empty
            if timeout is not None:
                deadline = time.time() + timeout

            while self.is_empty:
                if self._closed:
                    raise CommunicationError("Queue is closed")

                if timeout is not None:
                    remaining_time = deadline - time.time()
                    if remaining_time <= 0:
                        raise CommunicationError("Get operation timed out")

                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=remaining_time)
                    except TimeoutError:
                        raise CommunicationError("Get operation timed out")
                else:
                    await self._condition.wait()

            # Get message from queue
            if self.priority_queue:
                _, _, message = heapq.heappop(self._queue)
            else:
                message = self._queue.pop(0)

            # Notify waiters
            self._condition.notify()

            return message  # type: ignore[return-value]

    async def close(self) -> None:
        """Close the queue."""
        async with self._condition:
            self._closed = True
            self._condition.notify_all()


class MessageHandler:
    """
    Handler for processing messages by type.

    Provides a subscription-based message handling system.
    """

    def __init__(self) -> None:
        """Initialize the message handler."""
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, message_type: str, callback: Callable) -> None:
        """
        Subscribe to a message type.

        Args:
            message_type: Type of message to subscribe to
            callback: Callback function to handle messages
        """
        self.subscribers[message_type].append(callback)

    def unsubscribe(self, message_type: str, callback: Callable) -> None:
        """
        Unsubscribe from a message type.

        Args:
            message_type: Type of message to unsubscribe from
            callback: Callback function to remove
        """
        if message_type in self.subscribers:
            try:
                self.subscribers[message_type].remove(callback)
                if not self.subscribers[message_type]:
                    del self.subscribers[message_type]
            except ValueError:
                pass  # Callback not found

    async def handle_message(self, message: AgentMessage) -> List[Any]:
        """
        Handle a message by calling all subscribers.

        Args:
            message: Message to handle

        Returns:
            List of results from all callbacks
        """
        results = []
        callbacks = self.subscribers.get(message.message_type, [])

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(message)
                else:
                    result = callback(message)
                results.append(result)
            except Exception:
                # Log error but don't stop processing other callbacks
                # In a real implementation, this would use proper logging
                pass

        return results


class MessageRouter:
    """
    Router for directing messages to appropriate handlers.

    Provides flexible message routing based on recipient or message type.
    """

    def __init__(self) -> None:
        """Initialize the message router."""
        self.routes: Dict[str, Callable] = {}
        self.default_route: Optional[Callable] = None

    def add_route(self, recipient: str, handler: Callable) -> None:
        """
        Add a route for a specific recipient.

        Args:
            recipient: Recipient identifier
            handler: Handler function for messages to this recipient
        """
        self.routes[recipient] = handler

    def remove_route(self, recipient: str) -> None:
        """
        Remove a route for a specific recipient.

        Args:
            recipient: Recipient identifier to remove
        """
        self.routes.pop(recipient, None)

    def set_default_route(self, handler: Callable) -> None:
        """
        Set the default route for unmatched recipients.

        Args:
            handler: Default handler function
        """
        self.default_route = handler

    async def route_message(self, message: AgentMessage) -> Any:
        """
        Route a message to the appropriate handler.

        Args:
            message: Message to route

        Returns:
            Result from the handler

        Raises:
            MessageRoutingError: If no route is found
        """
        from ..agents.exceptions import MessageRoutingError

        handler = self.routes.get(message.recipient)

        if handler is None:
            handler = self.default_route

        if handler is None:
            raise MessageRoutingError(
                f"No route found for recipient {message.recipient}",
                context={"recipient": message.recipient, "message_id": message.id}
            )

        if asyncio.iscoroutinefunction(handler):
            return await handler(message)
        else:
            return handler(message)

    async def route_broadcast_message(self, message: AgentMessage) -> List[Any]:
        """
        Route a broadcast message to all registered handlers.

        Args:
            message: Broadcast message to route

        Returns:
            List of results from all handlers
        """
        results = []

        for recipient, handler in self.routes.items():
            try:
                # Create a copy of the message for each recipient
                recipient_message = AgentMessage(
                    id=message.id,
                    sender=message.sender,
                    recipient=recipient,
                    message_type=message.message_type,
                    payload=message.payload,
                    timestamp=message.timestamp
                )

                if asyncio.iscoroutinefunction(handler):
                    result = await handler(recipient_message)
                else:
                    result = handler(recipient_message)
                results.append(result)
            except Exception:
                # Log error but continue with other handlers
                pass

        return results


class MessageBroker:
    """
    Message broker for coordinating agent communication.

    Provides asynchronous message publishing, subscription, and routing
    between agents in the system.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the message broker.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.message_queue = MessageQueue(max_size=10000, priority_queue=True)
        self.message_handler = MessageHandler()
        self.message_router = MessageRouter()
        self.is_running = False
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the message broker."""
        if self.is_running:
            self.logger.warning("Message broker is already running")
            return

        self.logger.info("Starting message broker")
        self.is_running = True

        # Start message processing task
        self._processing_task = asyncio.create_task(self._message_processing_loop())

        self.logger.info("Message broker started successfully")

    async def stop(self) -> None:
        """Stop the message broker."""
        if not self.is_running:
            self.logger.warning("Message broker is not running")
            return

        self.logger.info("Stopping message broker")
        self.is_running = False

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel processing task
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task

        # Close message queue
        await self.message_queue.close()

        self.logger.info("Message broker stopped successfully")

    async def publish(self, message: AgentMessage) -> None:
        """
        Publish a message to the broker.

        Args:
            message: Message to publish

        Raises:
            CommunicationError: If broker is not running or publish fails
        """
        if not self.is_running:
            raise CommunicationError("Message broker is not running")

        try:
            await self.message_queue.put(message)
            self.logger.debug(f"Published message {message.id} from {message.sender} to {message.recipient}")
        except Exception as e:
            self.logger.error(f"Failed to publish message {message.id}: {e}")
            raise CommunicationError("Failed to publish message", cause=e)

    async def subscribe(self, message_type: str, callback: Callable) -> None:
        """
        Subscribe to messages of a specific type.

        Args:
            message_type: Type of message to subscribe to
            callback: Callback function to handle messages
        """
        self.message_handler.subscribe(message_type, callback)
        self.logger.debug(f"Subscribed to message type: {message_type}")

    def add_route(self, recipient: str, handler: Callable) -> None:
        """
        Add a route for a specific recipient.

        Args:
            recipient: Recipient identifier
            handler: Handler function for messages to this recipient
        """
        self.message_router.add_route(recipient, handler)
        self.logger.debug(f"Added route for recipient: {recipient}")

    def remove_route(self, recipient: str) -> None:
        """
        Remove a route for a specific recipient.

        Args:
            recipient: Recipient identifier to remove
        """
        self.message_router.remove_route(recipient)
        self.logger.debug(f"Removed route for recipient: {recipient}")

    async def disconnect(self) -> None:
        """Disconnect from the message broker."""
        await self.stop()

    async def _message_processing_loop(self) -> None:
        """Internal message processing loop."""
        while not self._shutdown_event.is_set():
            try:
                # Get message from queue with timeout
                try:
                    message = await self.message_queue.get(timeout=1.0)
                except CommunicationError:
                    continue  # Timeout or queue closed

                # Process the message
                await self._process_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(0.1)  # Brief pause before retrying

    async def _process_message(self, message: AgentMessage) -> None:
        """
        Process a single message.

        Args:
            message: Message to process
        """
        try:
            # Handle broadcast messages
            if message.recipient == "broadcast":
                await self.message_router.route_broadcast_message(message)
            else:
                # Route to specific recipient
                await self.message_router.route_message(message)

            # Handle by message type subscribers
            await self.message_handler.handle_message(message)

            self.logger.debug(f"Processed message {message.id}")

        except Exception as e:
            self.logger.error(f"Error processing message {message.id}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get broker status information.

        Returns:
            Dictionary containing broker status
        """
        return {
            "is_running": self.is_running,
            "queue_size": self.message_queue.size,
            "queue_max_size": self.message_queue.max_size,
            "subscribers": len(self.message_handler.subscribers),
            "routes": len(self.message_router.routes)
        }
