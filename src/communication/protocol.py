"""
Communication protocol implementation for message encoding, validation, and routing.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Union

from ..agents.base import AgentMessage
from ..agents.exceptions import CommunicationError, MessageValidationError


class MessageEncoder:
    """
    Encoder for serializing AgentMessage objects.

    Provides JSON-based message serialization with proper error handling.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        """
        Initialize the message encoder.

        Args:
            encoding: Character encoding to use
        """
        self.encoding = encoding

    def encode(self, message: AgentMessage) -> bytes:
        """
        Encode a message to bytes.

        Args:
            message: Message to encode

        Returns:
            Encoded message as bytes

        Raises:
            MessageValidationError: If message is invalid
        """
        if not isinstance(message, AgentMessage):
            raise MessageValidationError(
                f"Expected AgentMessage, got {type(message).__name__}",
                context={"message_type": type(message).__name__}
            )

        try:
            message_dict = message.to_dict()
            json_str = json.dumps(message_dict, ensure_ascii=False)
            return json_str.encode(self.encoding)
        except (TypeError, ValueError) as e:
            raise MessageValidationError(f"Failed to encode message: {e}", cause=e)

    def encode_dict(self, data: Dict[str, Any]) -> bytes:
        """
        Encode a dictionary to bytes.

        Args:
            data: Dictionary to encode

        Returns:
            Encoded data as bytes

        Raises:
            MessageValidationError: If encoding fails
        """
        try:
            json_str = json.dumps(data, ensure_ascii=False)
            return json_str.encode(self.encoding)
        except (TypeError, ValueError) as e:
            raise MessageValidationError(f"Failed to encode dictionary: {e}", cause=e)


class MessageDecoder:
    """
    Decoder for deserializing AgentMessage objects.

    Provides JSON-based message deserialization with validation.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        """
        Initialize the message decoder.

        Args:
            encoding: Character encoding to use
        """
        self.encoding = encoding

    def decode(self, data: bytes) -> AgentMessage:
        """
        Decode bytes to a message.

        Args:
            data: Encoded message data

        Returns:
            Decoded AgentMessage

        Raises:
            MessageValidationError: If decoding fails or data is invalid
        """
        try:
            json_str = data.decode(self.encoding)
            message_dict = json.loads(json_str)
            return self._dict_to_message(message_dict)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise MessageValidationError(f"Failed to decode message: {e}", cause=e)

    def decode_dict(self, data: bytes) -> Dict[str, Any]:
        """
        Decode bytes to a dictionary.

        Args:
            data: Encoded data

        Returns:
            Decoded dictionary

        Raises:
            MessageValidationError: If decoding fails
        """
        try:
            json_str = data.decode(self.encoding)
            return json.loads(json_str)  # type: ignore[return-value]
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise MessageValidationError(f"Failed to decode dictionary: {e}", cause=e)

    def _dict_to_message(self, data: Dict[str, Any]) -> AgentMessage:
        """
        Convert dictionary to AgentMessage.

        Args:
            data: Message data dictionary

        Returns:
            AgentMessage instance

        Raises:
            MessageValidationError: If required fields are missing
        """
        required_fields = ["id", "sender", "recipient", "message_type", "payload"]

        for field in required_fields:
            if field not in data:
                raise MessageValidationError(
                    f"Missing required field: {field}",
                    context={"missing_field": field, "available_fields": list(data.keys())}
                )

        try:
            return AgentMessage.from_dict(data)
        except Exception as e:
            raise MessageValidationError(f"Failed to create message from dictionary: {e}", cause=e)


class MessageValidator:
    """
    Validator for AgentMessage objects.

    Provides comprehensive validation of message structure and content.
    """

    def __init__(self, max_message_size: int = 1024 * 1024) -> None:  # 1MB default
        """
        Initialize the message validator.

        Args:
            max_message_size: Maximum allowed message size in bytes
        """
        self.max_message_size = max_message_size

    def validate(self, message: AgentMessage, max_size: Optional[int] = None) -> None:
        """
        Validate a message.

        Args:
            message: Message to validate
            max_size: Optional maximum size override

        Raises:
            MessageValidationError: If validation fails
        """
        if message is None:
            raise MessageValidationError("Message cannot be None")

        if not isinstance(message, AgentMessage):
            raise MessageValidationError(
                f"Expected AgentMessage, got {type(message).__name__}",
                context={"message_type": type(message).__name__}
            )

        # Validate required fields
        self._validate_required_fields(message)

        # Validate field types and values
        self._validate_field_types(message)

        # Validate message size
        size_limit = max_size or self.max_message_size
        self._validate_message_size(message, size_limit)

    def _validate_required_fields(self, message: AgentMessage) -> None:
        """Validate that all required fields are present and non-empty."""
        if not message.id or not message.id.strip():
            raise MessageValidationError("Message ID cannot be empty")

        if not message.sender or not message.sender.strip():
            raise MessageValidationError("Message sender cannot be empty")

        if not message.recipient or not message.recipient.strip():
            raise MessageValidationError("Message recipient cannot be empty")

        if not message.message_type or not message.message_type.strip():
            raise MessageValidationError("Message type cannot be empty")

        if message.payload is None:
            raise MessageValidationError("Message payload cannot be None")

    def _validate_field_types(self, message: AgentMessage) -> None:
        """Validate field types."""
        if not isinstance(message.id, str):
            raise MessageValidationError("Message ID must be a string")

        if not isinstance(message.sender, str):
            raise MessageValidationError("Message sender must be a string")

        if not isinstance(message.recipient, str):
            raise MessageValidationError("Message recipient must be a string")

        if not isinstance(message.message_type, str):
            raise MessageValidationError("Message type must be a string")

        if not isinstance(message.payload, dict):
            raise MessageValidationError("Message payload must be a dictionary")

        if not isinstance(message.timestamp, (int, float)):
            raise MessageValidationError("Message timestamp must be a number")

        if hasattr(message, 'priority') and not isinstance(message.priority, int):
            raise MessageValidationError("Message priority must be an integer")

    def _validate_message_size(self, message: AgentMessage, max_size: int) -> None:
        """Validate message size."""
        try:
            encoder = MessageEncoder()
            encoded_message = encoder.encode(message)
            message_size = len(encoded_message)

            if message_size > max_size:
                raise MessageValidationError(
                    f"Message size ({message_size} bytes) exceeds maximum ({max_size} bytes)",
                    context={"message_size": message_size, "max_size": max_size}
                )
        except MessageValidationError:
            raise
        except Exception as e:
            raise MessageValidationError(f"Failed to validate message size: {e}", cause=e)


class CommunicationProtocol:
    """
    High-level communication protocol for agent message handling.

    Provides encoding, decoding, validation, and retry logic for
    reliable inter-agent communication.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the communication protocol.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.encoder = MessageEncoder()
        self.decoder = MessageDecoder()
        self.validator = MessageValidator()

    def encode_message(self, message: AgentMessage) -> bytes:
        """
        Encode a message with validation.

        Args:
            message: Message to encode

        Returns:
            Encoded message bytes

        Raises:
            MessageValidationError: If message is invalid
        """
        # Validate before encoding
        self.validator.validate(message)

        # Encode the message
        return self.encoder.encode(message)

    def decode_message(self, data: bytes) -> AgentMessage:
        """
        Decode and validate a message.

        Args:
            data: Encoded message data

        Returns:
            Decoded and validated AgentMessage

        Raises:
            MessageValidationError: If decoding or validation fails
        """
        # Decode the message
        message = self.decoder.decode(data)

        # Validate after decoding
        self.validator.validate(message)

        return message

    def validate_message(self, message: AgentMessage) -> None:
        """
        Validate a message.

        Args:
            message: Message to validate

        Raises:
            MessageValidationError: If validation fails
        """
        self.validator.validate(message)

    async def send_message_with_retry(
        self,
        message: AgentMessage,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0
    ) -> bool:
        """
        Send a message with retry logic.

        Args:
            message: Message to send
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Factor to multiply delay by after each retry

        Returns:
            True if message was sent successfully

        Raises:
            CommunicationError: If all retry attempts fail
        """
        last_exception = None
        delay = retry_delay

        for attempt in range(max_retries + 1):
            try:
                # Validate message before sending
                self.validator.validate(message)

                # Attempt to send (this would be implemented by subclasses)
                await self._send(message)

                self.logger.debug(f"Message {message.id} sent successfully on attempt {attempt + 1}")
                return True

            except Exception as e:
                last_exception = e
                self.logger.warning(f"Send attempt {attempt + 1} failed for message {message.id}: {e}")

                if attempt < max_retries:
                    self.logger.debug(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= backoff_factor

        # All attempts failed
        raise CommunicationError(
            f"Failed to send message {message.id} after {max_retries + 1} attempts",
            cause=last_exception
        )

    async def _send(self, message: AgentMessage) -> None:
        """
        Internal send method to be implemented by subclasses.

        Args:
            message: Message to send

        Raises:
            CommunicationError: If sending fails
        """
        # This is a placeholder - concrete implementations would override this
        raise NotImplementedError("Subclasses must implement _send method")

    def create_message(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: int = 5
    ) -> AgentMessage:
        """
        Create a new message with automatic ID generation.

        Args:
            sender: Sender agent ID
            recipient: Recipient agent ID
            message_type: Type of message
            payload: Message payload
            priority: Message priority (lower = higher priority)

        Returns:
            New AgentMessage instance
        """
        import uuid

        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            priority=priority
        )

        # Validate the created message
        self.validator.validate(message)

        return message

    def create_response_message(
        self,
        original_message: AgentMessage,
        sender: str,
        payload: Dict[str, Any],
        message_type: Optional[str] = None
    ) -> AgentMessage:
        """
        Create a response message to an original message.

        Args:
            original_message: Original message to respond to
            sender: Sender of the response
            payload: Response payload
            message_type: Optional message type (defaults to "response")

        Returns:
            Response AgentMessage
        """
        response_type = message_type or "response"

        # Add reference to original message in payload
        response_payload = {
            "original_message_id": original_message.id,
            "original_sender": original_message.sender,
            **payload
        }

        return self.create_message(
            sender=sender,
            recipient=original_message.sender,
            message_type=response_type,
            payload=response_payload,
            priority=original_message.priority
        )

    def create_error_message(
        self,
        original_message: AgentMessage,
        sender: str,
        error_message: str,
        error_code: Optional[str] = None
    ) -> AgentMessage:
        """
        Create an error response message.

        Args:
            original_message: Original message that caused the error
            sender: Sender of the error response
            error_message: Error description
            error_code: Optional error code

        Returns:
            Error response AgentMessage
        """
        error_payload = {
            "error": True,
            "error_message": error_message,
            "original_message_id": original_message.id,
        }

        if error_code:
            error_payload["error_code"] = error_code

        return self.create_message(
            sender=sender,
            recipient=original_message.sender,
            message_type="error",
            payload=error_payload,
            priority=1  # High priority for errors
        )

    def get_protocol_stats(self) -> Dict[str, Any]:
        """
        Get protocol statistics.

        Returns:
            Dictionary containing protocol statistics
        """
        return {
            "max_message_size": self.validator.max_message_size,
            "encoding": self.encoder.encoding,
            "protocol_version": "1.0"
        }
