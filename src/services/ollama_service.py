"""
Ollama integration service for local AI processing.

This module provides comprehensive integration with Ollama for local AI processing,
including model management, conversation handling, and streaming capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential


# Exception classes
class OllamaError(Exception):
    """Base exception for Ollama-related errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(message)


class OllamaConnectionError(OllamaError):
    """Exception raised when connection to Ollama fails."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Exception raised when requested model is not found."""
    pass


class OllamaTimeoutError(OllamaError):
    """Exception raised when request times out."""
    pass


class OllamaRateLimitError(OllamaError):
    """Exception raised when rate limit is exceeded."""
    pass


# Data models
@dataclass
class ModelInfo:
    """Information about an Ollama model."""
    name: str
    family: str
    parameter_size: str
    size: int
    digest: Optional[str] = None
    modified_at: Optional[datetime] = None
    capabilities: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Set default capabilities based on model family."""
        if not self.capabilities:
            if "llama" in self.family.lower():
                self.capabilities = [
                    "text_generation", "question_answering", "summarization",
                    "translation", "conversation", "analysis"
                ]
            elif "codellama" in self.family.lower() or "code" in self.name.lower():
                self.capabilities = [
                    "code_generation", "code_completion", "code_explanation",
                    "debugging", "refactoring", "documentation"
                ]
            elif "mistral" in self.family.lower():
                self.capabilities = [
                    "text_generation", "question_answering", "reasoning",
                    "analysis", "conversation"
                ]
            else:
                self.capabilities = ["text_generation", "conversation"]


@dataclass
class ConversationContext:
    """Context for managing conversations."""
    conversation_id: str
    system_prompt: Optional[str] = None
    model: str = "llama3.1:8b"
    max_context_length: int = 4096
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, images: Optional[List[str]] = None):
        """Add a message to the conversation."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if images:
            message["images"] = images
        
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
        
        # Manage context length
        self._manage_context_length()
    
    def _manage_context_length(self):
        """Manage context length by removing old messages if needed."""
        total_length = sum(len(msg["content"]) for msg in self.messages)
        
        while total_length > self.max_context_length and len(self.messages) > 1:
            # Remove oldest message (but keep system message if it's first)
            if self.messages[0]["role"] == "system" and len(self.messages) > 2:
                removed = self.messages.pop(1)  # Remove second message
            else:
                removed = self.messages.pop(0)  # Remove first message
            
            total_length -= len(removed["content"])
    
    def get_messages_for_api(self) -> List[Dict[str, Any]]:
        """Get messages formatted for Ollama API."""
        api_messages = []
        
        for msg in self.messages:
            api_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            if "images" in msg:
                api_msg["images"] = msg["images"]
            api_messages.append(api_msg)
        
        return api_messages


@dataclass
class StreamingResponse:
    """Response for streaming operations."""
    content: str
    model: str
    done: bool
    stream_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingRequest:
    """Request for AI processing operations."""
    prompt: str
    model: str = "llama3.1:8b"
    task_type: str = "text_generation"
    system_prompt: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    context: Optional[List[int]] = None
    conversation_id: Optional[str] = None
    stream: bool = False
    
    def __post_init__(self):
        """Set default options based on task type."""
        default_options = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
        }
        
        # Task-specific defaults
        if self.task_type == "code_generation":
            default_options.update({
                "temperature": 0.2,
                "top_p": 0.95,
            })
        elif self.task_type == "creative_writing":
            default_options.update({
                "temperature": 0.9,
                "top_p": 0.95,
            })
        elif self.task_type == "analysis":
            default_options.update({
                "temperature": 0.3,
                "top_p": 0.9,
            })
        
        # Merge with provided options (provided options take precedence)
        merged_options = {**default_options, **self.options}
        self.options = merged_options


@dataclass
class ProcessingResponse:
    """Response for AI processing operations."""
    content: str
    model: str
    success: bool
    processing_time: float = 0.0
    token_count: Optional[int] = None
    options: Dict[str, Any] = field(default_factory=dict)
    context: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None


class OllamaModelManager:
    """Manager for Ollama models and their capabilities."""
    
    def __init__(self, ollama_service: 'OllamaService'):
        """Initialize the model manager.
        
        Args:
            ollama_service: Reference to the OllamaService instance
        """
        self.ollama_service = ollama_service
        self.available_models: List[ModelInfo] = []
        self.model_preferences: Dict[str, str] = {}
        self.model_capabilities_cache: Dict[str, List[str]] = {}
    
    async def load_available_models(self) -> None:
        """Load available models from Ollama."""
        try:
            self.available_models = await self.ollama_service.list_models()
            self._update_capabilities_cache()
        except Exception as e:
            raise OllamaError(f"Failed to load available models: {e}")
    
    def _update_capabilities_cache(self) -> None:
        """Update the model capabilities cache."""
        self.model_capabilities_cache.clear()
        for model in self.available_models:
            self.model_capabilities_cache[model.name] = model.capabilities
    
    def get_model_for_task(self, task_type: str) -> Optional[ModelInfo]:
        """Get the best model for a specific task.
        
        Args:
            task_type: Type of task (e.g., 'text_generation', 'code_generation')
            
        Returns:
            ModelInfo object for the best model, or None if no suitable model found
        """
        # Check if user has set a preference for this task
        if task_type in self.model_preferences:
            preferred_model = self.get_model_by_name(self.model_preferences[task_type])
            if preferred_model:
                return preferred_model
        
        # Find models that support this task
        suitable_models = []
        for model in self.available_models:
            if task_type in model.capabilities:
                suitable_models.append(model)
        
        if not suitable_models:
            # Fall back to the first available model
            return self.available_models[0] if self.available_models else None
        
        # Return the most suitable model (can be improved with better heuristics)
        return self._select_optimal_model(suitable_models, task_type)
    
    def _select_optimal_model(self, models: List[ModelInfo], task_type: str) -> ModelInfo:
        """Select the optimal model from a list of suitable models."""
        # Simple heuristic: prefer code models for code tasks, larger models otherwise
        if task_type in ["code_generation", "code_completion", "debugging"]:
            code_models = [m for m in models if "code" in m.name.lower()]
            if code_models:
                return code_models[0]
        
        # For other tasks, prefer larger models (rough heuristic based on parameter size)
        def model_score(model: ModelInfo) -> int:
            size_map = {"7B": 7, "8B": 8, "13B": 13, "70B": 70}
            return size_map.get(model.parameter_size, 0)
        
        return max(models, key=model_score)
    
    def get_model_by_name(self, name: str) -> Optional[ModelInfo]:
        """Get model information by name.
        
        Args:
            name: Model name
            
        Returns:
            ModelInfo object or None if not found
        """
        for model in self.available_models:
            if model.name == name:
                return model
        return None
    
    def set_model_preference(self, task_type: str, model_name: str) -> None:
        """Set a model preference for a specific task type.
        
        Args:
            task_type: Type of task
            model_name: Preferred model name
        """
        self.model_preferences[task_type] = model_name
    
    def get_model_capabilities(self, model_name: str) -> List[str]:
        """Get capabilities for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of capabilities
        """
        return self.model_capabilities_cache.get(model_name, [])
    
    def get_optimal_model(self, criteria: str = "size", task: Optional[str] = None) -> Optional[ModelInfo]:
        """Get optimal model based on criteria.
        
        Args:
            criteria: Selection criteria ('size', 'speed', 'task')
            task: Task type (required if criteria is 'task')
            
        Returns:
            Optimal ModelInfo object
        """
        if not self.available_models:
            return None
        
        if criteria == "task" and task:
            return self.get_model_for_task(task)
        elif criteria == "size":
            # Return largest model
            def model_size_score(model: ModelInfo) -> int:
                size_map = {"7B": 7, "8B": 8, "13B": 13, "70B": 70}
                return size_map.get(model.parameter_size, 0)
            return max(self.available_models, key=model_size_score)
        elif criteria == "speed":
            # Return smallest model (assumed to be fastest)
            def model_size_score(model: ModelInfo) -> int:
                size_map = {"7B": 7, "8B": 8, "13B": 13, "70B": 70}
                return size_map.get(model.parameter_size, 999)
            return min(self.available_models, key=model_size_score)
        
        return self.available_models[0]


class OllamaConversationManager:
    """Manager for conversation contexts and history."""
    
    def __init__(self, max_context_length: int = 4096):
        """Initialize the conversation manager.
        
        Args:
            max_context_length: Maximum context length for conversations
        """
        self.max_context_length = max_context_length
        self.conversations: Dict[str, ConversationContext] = {}
    
    async def create_conversation(
        self,
        system_prompt: Optional[str] = None,
        model: str = "llama3.1:8b",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new conversation.
        
        Args:
            system_prompt: Optional system prompt
            model: Model to use for the conversation
            metadata: Optional metadata for the conversation
            
        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())
        
        context = ConversationContext(
            conversation_id=conversation_id,
            system_prompt=system_prompt,
            model=model,
            max_context_length=self.max_context_length,
            metadata=metadata or {}
        )
        
        # Add system message if provided
        if system_prompt:
            context.add_message("system", system_prompt)
        
        self.conversations[conversation_id] = context
        return conversation_id
    
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        images: Optional[List[str]] = None
    ) -> None:
        """Add a message to a conversation.
        
        Args:
            conversation_id: ID of the conversation
            role: Role of the message sender (user, assistant, system)
            content: Message content
            images: Optional list of image data
        """
        if conversation_id not in self.conversations:
            raise OllamaError(f"Conversation {conversation_id} not found")
        
        context = self.conversations[conversation_id]
        context.add_message(role, content, images)
    
    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List of messages in the conversation
        """
        if conversation_id not in self.conversations:
            raise OllamaError(f"Conversation {conversation_id} not found")
        
        return self.conversations[conversation_id].get_messages_for_api()
    
    async def clear_conversation(self, conversation_id: str) -> None:
        """Clear conversation history (except system message).
        
        Args:
            conversation_id: ID of the conversation
        """
        if conversation_id not in self.conversations:
            raise OllamaError(f"Conversation {conversation_id} not found")
        
        context = self.conversations[conversation_id]
        # Keep system message if it exists
        system_messages = [msg for msg in context.messages if msg["role"] == "system"]
        context.messages = system_messages
        context.updated_at = datetime.now(timezone.utc)
    
    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation.
        
        Args:
            conversation_id: ID of the conversation
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    async def get_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """Get a summary of the conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Summary of the conversation
        """
        if conversation_id not in self.conversations:
            return None
        
        context = self.conversations[conversation_id]
        if not context.messages:
            return "Empty conversation"
        
        # Simple summary: extract key topics from messages
        all_content = " ".join([msg["content"] for msg in context.messages if msg["role"] != "system"])
        
        # For a real implementation, you might use the AI model to generate a summary
        # For now, return a simple summary
        words = all_content.split()
        if len(words) <= 50:
            return all_content
        else:
            return " ".join(words[:50]) + "..."


class OllamaStreamHandler:
    """Handler for streaming responses from Ollama."""
    
    def __init__(self):
        """Initialize the stream handler."""
        self.active_streams: Dict[str, StreamingResponse] = {}
        self.stream_callbacks: Dict[str, List[callable]] = {}
    
    async def handle_stream_chunk(self, chunk: StreamingResponse) -> None:
        """Handle a streaming chunk.
        
        Args:
            chunk: Streaming response chunk
        """
        if not chunk.stream_id:
            return
        
        stream_id = chunk.stream_id
        
        # Update or create stream state
        if stream_id in self.active_streams:
            # Accumulate content
            self.active_streams[stream_id].content += chunk.content
        else:
            self.active_streams[stream_id] = chunk
        
        # Call registered callbacks
        callbacks = self.stream_callbacks.get(stream_id, [])
        for callback in callbacks:
            try:
                callback(chunk)
            except Exception:
                pass  # Don't let callback errors break streaming
        
        # Clean up completed streams
        if chunk.done:
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
            if stream_id in self.stream_callbacks:
                del self.stream_callbacks[stream_id]
    
    def register_stream_callback(self, stream_id: str, callback: callable) -> None:
        """Register a callback for a stream.
        
        Args:
            stream_id: ID of the stream
            callback: Callback function to call for each chunk
        """
        if stream_id not in self.stream_callbacks:
            self.stream_callbacks[stream_id] = []
        self.stream_callbacks[stream_id].append(callback)
    
    def get_stream_content(self, stream_id: str) -> Optional[str]:
        """Get accumulated content for a stream.
        
        Args:
            stream_id: ID of the stream
            
        Returns:
            Accumulated content or None if stream not found
        """
        stream = self.active_streams.get(stream_id)
        return stream.content if stream else None
    
    async def cancel_stream(self, stream_id: str) -> None:
        """Cancel a stream.
        
        Args:
            stream_id: ID of the stream to cancel
        """
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
        if stream_id in self.stream_callbacks:
            del self.stream_callbacks[stream_id]


class OllamaService:
    """Main service for Ollama integration."""
    
    def __init__(
        self,
        config: Any,
        logger: logging.Logger,
        base_url: str = "http://localhost:11434"
    ):
        """Initialize the Ollama service.
        
        Args:
            config: Configuration manager instance
            logger: Logger instance
            base_url: Base URL for Ollama API
        """
        self.config = config
        self.logger = logger
        self.base_url = base_url
        
        # Configuration settings
        self.timeout = config.get("ollama.timeout", 30)
        self.default_model = config.get("ollama.default_model", "llama3.1:8b")
        self.max_context_length = config.get("ollama.max_context_length", 4096)
        self.stream_enabled = config.get("ollama.stream_enabled", True)
        self.retry_attempts = config.get("ollama.retry_attempts", 3)
        self.retry_delay = config.get("ollama.retry_delay", 1.0)
        
        # Default generation options
        self.default_options = {
            "temperature": config.get("ollama.temperature", 0.7),
            "top_p": config.get("ollama.top_p", 0.9),
            "top_k": config.get("ollama.top_k", 40),
        }
        
        # Initialize client
        self.client = ollama.AsyncClient(host=base_url)
        self.is_connected = False
        
        # Initialize managers
        self.model_manager = OllamaModelManager(self)
        self.conversation_manager = OllamaConversationManager(self.max_context_length)
        self.stream_handler = OllamaStreamHandler()
        
        # Metrics
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
    
    async def connect(self) -> None:
        """Connect to Ollama service."""
        try:
            # Test connection by listing models
            await self.client.list()
            self.is_connected = True
            self.logger.info(f"Connected to Ollama at {self.base_url}")
            
            # Load available models
            await self.model_manager.load_available_models()
            
        except Exception as e:
            self.is_connected = False
            self.logger.error(f"Failed to connect to Ollama: {e}")
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Ollama service."""
        self.is_connected = False
        self.logger.info("Disconnected from Ollama")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Ollama service.
        
        Returns:
            Health status dictionary
        """
        try:
            if not self.is_connected:
                return {
                    "status": "unhealthy",
                    "connected": False,
                    "error": "Not connected to Ollama",
                    "last_check": datetime.now(timezone.utc).isoformat()
                }
            
            # Try to list models as health check
            models = await self.client.list()
            model_count = len(models.get("models", []))
            
            uptime = time.time() - self.start_time
            
            return {
                "status": "healthy",
                "connected": True,
                "models_available": model_count,
                "uptime": uptime,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "last_check": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "last_check": datetime.now(timezone.utc).isoformat()
            }
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models.
        
        Returns:
            List of ModelInfo objects
        """
        try:
            response = await self.client.list()
            models = []
            
            for model_data in response.get("models", []):
                model_info = ModelInfo(
                    name=model_data["name"],
                    family=model_data.get("details", {}).get("family", "unknown"),
                    parameter_size=model_data.get("details", {}).get("parameter_size", "unknown"),
                    size=model_data["size"],
                    digest=model_data.get("digest"),
                    modified_at=datetime.fromisoformat(model_data["modified_at"].replace("Z", "+00:00"))
                    if model_data.get("modified_at") else None
                )
                models.append(model_info)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            raise OllamaError(f"Failed to list models: {e}")
    
    async def get_model_info(self, model_name: str) -> ModelInfo:
        """Get detailed information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo object
            
        Raises:
            OllamaModelNotFoundError: If model is not found
        """
        try:
            response = await self.client.show(model_name)
            
            model_info = ModelInfo(
                name=model_name,
                family=response.get("details", {}).get("family", "unknown"),
                parameter_size=response.get("details", {}).get("parameter_size", "unknown"),
                size=0,  # Size not available in show response
                digest=response.get("details", {}).get("digest")
            )
            
            return model_info
            
        except Exception as e:
            if "not found" in str(e).lower():
                raise OllamaModelNotFoundError(f"Model {model_name} not found")
            raise OllamaError(f"Failed to get model info for {model_name}: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        context: Optional[List[int]] = None
    ) -> ProcessingResponse:
        """Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            model: Model to use (defaults to default_model)
            system: System prompt
            options: Generation options
            context: Context from previous generation
            
        Returns:
            ProcessingResponse object
        """
        start_time = time.time()
        model = model or self.default_model
        options = {**self.default_options, **(options or {})}
        
        try:
            self.request_count += 1
            
            response = await asyncio.wait_for(
                self.client.generate(
                    model=model,
                    prompt=prompt,
                    system=system,
                    options=options,
                    context=context,
                    stream=False
                ),
                timeout=self.timeout
            )
            
            processing_time = time.time() - start_time
            
            return ProcessingResponse(
                content=response["response"],
                model=model,
                success=True,
                processing_time=processing_time,
                token_count=response.get("eval_count"),
                options=options,
                context=response.get("context"),
                metadata={
                    "total_duration": response.get("total_duration"),
                    "load_duration": response.get("load_duration"),
                    "prompt_eval_count": response.get("prompt_eval_count"),
                    "prompt_eval_duration": response.get("prompt_eval_duration"),
                    "eval_duration": response.get("eval_duration")
                }
            )
            
        except asyncio.TimeoutError:
            self.error_count += 1
            raise OllamaTimeoutError(f"Request timed out after {self.timeout} seconds")
        except Exception as e:
            self.error_count += 1
            if "rate limit" in str(e).lower():
                raise OllamaRateLimitError(f"Rate limit exceeded: {e}")
            
            processing_time = time.time() - start_time
            return ProcessingResponse(
                content="",
                model=model,
                success=False,
                processing_time=processing_time,
                error=str(e)
            )
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResponse:
        """Generate chat completion using Ollama.
        
        Args:
            messages: List of messages in chat format
            model: Model to use (defaults to default_model)
            options: Generation options
            
        Returns:
            ProcessingResponse object
        """
        start_time = time.time()
        model = model or self.default_model
        options = {**self.default_options, **(options or {})}
        
        try:
            self.request_count += 1
            
            response = await asyncio.wait_for(
                self.client.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    stream=False
                ),
                timeout=self.timeout
            )
            
            processing_time = time.time() - start_time
            
            return ProcessingResponse(
                content=response["message"]["content"],
                model=model,
                success=True,
                processing_time=processing_time,
                token_count=response.get("eval_count"),
                options=options,
                metadata={
                    "total_duration": response.get("total_duration"),
                    "load_duration": response.get("load_duration"),
                    "prompt_eval_count": response.get("prompt_eval_count"),
                    "prompt_eval_duration": response.get("prompt_eval_duration"),
                    "eval_duration": response.get("eval_duration")
                }
            )
            
        except asyncio.TimeoutError:
            self.error_count += 1
            raise OllamaTimeoutError(f"Request timed out after {self.timeout} seconds")
        except Exception as e:
            self.error_count += 1
            if "rate limit" in str(e).lower():
                raise OllamaRateLimitError(f"Rate limit exceeded: {e}")
            
            processing_time = time.time() - start_time
            return ProcessingResponse(
                content="",
                model=model,
                success=False,
                processing_time=processing_time,
                error=str(e)
            )
    
    async def stream_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        context: Optional[List[int]] = None
    ) -> AsyncGenerator[StreamingResponse, None]:
        """Stream text generation using Ollama.
        
        Args:
            prompt: Input prompt
            model: Model to use (defaults to default_model)
            system: System prompt
            options: Generation options
            context: Context from previous generation
            
        Yields:
            StreamingResponse objects
        """
        model = model or self.default_model
        options = {**self.default_options, **(options or {})}
        
        try:
            self.request_count += 1
            
            async for chunk in self.client.generate(
                model=model,
                prompt=prompt,
                system=system,
                options=options,
                context=context,
                stream=True
            ):
                yield StreamingResponse(
                    content=chunk.get("response", ""),
                    model=model,
                    done=chunk.get("done", False),
                    context=chunk.get("context") if chunk.get("done") else None,
                    metadata=chunk
                )
                
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Streaming generation failed: {e}")
            yield StreamingResponse(
                content="",
                model=model,
                done=True,
                metadata={"error": str(e)}
            )
    
    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[StreamingResponse, None]:
        """Stream chat completion using Ollama.
        
        Args:
            messages: List of messages in chat format
            model: Model to use (defaults to default_model)
            options: Generation options
            
        Yields:
            StreamingResponse objects
        """
        model = model or self.default_model
        options = {**self.default_options, **(options or {})}
        
        try:
            self.request_count += 1
            
            async for chunk in self.client.chat(
                model=model,
                messages=messages,
                options=options,
                stream=True
            ):
                yield StreamingResponse(
                    content=chunk.get("message", {}).get("content", ""),
                    model=model,
                    done=chunk.get("done", False),
                    metadata=chunk
                )
                
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Streaming chat failed: {e}")
            yield StreamingResponse(
                content="",
                model=model,
                done=True,
                metadata={"error": str(e)}
            )
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.client.pull(model_name)
            self.logger.info(f"Successfully pulled model: {model_name}")
            
            # Reload available models
            await self.model_manager.load_available_models()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model from Ollama.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.client.delete(model_name)
            self.logger.info(f"Successfully deleted model: {model_name}")
            
            # Reload available models
            await self.model_manager.load_available_models()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_name}: {e}")
            return False