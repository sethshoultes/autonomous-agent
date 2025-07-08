"""
Comprehensive tests for Ollama integration service.

This module tests the Ollama integration service following TDD principles,
ensuring all functionality is thoroughly tested before implementation.
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from src.services.ollama_service import (
    OllamaService,
    OllamaModelManager,
    OllamaConversationManager,
    OllamaStreamHandler,
    OllamaError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaRateLimitError,
    ModelInfo,
    ConversationContext,
    StreamingResponse,
    ProcessingRequest,
    ProcessingResponse,
)
from src.config.manager import ConfigManager
from src.logging.manager import LoggingManager
from tests.mocks.ollama_mocks import (
    MockOllamaClient,
    MockOllamaModel,
    MockOllamaResponse,
    MockOllamaChatResponse,
    MockOllamaError,
    mock_ollama_api,
)


class TestOllamaService:
    """Test suite for OllamaService class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: {
            "ollama.host": "localhost",
            "ollama.port": 11434,
            "ollama.timeout": 30,
            "ollama.default_model": "llama3.1:8b",
            "ollama.max_context_length": 4096,
            "ollama.stream_enabled": True,
            "ollama.retry_attempts": 3,
            "ollama.retry_delay": 1.0,
            "ollama.temperature": 0.7,
            "ollama.top_p": 0.9,
            "ollama.top_k": 40,
        }.get(key, default)
        return config
    
    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        logger = MagicMock()
        return logger
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Create mock Ollama client."""
        client = MockOllamaClient()
        client.add_model(MockOllamaModel("llama3.1:8b"))
        client.add_model(MockOllamaModel("codellama:7b"))
        return client
    
    @pytest.fixture
    def ollama_service(self, mock_config, mock_logger, mock_ollama_client):
        """Create OllamaService instance for testing."""
        with patch("src.services.ollama_service.ollama.AsyncClient") as mock_client:
            mock_client.return_value = mock_ollama_client.create_client_mock()
            service = OllamaService(
                config=mock_config,
                logger=mock_logger,
                base_url="http://localhost:11434"
            )
            return service
    
    async def test_initialization(self, mock_config, mock_logger):
        """Test OllamaService initialization."""
        service = OllamaService(
            config=mock_config,
            logger=mock_logger,
            base_url="http://localhost:11434"
        )
        
        assert service.config == mock_config
        assert service.logger == mock_logger
        assert service.base_url == "http://localhost:11434"
        assert service.timeout == 30
        assert service.default_model == "llama3.1:8b"
        assert service.max_context_length == 4096
        assert service.stream_enabled is True
        assert service.retry_attempts == 3
        assert service.retry_delay == 1.0
        assert service.is_connected is False
    
    async def test_connect_success(self, ollama_service):
        """Test successful connection to Ollama."""
        await ollama_service.connect()
        
        assert ollama_service.is_connected is True
        ollama_service.logger.info.assert_called_with("Connected to Ollama at http://localhost:11434")
    
    async def test_connect_failure(self, ollama_service):
        """Test connection failure to Ollama."""
        with patch.object(ollama_service.client, "list", side_effect=Exception("Connection failed")):
            with pytest.raises(OllamaConnectionError):
                await ollama_service.connect()
            
            assert ollama_service.is_connected is False
    
    async def test_disconnect(self, ollama_service):
        """Test disconnection from Ollama."""
        await ollama_service.connect()
        await ollama_service.disconnect()
        
        assert ollama_service.is_connected is False
        ollama_service.logger.info.assert_called_with("Disconnected from Ollama")
    
    async def test_health_check_healthy(self, ollama_service):
        """Test health check when service is healthy."""
        await ollama_service.connect()
        
        health_status = await ollama_service.health_check()
        
        assert health_status["status"] == "healthy"
        assert health_status["connected"] is True
        assert "models_available" in health_status
        assert "uptime" in health_status
        assert "last_check" in health_status
    
    async def test_health_check_unhealthy(self, ollama_service):
        """Test health check when service is unhealthy."""
        health_status = await ollama_service.health_check()
        
        assert health_status["status"] == "unhealthy"
        assert health_status["connected"] is False
    
    async def test_list_models(self, ollama_service):
        """Test listing available models."""
        await ollama_service.connect()
        
        models = await ollama_service.list_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, ModelInfo) for model in models)
    
    async def test_get_model_info(self, ollama_service):
        """Test getting model information."""
        await ollama_service.connect()
        
        model_info = await ollama_service.get_model_info("llama3.1:8b")
        
        assert isinstance(model_info, ModelInfo)
        assert model_info.name == "llama3.1:8b"
        assert model_info.family == "llama"
        assert model_info.parameter_size == "8B"
    
    async def test_get_model_info_not_found(self, ollama_service):
        """Test getting model information for non-existent model."""
        await ollama_service.connect()
        
        with pytest.raises(OllamaModelNotFoundError):
            await ollama_service.get_model_info("non-existent-model")
    
    async def test_generate_text(self, ollama_service):
        """Test text generation."""
        await ollama_service.connect()
        
        response = await ollama_service.generate_text(
            prompt="What is AI?",
            model="llama3.1:8b",
            system="You are a helpful assistant."
        )
        
        assert isinstance(response, ProcessingResponse)
        assert response.content is not None
        assert response.model == "llama3.1:8b"
        assert response.success is True
    
    async def test_generate_text_with_options(self, ollama_service):
        """Test text generation with custom options."""
        await ollama_service.connect()
        
        options = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repeat_penalty": 1.1,
        }
        
        response = await ollama_service.generate_text(
            prompt="Explain quantum computing",
            model="llama3.1:8b",
            options=options
        )
        
        assert isinstance(response, ProcessingResponse)
        assert response.content is not None
        assert response.options == options
    
    async def test_chat_completion(self, ollama_service):
        """Test chat completion."""
        await ollama_service.connect()
        
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
        ]
        
        response = await ollama_service.chat_completion(
            messages=messages,
            model="llama3.1:8b"
        )
        
        assert isinstance(response, ProcessingResponse)
        assert response.content is not None
        assert response.model == "llama3.1:8b"
        assert response.success is True
    
    async def test_stream_generation(self, ollama_service):
        """Test streaming text generation."""
        await ollama_service.connect()
        
        chunks = []
        async for chunk in ollama_service.stream_generate(
            prompt="Tell me a story",
            model="llama3.1:8b"
        ):
            chunks.append(chunk)
            assert isinstance(chunk, StreamingResponse)
        
        assert len(chunks) > 0
        assert chunks[-1].done is True
    
    async def test_stream_chat(self, ollama_service):
        """Test streaming chat completion."""
        await ollama_service.connect()
        
        messages = [
            {"role": "user", "content": "Write a poem about AI"},
        ]
        
        chunks = []
        async for chunk in ollama_service.stream_chat(
            messages=messages,
            model="llama3.1:8b"
        ):
            chunks.append(chunk)
            assert isinstance(chunk, StreamingResponse)
        
        assert len(chunks) > 0
        assert chunks[-1].done is True
    
    async def test_pull_model(self, ollama_service):
        """Test pulling a model."""
        await ollama_service.connect()
        
        success = await ollama_service.pull_model("mistral:7b")
        
        assert success is True
        ollama_service.logger.info.assert_called_with("Successfully pulled model: mistral:7b")
    
    async def test_delete_model(self, ollama_service):
        """Test deleting a model."""
        await ollama_service.connect()
        
        success = await ollama_service.delete_model("llama3.1:8b")
        
        assert success is True
        ollama_service.logger.info.assert_called_with("Successfully deleted model: llama3.1:8b")
    
    async def test_timeout_handling(self, ollama_service):
        """Test timeout handling."""
        await ollama_service.connect()
        
        with patch.object(ollama_service.client, "generate", side_effect=asyncio.TimeoutError()):
            with pytest.raises(OllamaTimeoutError):
                await ollama_service.generate_text("Test prompt", "llama3.1:8b")
    
    async def test_retry_mechanism(self, ollama_service):
        """Test retry mechanism for transient failures."""
        await ollama_service.connect()
        
        call_count = 0
        
        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient error")
            return MockOllamaResponse().to_ollama_format()
        
        with patch.object(ollama_service.client, "generate", side_effect=mock_generate):
            response = await ollama_service.generate_text("Test prompt", "llama3.1:8b")
            
            assert response.success is True
            assert call_count == 3
    
    async def test_rate_limit_handling(self, ollama_service):
        """Test rate limit handling."""
        await ollama_service.connect()
        
        with patch.object(ollama_service.client, "generate", side_effect=Exception("Rate limit exceeded")):
            with pytest.raises(OllamaRateLimitError):
                await ollama_service.generate_text("Test prompt", "llama3.1:8b")


class TestOllamaModelManager:
    """Test suite for OllamaModelManager class."""
    
    @pytest.fixture
    def mock_ollama_service(self):
        """Create mock OllamaService."""
        service = MagicMock()
        service.list_models.return_value = [
            ModelInfo(name="llama3.1:8b", family="llama", parameter_size="8B", size=4661211808),
            ModelInfo(name="codellama:7b", family="codellama", parameter_size="7B", size=3800000000),
        ]
        return service
    
    @pytest.fixture
    def model_manager(self, mock_ollama_service):
        """Create OllamaModelManager instance."""
        return OllamaModelManager(mock_ollama_service)
    
    async def test_initialization(self, model_manager, mock_ollama_service):
        """Test OllamaModelManager initialization."""
        assert model_manager.ollama_service == mock_ollama_service
        assert model_manager.available_models == []
        assert model_manager.model_preferences == {}
    
    async def test_load_available_models(self, model_manager):
        """Test loading available models."""
        await model_manager.load_available_models()
        
        assert len(model_manager.available_models) == 2
        assert model_manager.available_models[0].name == "llama3.1:8b"
        assert model_manager.available_models[1].name == "codellama:7b"
    
    async def test_get_model_for_task(self, model_manager):
        """Test getting appropriate model for a task."""
        await model_manager.load_available_models()
        
        # Test general text generation
        model = model_manager.get_model_for_task("text_generation")
        assert model.name == "llama3.1:8b"
        
        # Test code generation
        model = model_manager.get_model_for_task("code_generation")
        assert model.name == "codellama:7b"
    
    async def test_get_model_by_name(self, model_manager):
        """Test getting model by name."""
        await model_manager.load_available_models()
        
        model = model_manager.get_model_by_name("llama3.1:8b")
        assert model.name == "llama3.1:8b"
        
        model = model_manager.get_model_by_name("non-existent")
        assert model is None
    
    async def test_set_model_preference(self, model_manager):
        """Test setting model preference for a task."""
        await model_manager.load_available_models()
        
        model_manager.set_model_preference("summarization", "llama3.1:8b")
        
        assert model_manager.model_preferences["summarization"] == "llama3.1:8b"
        
        model = model_manager.get_model_for_task("summarization")
        assert model.name == "llama3.1:8b"
    
    async def test_get_model_capabilities(self, model_manager):
        """Test getting model capabilities."""
        await model_manager.load_available_models()
        
        capabilities = model_manager.get_model_capabilities("llama3.1:8b")
        
        assert "text_generation" in capabilities
        assert "question_answering" in capabilities
        assert "summarization" in capabilities
        
        capabilities = model_manager.get_model_capabilities("codellama:7b")
        assert "code_generation" in capabilities
        assert "code_completion" in capabilities
    
    async def test_get_optimal_model(self, model_manager):
        """Test getting optimal model based on criteria."""
        await model_manager.load_available_models()
        
        # Test getting largest model
        model = model_manager.get_optimal_model(criteria="size")
        assert model.name == "llama3.1:8b"
        
        # Test getting model for specific task
        model = model_manager.get_optimal_model(criteria="task", task="code_generation")
        assert model.name == "codellama:7b"


class TestOllamaConversationManager:
    """Test suite for OllamaConversationManager class."""
    
    @pytest.fixture
    def conversation_manager(self):
        """Create OllamaConversationManager instance."""
        return OllamaConversationManager(max_context_length=4096)
    
    async def test_initialization(self, conversation_manager):
        """Test OllamaConversationManager initialization."""
        assert conversation_manager.max_context_length == 4096
        assert conversation_manager.conversations == {}
    
    async def test_create_conversation(self, conversation_manager):
        """Test creating a new conversation."""
        conversation_id = await conversation_manager.create_conversation(
            system_prompt="You are a helpful assistant.",
            model="llama3.1:8b"
        )
        
        assert conversation_id is not None
        assert conversation_id in conversation_manager.conversations
        
        context = conversation_manager.conversations[conversation_id]
        assert context.system_prompt == "You are a helpful assistant."
        assert context.model == "llama3.1:8b"
        assert len(context.messages) == 0
    
    async def test_add_message(self, conversation_manager):
        """Test adding message to conversation."""
        conversation_id = await conversation_manager.create_conversation()
        
        await conversation_manager.add_message(
            conversation_id=conversation_id,
            role="user",
            content="Hello, how are you?"
        )
        
        context = conversation_manager.conversations[conversation_id]
        assert len(context.messages) == 1
        assert context.messages[0]["role"] == "user"
        assert context.messages[0]["content"] == "Hello, how are you?"
    
    async def test_get_conversation_history(self, conversation_manager):
        """Test getting conversation history."""
        conversation_id = await conversation_manager.create_conversation()
        
        await conversation_manager.add_message(conversation_id, "user", "Hello")
        await conversation_manager.add_message(conversation_id, "assistant", "Hi there!")
        
        history = await conversation_manager.get_conversation_history(conversation_id)
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    async def test_clear_conversation(self, conversation_manager):
        """Test clearing conversation."""
        conversation_id = await conversation_manager.create_conversation()
        
        await conversation_manager.add_message(conversation_id, "user", "Hello")
        await conversation_manager.clear_conversation(conversation_id)
        
        context = conversation_manager.conversations[conversation_id]
        assert len(context.messages) == 0
    
    async def test_delete_conversation(self, conversation_manager):
        """Test deleting conversation."""
        conversation_id = await conversation_manager.create_conversation()
        
        await conversation_manager.delete_conversation(conversation_id)
        
        assert conversation_id not in conversation_manager.conversations
    
    async def test_context_length_management(self, conversation_manager):
        """Test context length management."""
        conversation_manager.max_context_length = 100  # Very small for testing
        
        conversation_id = await conversation_manager.create_conversation()
        
        # Add messages that exceed context length
        for i in range(10):
            await conversation_manager.add_message(
                conversation_id, "user", f"This is a long message number {i} " * 20
            )
        
        context = conversation_manager.conversations[conversation_id]
        total_length = sum(len(msg["content"]) for msg in context.messages)
        
        assert total_length <= conversation_manager.max_context_length
    
    async def test_conversation_summary(self, conversation_manager):
        """Test generating conversation summary."""
        conversation_id = await conversation_manager.create_conversation()
        
        await conversation_manager.add_message(conversation_id, "user", "What is Python?")
        await conversation_manager.add_message(conversation_id, "assistant", "Python is a programming language.")
        
        summary = await conversation_manager.get_conversation_summary(conversation_id)
        
        assert summary is not None
        assert len(summary) > 0
        assert "Python" in summary


class TestOllamaStreamHandler:
    """Test suite for OllamaStreamHandler class."""
    
    @pytest.fixture
    def stream_handler(self):
        """Create OllamaStreamHandler instance."""
        return OllamaStreamHandler()
    
    async def test_initialization(self, stream_handler):
        """Test OllamaStreamHandler initialization."""
        assert stream_handler.active_streams == {}
        assert stream_handler.stream_callbacks == {}
    
    async def test_handle_stream_chunk(self, stream_handler):
        """Test handling stream chunks."""
        stream_id = "test_stream"
        
        chunk = StreamingResponse(
            content="Hello",
            model="llama3.1:8b",
            done=False,
            stream_id=stream_id
        )
        
        await stream_handler.handle_stream_chunk(chunk)
        
        assert stream_id in stream_handler.active_streams
        assert stream_handler.active_streams[stream_id].content == "Hello"
    
    async def test_register_stream_callback(self, stream_handler):
        """Test registering stream callback."""
        callback_called = False
        
        def test_callback(chunk):
            nonlocal callback_called
            callback_called = True
        
        stream_id = "test_stream"
        stream_handler.register_stream_callback(stream_id, test_callback)
        
        chunk = StreamingResponse(
            content="Hello",
            model="llama3.1:8b",
            done=False,
            stream_id=stream_id
        )
        
        await stream_handler.handle_stream_chunk(chunk)
        
        assert callback_called is True
    
    async def test_complete_stream(self, stream_handler):
        """Test completing a stream."""
        stream_id = "test_stream"
        
        # Add some chunks
        for i in range(3):
            chunk = StreamingResponse(
                content=f"Part {i}",
                model="llama3.1:8b",
                done=False,
                stream_id=stream_id
            )
            await stream_handler.handle_stream_chunk(chunk)
        
        # Complete the stream
        final_chunk = StreamingResponse(
            content="Final part",
            model="llama3.1:8b",
            done=True,
            stream_id=stream_id
        )
        await stream_handler.handle_stream_chunk(final_chunk)
        
        # Stream should be removed from active streams
        assert stream_id not in stream_handler.active_streams
    
    async def test_get_stream_content(self, stream_handler):
        """Test getting accumulated stream content."""
        stream_id = "test_stream"
        
        # Add chunks
        chunks = ["Hello", " world", "!"]
        for chunk_content in chunks:
            chunk = StreamingResponse(
                content=chunk_content,
                model="llama3.1:8b",
                done=False,
                stream_id=stream_id
            )
            await stream_handler.handle_stream_chunk(chunk)
        
        content = stream_handler.get_stream_content(stream_id)
        assert content == "Hello world!"
    
    async def test_cancel_stream(self, stream_handler):
        """Test canceling a stream."""
        stream_id = "test_stream"
        
        chunk = StreamingResponse(
            content="Hello",
            model="llama3.1:8b",
            done=False,
            stream_id=stream_id
        )
        await stream_handler.handle_stream_chunk(chunk)
        
        await stream_handler.cancel_stream(stream_id)
        
        assert stream_id not in stream_handler.active_streams
        assert stream_id not in stream_handler.stream_callbacks


class TestOllamaErrors:
    """Test suite for Ollama error classes."""
    
    def test_ollama_error(self):
        """Test OllamaError base class."""
        error = OllamaError("Test error", {"key": "value"})
        
        assert str(error) == "Test error"
        assert error.context == {"key": "value"}
    
    def test_ollama_connection_error(self):
        """Test OllamaConnectionError."""
        error = OllamaConnectionError("Connection failed")
        
        assert isinstance(error, OllamaError)
        assert str(error) == "Connection failed"
    
    def test_ollama_model_not_found_error(self):
        """Test OllamaModelNotFoundError."""
        error = OllamaModelNotFoundError("Model not found", {"model": "test"})
        
        assert isinstance(error, OllamaError)
        assert str(error) == "Model not found"
        assert error.context == {"model": "test"}
    
    def test_ollama_timeout_error(self):
        """Test OllamaTimeoutError."""
        error = OllamaTimeoutError("Request timed out")
        
        assert isinstance(error, OllamaError)
        assert str(error) == "Request timed out"
    
    def test_ollama_rate_limit_error(self):
        """Test OllamaRateLimitError."""
        error = OllamaRateLimitError("Rate limit exceeded")
        
        assert isinstance(error, OllamaError)
        assert str(error) == "Rate limit exceeded"


class TestDataModels:
    """Test suite for Ollama data models."""
    
    def test_model_info(self):
        """Test ModelInfo data model."""
        model_info = ModelInfo(
            name="llama3.1:8b",
            family="llama",
            parameter_size="8B",
            size=4661211808,
            digest="sha256:test",
            modified_at=datetime.now(timezone.utc)
        )
        
        assert model_info.name == "llama3.1:8b"
        assert model_info.family == "llama"
        assert model_info.parameter_size == "8B"
        assert model_info.size == 4661211808
    
    def test_conversation_context(self):
        """Test ConversationContext data model."""
        context = ConversationContext(
            conversation_id="test_conv",
            system_prompt="You are helpful",
            model="llama3.1:8b",
            max_context_length=4096
        )
        
        assert context.conversation_id == "test_conv"
        assert context.system_prompt == "You are helpful"
        assert context.model == "llama3.1:8b"
        assert context.max_context_length == 4096
        assert len(context.messages) == 0
    
    def test_streaming_response(self):
        """Test StreamingResponse data model."""
        response = StreamingResponse(
            content="Hello",
            model="llama3.1:8b",
            done=False,
            stream_id="test_stream"
        )
        
        assert response.content == "Hello"
        assert response.model == "llama3.1:8b"
        assert response.done is False
        assert response.stream_id == "test_stream"
    
    def test_processing_request(self):
        """Test ProcessingRequest data model."""
        request = ProcessingRequest(
            prompt="Test prompt",
            model="llama3.1:8b",
            task_type="text_generation",
            system_prompt="You are helpful",
            options={"temperature": 0.7}
        )
        
        assert request.prompt == "Test prompt"
        assert request.model == "llama3.1:8b"
        assert request.task_type == "text_generation"
        assert request.system_prompt == "You are helpful"
        assert request.options == {"temperature": 0.7}
    
    def test_processing_response(self):
        """Test ProcessingResponse data model."""
        response = ProcessingResponse(
            content="Response content",
            model="llama3.1:8b",
            success=True,
            processing_time=1.5,
            token_count=100
        )
        
        assert response.content == "Response content"
        assert response.model == "llama3.1:8b"
        assert response.success is True
        assert response.processing_time == 1.5
        assert response.token_count == 100


class TestIntegrationScenarios:
    """Test suite for integration scenarios."""
    
    @pytest.fixture
    def full_ollama_setup(self, mock_config, mock_logger):
        """Create full Ollama setup for integration tests."""
        with patch("src.services.ollama_service.ollama.AsyncClient") as mock_client:
            client = MockOllamaClient()
            client.add_model(MockOllamaModel("llama3.1:8b"))
            client.add_model(MockOllamaModel("codellama:7b"))
            
            mock_client.return_value = client.create_client_mock()
            
            service = OllamaService(mock_config, mock_logger)
            model_manager = OllamaModelManager(service)
            conversation_manager = OllamaConversationManager()
            stream_handler = OllamaStreamHandler()
            
            return service, model_manager, conversation_manager, stream_handler
    
    async def test_end_to_end_text_generation(self, full_ollama_setup):
        """Test end-to-end text generation workflow."""
        service, model_manager, conversation_manager, stream_handler = full_ollama_setup
        
        # Connect to Ollama
        await service.connect()
        
        # Load available models
        await model_manager.load_available_models()
        
        # Create conversation
        conversation_id = await conversation_manager.create_conversation(
            system_prompt="You are a helpful assistant.",
            model="llama3.1:8b"
        )
        
        # Add user message
        await conversation_manager.add_message(
            conversation_id, "user", "What is artificial intelligence?"
        )
        
        # Generate response
        messages = await conversation_manager.get_conversation_history(conversation_id)
        response = await service.chat_completion(messages, "llama3.1:8b")
        
        # Add assistant response
        await conversation_manager.add_message(
            conversation_id, "assistant", response.content
        )
        
        assert response.success is True
        assert response.content is not None
        assert len(await conversation_manager.get_conversation_history(conversation_id)) == 2
    
    async def test_end_to_end_streaming(self, full_ollama_setup):
        """Test end-to-end streaming workflow."""
        service, model_manager, conversation_manager, stream_handler = full_ollama_setup
        
        # Connect to Ollama
        await service.connect()
        
        # Create conversation
        conversation_id = await conversation_manager.create_conversation(
            system_prompt="You are a creative writer.",
            model="llama3.1:8b"
        )
        
        # Set up streaming callback
        accumulated_content = []
        
        def stream_callback(chunk):
            accumulated_content.append(chunk.content)
        
        stream_id = "test_stream"
        stream_handler.register_stream_callback(stream_id, stream_callback)
        
        # Stream generation
        messages = [{"role": "user", "content": "Write a short story about AI"}]
        
        async for chunk in service.stream_chat(messages, "llama3.1:8b"):
            chunk.stream_id = stream_id
            await stream_handler.handle_stream_chunk(chunk)
            
            if chunk.done:
                break
        
        # Verify streaming worked
        assert len(accumulated_content) > 0
        final_content = stream_handler.get_stream_content(stream_id)
        assert final_content is not None
        assert len(final_content) > 0
    
    async def test_model_switching(self, full_ollama_setup):
        """Test switching between models."""
        service, model_manager, conversation_manager, stream_handler = full_ollama_setup
        
        # Connect and load models
        await service.connect()
        await model_manager.load_available_models()
        
        # Test text generation with general model
        text_model = model_manager.get_model_for_task("text_generation")
        response1 = await service.generate_text(
            "Explain quantum computing", text_model.name
        )
        
        # Test code generation with code model
        code_model = model_manager.get_model_for_task("code_generation")
        response2 = await service.generate_text(
            "Write a Python function to sort a list", code_model.name
        )
        
        assert response1.success is True
        assert response2.success is True
        assert response1.model == "llama3.1:8b"
        assert response2.model == "codellama:7b"
    
    async def test_error_recovery(self, full_ollama_setup):
        """Test error recovery mechanisms."""
        service, model_manager, conversation_manager, stream_handler = full_ollama_setup
        
        # Connect to Ollama
        await service.connect()
        
        # Test recovery from model not found
        try:
            await service.generate_text("Test", "non-existent-model")
            assert False, "Should have raised error"
        except OllamaModelNotFoundError:
            pass  # Expected
        
        # Service should still work after error
        response = await service.generate_text("Test", "llama3.1:8b")
        assert response.success is True
    
    async def test_concurrent_requests(self, full_ollama_setup):
        """Test handling concurrent requests."""
        service, model_manager, conversation_manager, stream_handler = full_ollama_setup
        
        # Connect to Ollama
        await service.connect()
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            task = service.generate_text(f"Question {i}", "llama3.1:8b")
            tasks.append(task)
        
        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        assert all(response.success for response in responses)
        assert len(responses) == 5