"""
Mock configurations for Ollama API testing.

This module provides comprehensive mocks for Ollama API interactions,
following the TDD approach and ensuring complete isolation from external services.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class MockOllamaModel:
    """Mock Ollama model with realistic data."""
    
    def __init__(self,
                 name: str = "llama3.1:8b",
                 size: int = 4661211808,
                 digest: str = "sha256:mock_digest",
                 modified_at: Optional[datetime] = None):
        
        self.name = name
        self.size = size
        self.digest = digest
        self.modified_at = modified_at or datetime.now(timezone.utc)
        self.family = self._extract_family()
        self.parameter_size = self._extract_parameter_size()
    
    def _extract_family(self) -> str:
        """Extract model family from name."""
        if "llama" in self.name.lower():
            return "llama"
        elif "mistral" in self.name.lower():
            return "mistral"
        elif "codellama" in self.name.lower():
            return "codellama"
        else:
            return "unknown"
    
    def _extract_parameter_size(self) -> str:
        """Extract parameter size from name."""
        if "7b" in self.name.lower():
            return "7B"
        elif "8b" in self.name.lower():
            return "8B"
        elif "13b" in self.name.lower():
            return "13B"
        elif "70b" in self.name.lower():
            return "70B"
        else:
            return "unknown"
    
    def to_ollama_format(self) -> Dict[str, Any]:
        """Convert to Ollama API format."""
        return {
            "name": self.name,
            "size": self.size,
            "digest": self.digest,
            "modified_at": self.modified_at.isoformat(),
            "details": {
                "family": self.family,
                "parameter_size": self.parameter_size,
                "quantization_level": "Q4_0",
            },
        }


class MockOllamaResponse:
    """Mock Ollama response with realistic data."""
    
    def __init__(self,
                 model: str = "llama3.1:8b",
                 response: str = "This is a mock response from Ollama",
                 done: bool = True,
                 context: Optional[List[int]] = None,
                 total_duration: int = 5000000000,
                 load_duration: int = 1000000000,
                 prompt_eval_count: int = 10,
                 prompt_eval_duration: int = 1000000000,
                 eval_count: int = 20,
                 eval_duration: int = 3000000000):
        
        self.model = model
        self.response = response
        self.done = done
        self.context = context or [1, 2, 3, 4, 5]
        self.total_duration = total_duration
        self.load_duration = load_duration
        self.prompt_eval_count = prompt_eval_count
        self.prompt_eval_duration = prompt_eval_duration
        self.eval_count = eval_count
        self.eval_duration = eval_duration
        self.created_at = datetime.now(timezone.utc)
    
    def to_ollama_format(self) -> Dict[str, Any]:
        """Convert to Ollama API format."""
        return {
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "response": self.response,
            "done": self.done,
            "context": self.context,
            "total_duration": self.total_duration,
            "load_duration": self.load_duration,
            "prompt_eval_count": self.prompt_eval_count,
            "prompt_eval_duration": self.prompt_eval_duration,
            "eval_count": self.eval_count,
            "eval_duration": self.eval_duration,
        }


class MockOllamaChatMessage:
    """Mock Ollama chat message."""
    
    def __init__(self,
                 role: str = "assistant",
                 content: str = "This is a mock chat response",
                 images: Optional[List[str]] = None):
        
        self.role = role
        self.content = content
        self.images = images or []
    
    def to_ollama_format(self) -> Dict[str, Any]:
        """Convert to Ollama API format."""
        message = {
            "role": self.role,
            "content": self.content,
        }
        if self.images:
            message["images"] = self.images
        return message


class MockOllamaChatResponse:
    """Mock Ollama chat response."""
    
    def __init__(self,
                 model: str = "llama3.1:8b",
                 message: Optional[MockOllamaChatMessage] = None,
                 done: bool = True,
                 total_duration: int = 5000000000,
                 load_duration: int = 1000000000,
                 prompt_eval_count: int = 10,
                 prompt_eval_duration: int = 1000000000,
                 eval_count: int = 20,
                 eval_duration: int = 3000000000):
        
        self.model = model
        self.message = message or MockOllamaChatMessage()
        self.done = done
        self.total_duration = total_duration
        self.load_duration = load_duration
        self.prompt_eval_count = prompt_eval_count
        self.prompt_eval_duration = prompt_eval_duration
        self.eval_count = eval_count
        self.eval_duration = eval_duration
        self.created_at = datetime.now(timezone.utc)
    
    def to_ollama_format(self) -> Dict[str, Any]:
        """Convert to Ollama API format."""
        return {
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "message": self.message.to_ollama_format(),
            "done": self.done,
            "total_duration": self.total_duration,
            "load_duration": self.load_duration,
            "prompt_eval_count": self.prompt_eval_count,
            "prompt_eval_duration": self.prompt_eval_duration,
            "eval_count": self.eval_count,
            "eval_duration": self.eval_duration,
        }


class MockOllamaClient:
    """Mock Ollama client with realistic behavior."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.models = []
        self.chat_history = []
        self.generation_history = []
        self.is_connected = True
        self.response_delay = 0.1  # Simulate response delay
    
    def add_model(self, model: MockOllamaModel) -> None:
        """Add a model to the mock client."""
        self.models.append(model)
    
    def create_client_mock(self) -> AsyncMock:
        """Create a mock client object."""
        client = AsyncMock()
        
        # Mock basic client methods
        client.list = AsyncMock(side_effect=self._list_models)
        client.show = AsyncMock(side_effect=self._show_model)
        client.generate = AsyncMock(side_effect=self._generate)
        client.chat = AsyncMock(side_effect=self._chat)
        client.pull = AsyncMock(side_effect=self._pull_model)
        client.push = AsyncMock(side_effect=self._push_model)
        client.create = AsyncMock(side_effect=self._create_model)
        client.delete = AsyncMock(side_effect=self._delete_model)
        client.copy = AsyncMock(side_effect=self._copy_model)
        client.embeddings = AsyncMock(side_effect=self._embeddings)
        
        # Mock streaming methods
        client.generate_stream = AsyncMock(side_effect=self._generate_stream)
        client.chat_stream = AsyncMock(side_effect=self._chat_stream)
        
        return client
    
    async def _list_models(self) -> Dict[str, Any]:
        """Mock list models method."""
        return {
            "models": [model.to_ollama_format() for model in self.models]
        }
    
    async def _show_model(self, name: str) -> Dict[str, Any]:
        """Mock show model method."""
        model = next((m for m in self.models if m.name == name), None)
        if model:
            return {
                "modelfile": f"# Mock modelfile for {name}",
                "parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                },
                "template": "{{ .System }}\n{{ .Prompt }}",
                "details": model.to_ollama_format()["details"],
            }
        raise Exception(f"Model {name} not found")
    
    async def _generate(self, 
                       model: str,
                       prompt: str,
                       system: Optional[str] = None,
                       template: Optional[str] = None,
                       context: Optional[List[int]] = None,
                       stream: bool = False,
                       raw: bool = False,
                       format: Optional[str] = None,
                       options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock generate method."""
        response = MockOllamaResponse(
            model=model,
            response=f"Generated response for prompt: {prompt[:50]}...",
            context=context or [1, 2, 3, 4, 5],
        )
        
        self.generation_history.append({
            "model": model,
            "prompt": prompt,
            "system": system,
            "response": response.response,
            "timestamp": datetime.now(timezone.utc),
        })
        
        return response.to_ollama_format()
    
    async def _chat(self,
                   model: str,
                   messages: List[Dict[str, Any]],
                   stream: bool = False,
                   format: Optional[str] = None,
                   options: Optional[Dict[str, Any]] = None,
                   template: Optional[str] = None,
                   keep_alive: Optional[str] = None) -> Dict[str, Any]:
        """Mock chat method."""
        last_message = messages[-1] if messages else {"content": "Hello"}
        
        response = MockOllamaChatResponse(
            model=model,
            message=MockOllamaChatMessage(
                role="assistant",
                content=f"Chat response to: {last_message.get('content', '')[:50]}...",
            ),
        )
        
        self.chat_history.append({
            "model": model,
            "messages": messages,
            "response": response.message.content,
            "timestamp": datetime.now(timezone.utc),
        })
        
        return response.to_ollama_format()
    
    async def _generate_stream(self, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Mock generate stream method."""
        response = await self._generate(**kwargs)
        
        # Simulate streaming by yielding chunks
        chunks = response["response"].split(" ")
        for i, chunk in enumerate(chunks):
            yield {
                "model": response["model"],
                "created_at": response["created_at"],
                "response": chunk + " " if i < len(chunks) - 1 else chunk,
                "done": i == len(chunks) - 1,
                "context": response["context"] if i == len(chunks) - 1 else None,
            }
    
    async def _chat_stream(self, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Mock chat stream method."""
        response = await self._chat(**kwargs)
        
        # Simulate streaming by yielding chunks
        content = response["message"]["content"]
        chunks = content.split(" ")
        
        for i, chunk in enumerate(chunks):
            yield {
                "model": response["model"],
                "created_at": response["created_at"],
                "message": {
                    "role": "assistant",
                    "content": chunk + " " if i < len(chunks) - 1 else chunk,
                },
                "done": i == len(chunks) - 1,
            }
    
    async def _pull_model(self, name: str, insecure: bool = False, stream: bool = False) -> Dict[str, Any]:
        """Mock pull model method."""
        # Simulate adding a new model
        model = MockOllamaModel(name=name)
        self.add_model(model)
        
        return {
            "status": "success",
            "digest": model.digest,
            "total": model.size,
        }
    
    async def _push_model(self, name: str, insecure: bool = False, stream: bool = False) -> Dict[str, Any]:
        """Mock push model method."""
        model = next((m for m in self.models if m.name == name), None)
        if model:
            return {
                "status": "success",
                "digest": model.digest,
                "total": model.size,
            }
        raise Exception(f"Model {name} not found")
    
    async def _create_model(self, name: str, modelfile: str, stream: bool = False) -> Dict[str, Any]:
        """Mock create model method."""
        model = MockOllamaModel(name=name)
        self.add_model(model)
        
        return {
            "status": "success",
            "digest": model.digest,
        }
    
    async def _delete_model(self, name: str) -> Dict[str, Any]:
        """Mock delete model method."""
        model = next((m for m in self.models if m.name == name), None)
        if model:
            self.models.remove(model)
            return {"status": "success"}
        raise Exception(f"Model {name} not found")
    
    async def _copy_model(self, source: str, destination: str) -> Dict[str, Any]:
        """Mock copy model method."""
        source_model = next((m for m in self.models if m.name == source), None)
        if source_model:
            new_model = MockOllamaModel(
                name=destination,
                size=source_model.size,
                digest=f"sha256:mock_digest_{destination}",
            )
            self.add_model(new_model)
            return {"status": "success"}
        raise Exception(f"Source model {source} not found")
    
    async def _embeddings(self, 
                         model: str, 
                         prompt: str, 
                         options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock embeddings method."""
        # Generate mock embeddings (384-dimensional vector)
        embedding = [0.1 * i for i in range(384)]
        
        return {
            "embedding": embedding,
            "model": model,
            "prompt": prompt,
        }


class MockOllamaHealthChecker:
    """Mock health checker for Ollama service."""
    
    def __init__(self, is_healthy: bool = True):
        self.is_healthy = is_healthy
        self.last_check = datetime.now(timezone.utc)
    
    async def check_health(self) -> Dict[str, Any]:
        """Check Ollama service health."""
        return {
            "status": "healthy" if self.is_healthy else "unhealthy",
            "timestamp": self.last_check.isoformat(),
            "version": "0.1.0",
            "models_loaded": 2,
        }
    
    def set_healthy(self, healthy: bool) -> None:
        """Set health status."""
        self.is_healthy = healthy
        self.last_check = datetime.now(timezone.utc)


# Pytest fixtures for Ollama mocks
@pytest.fixture
def mock_ollama_model() -> MockOllamaModel:
    """Provide a mock Ollama model."""
    return MockOllamaModel()


@pytest.fixture
def mock_ollama_response() -> MockOllamaResponse:
    """Provide a mock Ollama response."""
    return MockOllamaResponse()


@pytest.fixture
def mock_ollama_chat_message() -> MockOllamaChatMessage:
    """Provide a mock Ollama chat message."""
    return MockOllamaChatMessage()


@pytest.fixture
def mock_ollama_chat_response() -> MockOllamaChatResponse:
    """Provide a mock Ollama chat response."""
    return MockOllamaChatResponse()


@pytest.fixture
def mock_ollama_client() -> MockOllamaClient:
    """Provide a mock Ollama client."""
    client = MockOllamaClient()
    # Add some default models
    client.add_model(MockOllamaModel("llama3.1:8b", 4661211808))
    client.add_model(MockOllamaModel("llama3.1:70b", 39017093632))
    client.add_model(MockOllamaModel("codellama:7b", 3800000000))
    return client


@pytest.fixture
def mock_ollama_health_checker() -> MockOllamaHealthChecker:
    """Provide a mock Ollama health checker."""
    return MockOllamaHealthChecker()


# Context managers for patching Ollama API
class MockOllamaAPIContext:
    """Context manager for mocking Ollama API calls."""
    
    def __init__(self, client: MockOllamaClient):
        self.client = client
        self.patches = []
    
    def __enter__(self):
        # Mock ollama client
        ollama_patch = patch("ollama.AsyncClient")
        mock_ollama = ollama_patch.start()
        mock_ollama.return_value = self.client.create_client_mock()
        self.patches.append(ollama_patch)
        
        # Mock HTTP requests to Ollama
        requests_patch = patch("httpx.AsyncClient")
        mock_requests = requests_patch.start()
        mock_requests.return_value.__aenter__.return_value = AsyncMock()
        self.patches.append(requests_patch)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for patch_obj in self.patches:
            patch_obj.stop()


# Decorator for Ollama API mocking
def mock_ollama_api(client: Optional[MockOllamaClient] = None):
    """Decorator to mock Ollama API for a test function."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            mock_client = client or MockOllamaClient()
            with MockOllamaAPIContext(mock_client):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# Sample data generators
def generate_sample_models(count: int = 5) -> List[MockOllamaModel]:
    """Generate sample models for testing."""
    models = []
    model_names = [
        "llama3.1:8b",
        "llama3.1:70b",
        "codellama:7b",
        "mistral:7b",
        "vicuna:13b",
    ]
    
    for i in range(min(count, len(model_names))):
        model = MockOllamaModel(
            name=model_names[i],
            size=4000000000 + i * 1000000000,
            digest=f"sha256:mock_digest_{i:03d}",
        )
        models.append(model)
    
    return models


def generate_sample_chat_history(count: int = 5) -> List[Dict[str, Any]]:
    """Generate sample chat history for testing."""
    history = []
    for i in range(count):
        history.append({
            "model": "llama3.1:8b",
            "messages": [
                {"role": "user", "content": f"Question {i+1}: What is AI?"},
                {"role": "assistant", "content": f"Answer {i+1}: AI stands for Artificial Intelligence..."},
            ],
            "timestamp": datetime.now(timezone.utc),
        })
    return history


def generate_sample_prompts() -> List[str]:
    """Generate sample prompts for testing."""
    return [
        "Explain quantum computing in simple terms",
        "Write a Python function to calculate fibonacci numbers",
        "What are the benefits of renewable energy?",
        "How does machine learning work?",
        "Describe the process of photosynthesis",
    ]


def generate_sample_system_prompts() -> List[str]:
    """Generate sample system prompts for testing."""
    return [
        "You are a helpful assistant that provides accurate information.",
        "You are a code expert who helps with programming questions.",
        "You are a teacher who explains complex topics in simple terms.",
        "You are a scientist who provides factual information.",
        "You are a creative writer who helps with storytelling.",
    ]


def generate_sample_embeddings(dimension: int = 384) -> List[float]:
    """Generate sample embeddings for testing."""
    import math
    return [math.sin(i * 0.1) for i in range(dimension)]


# Mock error scenarios
class MockOllamaError(Exception):
    """Mock Ollama error for testing error handling."""
    
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def create_connection_error() -> MockOllamaError:
    """Create a mock connection error."""
    return MockOllamaError("Connection to Ollama failed", 503)


def create_model_not_found_error(model_name: str) -> MockOllamaError:
    """Create a mock model not found error."""
    return MockOllamaError(f"Model {model_name} not found", 404)


def create_timeout_error() -> MockOllamaError:
    """Create a mock timeout error."""
    return MockOllamaError("Request timed out", 408)


def create_rate_limit_error() -> MockOllamaError:
    """Create a mock rate limit error."""
    return MockOllamaError("Rate limit exceeded", 429)