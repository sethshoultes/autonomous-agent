"""
Services module for the Autonomous Agent System.

This module contains all service implementations and related functionality.
"""

from typing import List

# Ollama integration services (conditional import)
try:
    from .ollama_service import (
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
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    # Create placeholder classes for when Ollama is not available
    OllamaService = None
    OllamaModelManager = None
    OllamaConversationManager = None
    OllamaStreamHandler = None
    OllamaError = Exception
    OllamaConnectionError = Exception
    OllamaModelNotFoundError = Exception
    OllamaTimeoutError = Exception
    OllamaRateLimitError = Exception
    ModelInfo = None
    ConversationContext = None
    StreamingResponse = None
    ProcessingRequest = None
    ProcessingResponse = None

# AI processing features
from .ai_processing import AIProcessor

# Service management
from .service_manager import ServiceManager

# AI caching for performance
from .ai_cache import AICache, CachedAIProcessor

__all__: List[str] = [
    # Ollama services
    "OllamaService",
    "OllamaModelManager",
    "OllamaConversationManager",
    "OllamaStreamHandler",
    # Ollama exceptions
    "OllamaError",
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "OllamaTimeoutError",
    "OllamaRateLimitError",
    # Ollama data models
    "ModelInfo",
    "ConversationContext",
    "StreamingResponse",
    "ProcessingRequest",
    "ProcessingResponse",
    # AI processing
    "AIProcessor",
    # Service management
    "ServiceManager",
    # AI caching
    "AICache",
    "CachedAIProcessor",
]
