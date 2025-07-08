"""
Service manager for coordinating AI services with the agent system.

This module provides integration between Ollama AI services and the existing
agent management infrastructure, enabling seamless AI-powered agent capabilities.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from src.config.manager import ConfigManager
from src.logging.manager import LoggingManager
from .ollama_service import OllamaService
from .ai_processing import AIProcessor


class ServiceManager:
    """
    Central manager for AI services integration with the agent system.
    
    Coordinates between Ollama services, AI processing capabilities,
    and the agent management infrastructure.
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logging_manager: LoggingManager,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the service manager.
        
        Args:
            config_manager: Configuration manager instance
            logging_manager: Logging manager instance
            logger: Optional logger instance
        """
        self.config_manager = config_manager
        self.logging_manager = logging_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Service instances
        self.ollama_service: Optional[OllamaService] = None
        self.ai_processor: Optional[AIProcessor] = None
        
        # Service status
        self.is_initialized = False
        self.services_healthy = False
        
        # Configuration
        self.ollama_config = self._load_ollama_config()
    
    def _load_ollama_config(self) -> Dict[str, Any]:
        """Load Ollama configuration from config manager."""
        return {
            "host": self.config_manager.get("ollama.host", "localhost"),
            "port": self.config_manager.get("ollama.port", 11434),
            "timeout": self.config_manager.get("ollama.timeout", 30),
            "default_model": self.config_manager.get("ollama.default_model", "llama3.1:8b"),
            "max_context_length": self.config_manager.get("ollama.max_context_length", 4096),
            "stream_enabled": self.config_manager.get("ollama.stream_enabled", True),
            "retry_attempts": self.config_manager.get("ollama.retry_attempts", 3),
            "retry_delay": self.config_manager.get("ollama.retry_delay", 1.0),
            "temperature": self.config_manager.get("ollama.temperature", 0.7),
            "top_p": self.config_manager.get("ollama.top_p", 0.9),
            "top_k": self.config_manager.get("ollama.top_k", 40),
        }
    
    async def initialize(self) -> None:
        """Initialize all AI services."""
        try:
            self.logger.info("Initializing AI services...")
            
            # Initialize Ollama service
            await self._initialize_ollama_service()
            
            # Initialize AI processor
            await self._initialize_ai_processor()
            
            # Perform health checks
            await self._perform_health_checks()
            
            self.is_initialized = True
            self.logger.info("AI services initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI services: {e}")
            self.is_initialized = False
            raise
    
    async def _initialize_ollama_service(self) -> None:
        """Initialize the Ollama service."""
        try:
            base_url = f"http://{self.ollama_config['host']}:{self.ollama_config['port']}"
            
            # Create mock config object for OllamaService
            class MockConfig:
                def get(self, key: str, default: Any = None) -> Any:
                    # Convert service manager config to ollama service format
                    key_map = {
                        "ollama.timeout": self.ollama_config["timeout"],
                        "ollama.default_model": self.ollama_config["default_model"],
                        "ollama.max_context_length": self.ollama_config["max_context_length"],
                        "ollama.stream_enabled": self.ollama_config["stream_enabled"],
                        "ollama.retry_attempts": self.ollama_config["retry_attempts"],
                        "ollama.retry_delay": self.ollama_config["retry_delay"],
                        "ollama.temperature": self.ollama_config["temperature"],
                        "ollama.top_p": self.ollama_config["top_p"],
                        "ollama.top_k": self.ollama_config["top_k"],
                    }
                    return key_map.get(key, default)
            
            self.ollama_service = OllamaService(
                config=MockConfig(),
                logger=self.logger,
                base_url=base_url
            )
            
            # Connect to Ollama
            await self.ollama_service.connect()
            
            self.logger.info(f"Ollama service initialized at {base_url}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama service: {e}")
            raise
    
    async def _initialize_ai_processor(self) -> None:
        """Initialize the AI processor."""
        try:
            if not self.ollama_service:
                raise RuntimeError("Ollama service must be initialized first")
            
            self.ai_processor = AIProcessor(
                ollama_service=self.ollama_service,
                logger=self.logger
            )
            
            self.logger.info("AI processor initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI processor: {e}")
            raise
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all services."""
        try:
            if self.ollama_service:
                health = await self.ollama_service.health_check()
                if health.get("status") == "healthy":
                    self.services_healthy = True
                    self.logger.info("All AI services are healthy")
                else:
                    self.services_healthy = False
                    self.logger.warning(f"Ollama service health check failed: {health}")
            else:
                self.services_healthy = False
                self.logger.error("Ollama service not initialized")
                
        except Exception as e:
            self.services_healthy = False
            self.logger.error(f"Health check failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown all AI services."""
        try:
            self.logger.info("Shutting down AI services...")
            
            if self.ollama_service:
                await self.ollama_service.disconnect()
            
            self.is_initialized = False
            self.services_healthy = False
            
            self.logger.info("AI services shut down successfully")
            
        except Exception as e:
            self.logger.error(f"Error during AI services shutdown: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all AI services.
        
        Returns:
            Status dictionary with service information
        """
        return {
            "initialized": self.is_initialized,
            "healthy": self.services_healthy,
            "ollama_service": {
                "available": self.ollama_service is not None,
                "connected": self.ollama_service.is_connected if self.ollama_service else False,
                "base_url": f"http://{self.ollama_config['host']}:{self.ollama_config['port']}",
                "default_model": self.ollama_config["default_model"]
            },
            "ai_processor": {
                "available": self.ai_processor is not None
            }
        }
    
    # AI Service Methods - Proxies to underlying services
    
    async def summarize_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Summarize text using AI processor.
        
        Args:
            text: Text to summarize
            **kwargs: Additional arguments for summarization
            
        Returns:
            Summarization result
        """
        if not self.ai_processor:
            raise RuntimeError("AI processor not initialized")
        
        response = await self.ai_processor.summarize_text(text, **kwargs)
        return {
            "content": response.content,
            "success": response.success,
            "processing_time": response.processing_time,
            "error": response.error
        }
    
    async def classify_content(self, content: str, categories: List[str], **kwargs) -> Dict[str, Any]:
        """Classify content using AI processor.
        
        Args:
            content: Content to classify
            categories: List of possible categories
            **kwargs: Additional arguments for classification
            
        Returns:
            Classification result
        """
        if not self.ai_processor:
            raise RuntimeError("AI processor not initialized")
        
        return await self.ai_processor.classify_content(content, categories, **kwargs)
    
    async def analyze_email(self, email_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze email content using AI processor.
        
        Args:
            email_content: Email data to analyze
            
        Returns:
            Email analysis result
        """
        if not self.ai_processor:
            raise RuntimeError("AI processor not initialized")
        
        return await self.ai_processor.analyze_email(email_content)
    
    async def analyze_research_content(
        self,
        content: str,
        research_questions: Optional[List[str]] = None,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze research content using AI processor.
        
        Args:
            content: Research content to analyze
            research_questions: Specific questions to address
            focus_areas: Areas to focus analysis on
            
        Returns:
            Research analysis result
        """
        if not self.ai_processor:
            raise RuntimeError("AI processor not initialized")
        
        return await self.ai_processor.analyze_research_content(
            content, research_questions, focus_areas
        )
    
    async def extract_structured_data(
        self,
        text: str,
        data_types: List[str],
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """Extract structured data from text using AI processor.
        
        Args:
            text: Text to extract data from
            data_types: Types of data to extract
            output_format: Output format
            
        Returns:
            Data extraction result
        """
        if not self.ai_processor:
            raise RuntimeError("AI processor not initialized")
        
        return await self.ai_processor.extract_structured_data(
            text, data_types, output_format
        )
    
    async def generate_response_suggestions(
        self,
        context: str,
        response_type: str = "email",
        tone: str = "professional",
        max_length: int = 200
    ) -> List[str]:
        """Generate response suggestions using AI processor.
        
        Args:
            context: Context for the response
            response_type: Type of response
            tone: Tone of response
            max_length: Maximum length of each suggestion
            
        Returns:
            List of response suggestions
        """
        if not self.ai_processor:
            raise RuntimeError("AI processor not initialized")
        
        return await self.ai_processor.generate_response_suggestions(
            context, response_type, tone, max_length
        )
    
    async def chat_with_ai(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Chat with AI using Ollama service.
        
        Args:
            message: User message
            conversation_id: Optional conversation ID for context
            model: Optional model to use
            system_prompt: Optional system prompt
            
        Returns:
            Chat response
        """
        if not self.ollama_service:
            raise RuntimeError("Ollama service not initialized")
        
        try:
            # Handle conversation context
            if conversation_id:
                # Add user message to conversation
                await self.ollama_service.conversation_manager.add_message(
                    conversation_id, "user", message
                )
                
                # Get conversation history
                messages = await self.ollama_service.conversation_manager.get_conversation_history(
                    conversation_id
                )
                
                # Generate response using chat completion
                response = await self.ollama_service.chat_completion(
                    messages=messages,
                    model=model
                )
                
                # Add assistant response to conversation
                if response.success:
                    await self.ollama_service.conversation_manager.add_message(
                        conversation_id, "assistant", response.content
                    )
            else:
                # Direct text generation without conversation context
                response = await self.ollama_service.generate_text(
                    prompt=message,
                    model=model,
                    system=system_prompt
                )
            
            return {
                "content": response.content,
                "success": response.success,
                "model": response.model,
                "processing_time": response.processing_time,
                "conversation_id": conversation_id,
                "error": response.error
            }
            
        except Exception as e:
            self.logger.error(f"Chat with AI failed: {e}")
            return {
                "content": "",
                "success": False,
                "error": str(e),
                "conversation_id": conversation_id
            }
    
    async def create_conversation(
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """Create a new conversation context.
        
        Args:
            system_prompt: Optional system prompt for the conversation
            model: Optional model to use for the conversation
            
        Returns:
            Conversation ID
        """
        if not self.ollama_service:
            raise RuntimeError("Ollama service not initialized")
        
        return await self.ollama_service.conversation_manager.create_conversation(
            system_prompt=system_prompt,
            model=model or self.ollama_config["default_model"]
        )
    
    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation context.
        
        Args:
            conversation_id: ID of the conversation to delete
        """
        if not self.ollama_service:
            raise RuntimeError("Ollama service not initialized")
        
        await self.ollama_service.conversation_manager.delete_conversation(conversation_id)
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List available AI models.
        
        Returns:
            List of available models with their information
        """
        if not self.ollama_service:
            raise RuntimeError("Ollama service not initialized")
        
        try:
            models = await self.ollama_service.list_models()
            return [
                {
                    "name": model.name,
                    "family": model.family,
                    "parameter_size": model.parameter_size,
                    "size": model.size,
                    "capabilities": model.capabilities
                }
                for model in models
            ]
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    async def get_optimal_model_for_task(self, task_type: str) -> Optional[str]:
        """Get the optimal model for a specific task.
        
        Args:
            task_type: Type of task (e.g., 'text_generation', 'code_generation')
            
        Returns:
            Model name or None if no suitable model found
        """
        if not self.ollama_service:
            return None
        
        try:
            model = self.ollama_service.model_manager.get_model_for_task(task_type)
            return model.name if model else None
        except Exception as e:
            self.logger.error(f"Failed to get optimal model for task {task_type}: {e}")
            return None
    
    async def batch_process_items(
        self,
        items: List[Dict[str, Any]],
        processing_type: str,
        batch_size: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process multiple items in batches using AI processor.
        
        Args:
            items: List of items to process
            processing_type: Type of processing to perform
            batch_size: Number of items to process concurrently
            **kwargs: Additional arguments for processing
            
        Returns:
            List of processing results
        """
        if not self.ai_processor:
            raise RuntimeError("AI processor not initialized")
        
        return await self.ai_processor.batch_process(
            items, processing_type, batch_size, **kwargs
        )
    
    def register_with_agent_manager(self, agent_manager: Any) -> None:
        """Register AI services with the agent manager.
        
        Args:
            agent_manager: Agent manager instance to register with
        """
        try:
            # Register service manager as a service dependency
            if hasattr(agent_manager, 'register_service'):
                agent_manager.register_service('ai_services', self)
                self.logger.info("AI services registered with agent manager")
            else:
                self.logger.warning("Agent manager does not support service registration")
                
        except Exception as e:
            self.logger.error(f"Failed to register with agent manager: {e}")
    
    async def handle_agent_ai_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle AI processing requests from agents.
        
        Args:
            request: AI processing request from an agent
            
        Returns:
            Processing result
        """
        try:
            request_type = request.get("type")
            
            if request_type == "summarize":
                return await self.summarize_text(
                    request["text"],
                    **request.get("options", {})
                )
            elif request_type == "classify":
                return await self.classify_content(
                    request["content"],
                    request["categories"],
                    **request.get("options", {})
                )
            elif request_type == "analyze_email":
                return await self.analyze_email(request["email_content"])
            elif request_type == "analyze_research":
                return await self.analyze_research_content(
                    request["content"],
                    request.get("research_questions"),
                    request.get("focus_areas")
                )
            elif request_type == "extract_data":
                return await self.extract_structured_data(
                    request["text"],
                    request["data_types"],
                    request.get("output_format", "json")
                )
            elif request_type == "generate_responses":
                suggestions = await self.generate_response_suggestions(
                    request["context"],
                    request.get("response_type", "email"),
                    request.get("tone", "professional"),
                    request.get("max_length", 200)
                )
                return {"suggestions": suggestions, "success": True}
            elif request_type == "chat":
                return await self.chat_with_ai(
                    request["message"],
                    request.get("conversation_id"),
                    request.get("model"),
                    request.get("system_prompt")
                )
            else:
                return {
                    "error": f"Unknown request type: {request_type}",
                    "success": False
                }
                
        except Exception as e:
            self.logger.error(f"AI request handling failed: {e}")
            return {
                "error": str(e),
                "success": False
            }