"""
AI processing features for the autonomous agent system.

This module provides specialized AI processing capabilities including
text summarization, content classification, email analysis, and research assistance.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from .ollama_service import OllamaService, ProcessingRequest, ProcessingResponse


class AIProcessor:
    """
    Specialized AI processor for common autonomous agent tasks.
    
    Provides high-level AI processing capabilities built on top of the
    Ollama service for text analysis, classification, and generation.
    """
    
    def __init__(self, ollama_service: OllamaService, logger: logging.Logger):
        """Initialize the AI processor.
        
        Args:
            ollama_service: Ollama service instance
            logger: Logger instance
        """
        self.ollama_service = ollama_service
        self.logger = logger
        
        # Task-specific prompts and configurations
        self.system_prompts = self._initialize_system_prompts()
        self.processing_configs = self._initialize_processing_configs()
    
    def _initialize_system_prompts(self) -> Dict[str, str]:
        """Initialize system prompts for different tasks."""
        return {
            "summarization": (
                "You are an expert at creating concise, accurate summaries. "
                "Focus on key points, main ideas, and important details. "
                "Keep summaries clear and well-structured."
            ),
            "classification": (
                "You are an expert classifier. Analyze the provided content "
                "and categorize it accurately based on the given criteria. "
                "Provide clear reasoning for your classification."
            ),
            "email_analysis": (
                "You are an expert email analyst. Analyze emails for content, "
                "sentiment, urgency, and actionable items. Provide structured "
                "analysis including key insights and recommendations."
            ),
            "research_analysis": (
                "You are a research expert. Analyze content for key insights, "
                "factual accuracy, relevance, and research value. Provide "
                "structured analysis with clear findings and recommendations."
            ),
            "content_extraction": (
                "You are an expert at extracting structured information from text. "
                "Focus on identifying key entities, relationships, and important "
                "data points. Present findings in a clear, organized manner."
            ),
            "question_answering": (
                "You are a knowledgeable assistant. Provide accurate, helpful "
                "answers based on the provided context. If information is "
                "insufficient, clearly state limitations."
            ),
            "code_analysis": (
                "You are an expert software engineer. Analyze code for quality, "
                "security issues, performance considerations, and best practices. "
                "Provide actionable feedback and suggestions."
            ),
            "writing_assistance": (
                "You are an expert writing assistant. Help improve text clarity, "
                "structure, grammar, and style while maintaining the original "
                "intent and voice."
            )
        }
    
    def _initialize_processing_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize processing configurations for different tasks."""
        return {
            "summarization": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 500,
                "model_preference": "text_generation"
            },
            "classification": {
                "temperature": 0.2,
                "top_p": 0.8,
                "max_tokens": 200,
                "model_preference": "text_generation"
            },
            "email_analysis": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 800,
                "model_preference": "analysis"
            },
            "research_analysis": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 1000,
                "model_preference": "analysis"
            },
            "content_extraction": {
                "temperature": 0.2,
                "top_p": 0.8,
                "max_tokens": 600,
                "model_preference": "text_generation"
            },
            "question_answering": {
                "temperature": 0.4,
                "top_p": 0.9,
                "max_tokens": 600,
                "model_preference": "text_generation"
            },
            "code_analysis": {
                "temperature": 0.2,
                "top_p": 0.85,
                "max_tokens": 800,
                "model_preference": "code_generation"
            },
            "writing_assistance": {
                "temperature": 0.4,
                "top_p": 0.9,
                "max_tokens": 800,
                "model_preference": "text_generation"
            }
        }
    
    async def summarize_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        focus_areas: Optional[List[str]] = None,
        summary_type: str = "general"
    ) -> ProcessingResponse:
        """Summarize text content.
        
        Args:
            text: Text content to summarize
            max_length: Maximum summary length in words
            focus_areas: Specific areas to focus on
            summary_type: Type of summary (general, bullet_points, executive)
            
        Returns:
            ProcessingResponse with summary
        """
        try:
            # Build prompt based on summary type and requirements
            prompt = self._build_summarization_prompt(
                text, max_length, focus_areas, summary_type
            )
            
            # Get optimal model
            model = self.ollama_service.model_manager.get_model_for_task("text_generation")
            model_name = model.name if model else self.ollama_service.default_model
            
            # Process request
            config = self.processing_configs["summarization"]
            response = await self.ollama_service.generate_text(
                prompt=prompt,
                model=model_name,
                system=self.system_prompts["summarization"],
                options={
                    "temperature": config["temperature"],
                    "top_p": config["top_p"]
                }
            )
            
            if response.success:
                self.logger.debug(f"Successfully summarized text ({len(text)} chars -> {len(response.content)} chars)")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Text summarization failed: {e}")
            return ProcessingResponse(
                content="",
                model=self.ollama_service.default_model,
                success=False,
                error=str(e)
            )
    
    def _build_summarization_prompt(
        self,
        text: str,
        max_length: Optional[int],
        focus_areas: Optional[List[str]],
        summary_type: str
    ) -> str:
        """Build prompt for text summarization."""
        prompt_parts = ["Please summarize the following text"]
        
        if summary_type == "bullet_points":
            prompt_parts.append("as a list of bullet points")
        elif summary_type == "executive":
            prompt_parts.append("as an executive summary")
        elif summary_type == "abstract":
            prompt_parts.append("as an academic abstract")
        
        if max_length:
            prompt_parts.append(f"in approximately {max_length} words")
        
        if focus_areas:
            areas_str = ", ".join(focus_areas)
            prompt_parts.append(f"focusing on: {areas_str}")
        
        prompt = " ".join(prompt_parts) + ":\n\n" + text
        return prompt
    
    async def classify_content(
        self,
        content: str,
        categories: List[str],
        classification_type: str = "single",
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Classify content into predefined categories.
        
        Args:
            content: Content to classify
            categories: List of possible categories
            classification_type: single, multi, or ranked
            confidence_threshold: Minimum confidence for classification
            
        Returns:
            Classification results with categories and confidence
        """
        try:
            prompt = self._build_classification_prompt(
                content, categories, classification_type
            )
            
            model = self.ollama_service.model_manager.get_model_for_task("text_generation")
            model_name = model.name if model else self.ollama_service.default_model
            
            config = self.processing_configs["classification"]
            response = await self.ollama_service.generate_text(
                prompt=prompt,
                model=model_name,
                system=self.system_prompts["classification"],
                options={
                    "temperature": config["temperature"],
                    "top_p": config["top_p"]
                }
            )
            
            if response.success:
                # Parse classification results
                results = self._parse_classification_response(
                    response.content, categories, classification_type
                )
                
                self.logger.debug(f"Successfully classified content: {results}")
                return results
            else:
                return {"error": response.error, "success": False}
                
        except Exception as e:
            self.logger.error(f"Content classification failed: {e}")
            return {"error": str(e), "success": False}
    
    def _build_classification_prompt(
        self,
        content: str,
        categories: List[str],
        classification_type: str
    ) -> str:
        """Build prompt for content classification."""
        categories_str = ", ".join(categories)
        
        if classification_type == "single":
            instruction = f"Classify this content into ONE of these categories: {categories_str}"
        elif classification_type == "multi":
            instruction = f"Classify this content into one or more of these categories: {categories_str}"
        else:  # ranked
            instruction = f"Rank these categories by relevance to the content: {categories_str}"
        
        prompt = f"{instruction}\n\nProvide your answer in JSON format with category names and confidence scores (0-1).\n\nContent:\n{content}"
        return prompt
    
    def _parse_classification_response(
        self,
        response: str,
        categories: List[str],
        classification_type: str
    ) -> Dict[str, Any]:
        """Parse classification response from AI model."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "classifications": result,
                    "type": classification_type,
                    "success": True
                }
            else:
                # Fallback: parse text response
                result = self._parse_text_classification(response, categories)
                return {
                    "classifications": result,
                    "type": classification_type,
                    "success": True
                }
        except Exception as e:
            return {
                "error": f"Failed to parse classification: {e}",
                "raw_response": response,
                "success": False
            }
    
    def _parse_text_classification(self, response: str, categories: List[str]) -> Dict[str, float]:
        """Parse text-based classification response."""
        results = {}
        response_lower = response.lower()
        
        for category in categories:
            if category.lower() in response_lower:
                # Simple confidence scoring based on presence
                results[category] = 0.8
        
        # If no categories found, assign to first category with low confidence
        if not results and categories:
            results[categories[0]] = 0.3
        
        return results
    
    async def analyze_email(self, email_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze email content for insights and recommendations.
        
        Args:
            email_content: Email data including subject, body, sender, etc.
            
        Returns:
            Analysis results with insights and recommendations
        """
        try:
            prompt = self._build_email_analysis_prompt(email_content)
            
            model = self.ollama_service.model_manager.get_model_for_task("analysis")
            model_name = model.name if model else self.ollama_service.default_model
            
            config = self.processing_configs["email_analysis"]
            response = await self.ollama_service.generate_text(
                prompt=prompt,
                model=model_name,
                system=self.system_prompts["email_analysis"],
                options={
                    "temperature": config["temperature"],
                    "top_p": config["top_p"]
                }
            )
            
            if response.success:
                analysis = self._parse_email_analysis(response.content)
                analysis["processing_time"] = response.processing_time
                self.logger.debug(f"Successfully analyzed email: {email_content.get('subject', 'No subject')}")
                return analysis
            else:
                return {"error": response.error, "success": False}
                
        except Exception as e:
            self.logger.error(f"Email analysis failed: {e}")
            return {"error": str(e), "success": False}
    
    def _build_email_analysis_prompt(self, email_content: Dict[str, Any]) -> str:
        """Build prompt for email analysis."""
        prompt_parts = ["Analyze this email and provide insights in JSON format including:"]
        prompt_parts.append("- sentiment (positive/negative/neutral)")
        prompt_parts.append("- urgency (high/medium/low)")
        prompt_parts.append("- category (work/personal/newsletter/spam/etc)")
        prompt_parts.append("- action_required (true/false)")
        prompt_parts.append("- key_points (list of main points)")
        prompt_parts.append("- suggested_response (brief suggestion)")
        prompt_parts.append("")
        
        # Format email content
        prompt_parts.append("Email details:")
        if "subject" in email_content:
            prompt_parts.append(f"Subject: {email_content['subject']}")
        if "sender" in email_content:
            prompt_parts.append(f"From: {email_content['sender']}")
        if "body" in email_content:
            prompt_parts.append(f"Body:\n{email_content['body']}")
        
        return "\n".join(prompt_parts)
    
    def _parse_email_analysis(self, response: str) -> Dict[str, Any]:
        """Parse email analysis response."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["success"] = True
                return result
            else:
                # Fallback parsing
                return {
                    "sentiment": "neutral",
                    "urgency": "medium",
                    "category": "unknown",
                    "action_required": False,
                    "key_points": ["Analysis parsing failed"],
                    "suggested_response": "Manual review needed",
                    "raw_response": response,
                    "success": False
                }
        except Exception as e:
            return {
                "error": f"Failed to parse email analysis: {e}",
                "raw_response": response,
                "success": False
            }
    
    async def analyze_research_content(
        self,
        content: str,
        research_questions: Optional[List[str]] = None,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze content for research insights.
        
        Args:
            content: Research content to analyze
            research_questions: Specific questions to address
            focus_areas: Areas to focus analysis on
            
        Returns:
            Research analysis with insights and findings
        """
        try:
            prompt = self._build_research_analysis_prompt(
                content, research_questions, focus_areas
            )
            
            model = self.ollama_service.model_manager.get_model_for_task("analysis")
            model_name = model.name if model else self.ollama_service.default_model
            
            config = self.processing_configs["research_analysis"]
            response = await self.ollama_service.generate_text(
                prompt=prompt,
                model=model_name,
                system=self.system_prompts["research_analysis"],
                options={
                    "temperature": config["temperature"],
                    "top_p": config["top_p"]
                }
            )
            
            if response.success:
                analysis = self._parse_research_analysis(response.content)
                analysis["processing_time"] = response.processing_time
                self.logger.debug(f"Successfully analyzed research content ({len(content)} chars)")
                return analysis
            else:
                return {"error": response.error, "success": False}
                
        except Exception as e:
            self.logger.error(f"Research analysis failed: {e}")
            return {"error": str(e), "success": False}
    
    def _build_research_analysis_prompt(
        self,
        content: str,
        research_questions: Optional[List[str]],
        focus_areas: Optional[List[str]]
    ) -> str:
        """Build prompt for research analysis."""
        prompt_parts = ["Analyze this research content and provide insights in JSON format including:"]
        prompt_parts.append("- key_findings (list of main discoveries)")
        prompt_parts.append("- methodology (description of approach used)")
        prompt_parts.append("- credibility_score (1-10 rating)")
        prompt_parts.append("- relevance_score (1-10 rating)")
        prompt_parts.append("- gaps_identified (list of research gaps)")
        prompt_parts.append("- recommendations (list of next steps)")
        prompt_parts.append("- sources_cited (number of references)")
        prompt_parts.append("")
        
        if research_questions:
            prompt_parts.append("Please specifically address these research questions:")
            for i, question in enumerate(research_questions, 1):
                prompt_parts.append(f"{i}. {question}")
            prompt_parts.append("")
        
        if focus_areas:
            prompt_parts.append(f"Focus your analysis on: {', '.join(focus_areas)}")
            prompt_parts.append("")
        
        prompt_parts.append("Content to analyze:")
        prompt_parts.append(content)
        
        return "\n".join(prompt_parts)
    
    def _parse_research_analysis(self, response: str) -> Dict[str, Any]:
        """Parse research analysis response."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["success"] = True
                return result
            else:
                return {
                    "key_findings": ["Analysis parsing failed"],
                    "methodology": "Unknown",
                    "credibility_score": 5,
                    "relevance_score": 5,
                    "gaps_identified": ["Analysis incomplete"],
                    "recommendations": ["Manual review needed"],
                    "sources_cited": 0,
                    "raw_response": response,
                    "success": False
                }
        except Exception as e:
            return {
                "error": f"Failed to parse research analysis: {e}",
                "raw_response": response,
                "success": False
            }
    
    async def extract_structured_data(
        self,
        text: str,
        data_types: List[str],
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """Extract structured data from unstructured text.
        
        Args:
            text: Text to extract data from
            data_types: Types of data to extract (entities, dates, urls, etc.)
            output_format: Output format (json, csv, structured)
            
        Returns:
            Extracted structured data
        """
        try:
            prompt = self._build_extraction_prompt(text, data_types, output_format)
            
            model = self.ollama_service.model_manager.get_model_for_task("text_generation")
            model_name = model.name if model else self.ollama_service.default_model
            
            config = self.processing_configs["content_extraction"]
            response = await self.ollama_service.generate_text(
                prompt=prompt,
                model=model_name,
                system=self.system_prompts["content_extraction"],
                options={
                    "temperature": config["temperature"],
                    "top_p": config["top_p"]
                }
            )
            
            if response.success:
                extracted = self._parse_extraction_response(response.content, output_format)
                extracted["processing_time"] = response.processing_time
                self.logger.debug(f"Successfully extracted data types: {data_types}")
                return extracted
            else:
                return {"error": response.error, "success": False}
                
        except Exception as e:
            self.logger.error(f"Data extraction failed: {e}")
            return {"error": str(e), "success": False}
    
    def _build_extraction_prompt(
        self,
        text: str,
        data_types: List[str],
        output_format: str
    ) -> str:
        """Build prompt for data extraction."""
        prompt_parts = [f"Extract the following types of data from the text in {output_format} format:"]
        
        for data_type in data_types:
            if data_type == "entities":
                prompt_parts.append("- Named entities (people, organizations, locations)")
            elif data_type == "dates":
                prompt_parts.append("- Dates and time references")
            elif data_type == "urls":
                prompt_parts.append("- URLs and web links")
            elif data_type == "emails":
                prompt_parts.append("- Email addresses")
            elif data_type == "phones":
                prompt_parts.append("- Phone numbers")
            elif data_type == "monetary":
                prompt_parts.append("- Monetary amounts and financial data")
            elif data_type == "keywords":
                prompt_parts.append("- Key terms and important concepts")
            else:
                prompt_parts.append(f"- {data_type}")
        
        prompt_parts.append("")
        prompt_parts.append("Text to analyze:")
        prompt_parts.append(text)
        
        return "\n".join(prompt_parts)
    
    def _parse_extraction_response(self, response: str, output_format: str) -> Dict[str, Any]:
        """Parse data extraction response."""
        try:
            if output_format == "json":
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    result["success"] = True
                    return result
            
            # Fallback: return raw response
            return {
                "extracted_data": response,
                "format": output_format,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"Failed to parse extraction: {e}",
                "raw_response": response,
                "success": False
            }
    
    async def generate_response_suggestions(
        self,
        context: str,
        response_type: str = "email",
        tone: str = "professional",
        max_length: int = 200
    ) -> List[str]:
        """Generate response suggestions for various contexts.
        
        Args:
            context: Context for the response
            response_type: Type of response (email, chat, comment, etc.)
            tone: Tone of response (professional, casual, friendly, etc.)
            max_length: Maximum length of each suggestion
            
        Returns:
            List of response suggestions
        """
        try:
            prompt = self._build_response_generation_prompt(
                context, response_type, tone, max_length
            )
            
            model = self.ollama_service.model_manager.get_model_for_task("text_generation")
            model_name = model.name if model else self.ollama_service.default_model
            
            config = self.processing_configs["writing_assistance"]
            response = await self.ollama_service.generate_text(
                prompt=prompt,
                model=model_name,
                system=self.system_prompts["writing_assistance"],
                options={
                    "temperature": config["temperature"],
                    "top_p": config["top_p"]
                }
            )
            
            if response.success:
                suggestions = self._parse_response_suggestions(response.content)
                self.logger.debug(f"Generated {len(suggestions)} response suggestions")
                return suggestions
            else:
                return [f"Error generating suggestions: {response.error}"]
                
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return [f"Error: {e}"]
    
    def _build_response_generation_prompt(
        self,
        context: str,
        response_type: str,
        tone: str,
        max_length: int
    ) -> str:
        """Build prompt for response generation."""
        prompt = f"""Generate 3 different {response_type} response suggestions for the following context.
Each response should be {tone} in tone and approximately {max_length} characters.

Format your response as numbered suggestions:
1. [First suggestion]
2. [Second suggestion]
3. [Third suggestion]

Context:
{context}"""
        return prompt
    
    def _parse_response_suggestions(self, response: str) -> List[str]:
        """Parse response suggestions from AI output."""
        suggestions = []
        
        # Look for numbered items
        pattern = r'^\d+\.\s*(.+)$'
        for line in response.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                suggestions.append(match.group(1))
        
        # If no numbered items found, split by double newlines
        if not suggestions:
            parts = response.split('\n\n')
            suggestions = [part.strip() for part in parts if part.strip()]
        
        # Limit to 5 suggestions
        return suggestions[:5]
    
    async def batch_process(
        self,
        items: List[Dict[str, Any]],
        processing_type: str,
        batch_size: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process multiple items in batches.
        
        Args:
            items: List of items to process
            processing_type: Type of processing to perform
            batch_size: Number of items to process concurrently
            **kwargs: Additional arguments for processing functions
            
        Returns:
            List of processing results
        """
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Create processing tasks
            tasks = []
            for item in batch:
                if processing_type == "summarize":
                    task = self.summarize_text(item.get("text", ""), **kwargs)
                elif processing_type == "classify":
                    task = self.classify_content(item.get("content", ""), **kwargs)
                elif processing_type == "analyze_email":
                    task = self.analyze_email(item)
                elif processing_type == "analyze_research":
                    task = self.analyze_research_content(item.get("content", ""), **kwargs)
                elif processing_type == "extract_data":
                    task = self.extract_structured_data(item.get("text", ""), **kwargs)
                else:
                    # Default to text generation
                    task = self.ollama_service.generate_text(
                        item.get("prompt", ""), **kwargs
                    )
                
                tasks.append(task)
            
            # Process batch
            try:
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                
                self.logger.debug(f"Processed batch {i//batch_size + 1} ({len(batch)} items)")
                
            except Exception as e:
                self.logger.error(f"Batch processing failed: {e}")
                # Add error results for failed batch
                error_result = {"error": str(e), "success": False}
                results.extend([error_result] * len(batch))
        
        return results