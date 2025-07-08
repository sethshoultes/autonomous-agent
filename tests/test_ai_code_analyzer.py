"""
Test suite for the AI Code Analyzer implementation.

This test suite covers AI-powered code analysis functionality:
- Code quality analysis
- Security vulnerability detection
- Pull request review automation
- Performance optimization suggestions
- Code documentation generation
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List, Optional

from src.services.ai_code_analyzer import (
    AICodeAnalyzer,
    CodeAnalysisResult,
    SecurityAnalysisResult,
    PullRequestReviewResult,
    DocumentationResult,
    AIAnalysisError,
    CodeQualityScore,
    SecurityIssue,
    PerformanceIssue,
    StyleIssue,
    ReviewComment,
)
from src.services.ollama_service import OllamaService
from tests.mocks.ollama_mocks import MockOllamaService


class TestAICodeAnalyzer:
    """Test suite for the AI Code Analyzer."""
    
    @pytest.fixture
    def mock_config(self):
        """Provide a mock configuration."""
        return {
            "model": "codellama:7b",
            "temperature": 0.2,
            "max_context_length": 8192,
            "analysis_timeout": 60,
            "confidence_threshold": 0.7,
            "security_enabled": True,
            "performance_enabled": True,
            "style_enabled": True,
            "documentation_enabled": True,
        }
    
    @pytest.fixture
    def mock_logger(self):
        """Provide a mock logger."""
        return MagicMock()
    
    @pytest.fixture
    def mock_ollama_service(self):
        """Provide a mock Ollama service."""
        return MockOllamaService()
    
    @pytest.fixture
    def ai_analyzer(self, mock_config, mock_logger, mock_ollama_service):
        """Provide an AI Code Analyzer instance."""
        with patch('src.services.ai_code_analyzer.OllamaService', return_value=mock_ollama_service):
            analyzer = AICodeAnalyzer(mock_config, mock_logger)
            analyzer.ollama_service = mock_ollama_service
            return analyzer
    
    @pytest.fixture
    def sample_python_code(self):
        """Provide sample Python code for testing."""
        return """
def vulnerable_function(user_input):
    # This function has security vulnerabilities
    import os
    command = "ls " + user_input
    result = os.system(command)  # Command injection vulnerability
    
    # SQL injection vulnerability
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    
    # Performance issue - inefficient loop
    numbers = []
    for i in range(1000000):
        numbers.append(i * 2)
    
    return result

def good_function(param):
    '''This function is well-written'''
    if not param:
        return None
    
    result = param.strip().lower()
    return result

class ExampleClass:
    def __init__(self, name):
        self.name = name
        self._private_attr = "private"
    
    def public_method(self):
        return f"Hello, {self.name}!"
    
    def _private_method(self):
        return self._private_attr
        """
    
    @pytest.fixture
    def sample_javascript_code(self):
        """Provide sample JavaScript code for testing."""
        return """
function vulnerableFunction(userInput) {
    // XSS vulnerability
    document.getElementById('output').innerHTML = userInput;
    
    // Prototype pollution vulnerability
    let obj = {};
    obj[userInput] = 'value';
    
    // Performance issue - inefficient DOM manipulation
    for (let i = 0; i < 1000; i++) {
        document.createElement('div');
    }
    
    return obj;
}

function goodFunction(param) {
    if (!param) return null;
    
    // Safe DOM manipulation
    const element = document.createElement('div');
    element.textContent = param;
    
    return element;
}

class ExampleClass {
    constructor(name) {
        this.name = name;
    }
    
    getName() {
        return this.name;
    }
}
        """
    
    @pytest.fixture
    def sample_pull_request_data(self):
        """Provide sample pull request data for testing."""
        return {
            "number": 123,
            "title": "Add new authentication system",
            "body": "This PR implements a new authentication system with improved security.",
            "files": [
                {
                    "filename": "auth.py",
                    "patch": """@@ -1,10 +1,15 @@
 import hashlib
 import secrets
+import bcrypt
 
 def authenticate_user(username, password):
-    # Old insecure method
-    return password == "admin123"
+    # New secure method
+    stored_hash = get_stored_hash(username)
+    return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
+
+def get_stored_hash(username):
+    # Retrieve stored hash from database
+    return database.get_user_hash(username)
 
 def generate_token():
     return secrets.token_urlsafe(32)""",
                    "additions": 8,
                    "deletions": 2,
                    "changes": 10
                },
                {
                    "filename": "config.py",
                    "patch": """@@ -5,6 +5,7 @@
 
 # Security configuration
 SECRET_KEY = "your-secret-key-here"
+BCRYPT_ROUNDS = 12
 
 # Database configuration
 DATABASE_URL = "sqlite:///app.db\"""",
                    "additions": 1,
                    "deletions": 0,
                    "changes": 1
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_ai_analyzer_initialization(self, ai_analyzer, mock_config):
        """Test AI Code Analyzer initialization."""
        assert ai_analyzer.config == mock_config
        assert ai_analyzer.model == "codellama:7b"
        assert ai_analyzer.temperature == 0.2
        assert ai_analyzer.max_context_length == 8192
        assert ai_analyzer.confidence_threshold == 0.7
        assert ai_analyzer.security_enabled is True
        assert ai_analyzer.performance_enabled is True
        assert ai_analyzer.style_enabled is True
    
    @pytest.mark.asyncio
    async def test_analyze_code_python(self, ai_analyzer, sample_python_code):
        """Test analyzing Python code."""
        # Mock Ollama response
        ai_analyzer.ollama_service.generate_text = AsyncMock(return_value=type('Response', (), {
            'content': '''
            {
                "overall_score": 6.5,
                "security_issues": [
                    {
                        "type": "command_injection",
                        "severity": "high",
                        "line": 5,
                        "description": "Command injection vulnerability in os.system() call",
                        "recommendation": "Use subprocess with shell=False instead",
                        "confidence": 0.95
                    },
                    {
                        "type": "sql_injection", 
                        "severity": "high",
                        "line": 8,
                        "description": "SQL injection vulnerability in query construction",
                        "recommendation": "Use parameterized queries or ORM",
                        "confidence": 0.92
                    }
                ],
                "performance_issues": [
                    {
                        "type": "inefficient_loop",
                        "severity": "medium",
                        "line": 11,
                        "description": "Inefficient list building in loop",
                        "recommendation": "Use list comprehension or numpy for better performance",
                        "confidence": 0.85
                    }
                ],
                "style_issues": [
                    {
                        "type": "missing_docstring",
                        "severity": "low",
                        "line": 2,
                        "description": "Function missing docstring",
                        "recommendation": "Add docstring to document function purpose",
                        "confidence": 0.8
                    }
                ],
                "best_practices": [
                    "Use type hints for better code documentation",
                    "Consider using logging instead of print statements",
                    "Implement proper error handling"
                ]
            }
            ''',
            'success': True
        })())
        
        # Analyze the code
        result = await ai_analyzer.analyze_code(sample_python_code, "python")
        
        # Verify the analysis
        assert isinstance(result, CodeAnalysisResult)
        assert result.language == "python"
        assert result.overall_score == 6.5
        assert len(result.security_issues) == 2
        assert len(result.performance_issues) == 1
        assert len(result.style_issues) == 1
        assert len(result.best_practices) == 3
        
        # Verify security issues
        assert result.security_issues[0].type == "command_injection"
        assert result.security_issues[0].severity == "high"
        assert result.security_issues[0].line == 5
        assert result.security_issues[0].confidence == 0.95
        
        assert result.security_issues[1].type == "sql_injection"
        assert result.security_issues[1].severity == "high"
        assert result.security_issues[1].line == 8
        assert result.security_issues[1].confidence == 0.92
        
        # Verify performance issues
        assert result.performance_issues[0].type == "inefficient_loop"
        assert result.performance_issues[0].severity == "medium"
        assert result.performance_issues[0].line == 11
        assert result.performance_issues[0].confidence == 0.85
        
        # Verify style issues
        assert result.style_issues[0].type == "missing_docstring"
        assert result.style_issues[0].severity == "low"
        assert result.style_issues[0].line == 2
        assert result.style_issues[0].confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_analyze_code_javascript(self, ai_analyzer, sample_javascript_code):
        """Test analyzing JavaScript code."""
        # Mock Ollama response
        ai_analyzer.ollama_service.generate_text = AsyncMock(return_value=type('Response', (), {
            'content': '''
            {
                "overall_score": 5.5,
                "security_issues": [
                    {
                        "type": "xss_vulnerability",
                        "severity": "high",
                        "line": 3,
                        "description": "XSS vulnerability in innerHTML assignment",
                        "recommendation": "Use textContent or sanitize input",
                        "confidence": 0.9
                    },
                    {
                        "type": "prototype_pollution",
                        "severity": "medium",
                        "line": 7,
                        "description": "Potential prototype pollution vulnerability",
                        "recommendation": "Validate object keys before assignment",
                        "confidence": 0.8
                    }
                ],
                "performance_issues": [
                    {
                        "type": "inefficient_dom_manipulation",
                        "severity": "medium",
                        "line": 10,
                        "description": "Inefficient DOM manipulation in loop",
                        "recommendation": "Use document fragments or batch DOM operations",
                        "confidence": 0.85
                    }
                ],
                "style_issues": [
                    {
                        "type": "missing_jsdoc",
                        "severity": "low",
                        "line": 1,
                        "description": "Function missing JSDoc documentation",
                        "recommendation": "Add JSDoc comments for better documentation",
                        "confidence": 0.75
                    }
                ],
                "best_practices": [
                    "Use const/let instead of var",
                    "Implement proper error handling",
                    "Consider using TypeScript for better type safety"
                ]
            }
            ''',
            'success': True
        })())
        
        # Analyze the code
        result = await ai_analyzer.analyze_code(sample_javascript_code, "javascript")
        
        # Verify the analysis
        assert isinstance(result, CodeAnalysisResult)
        assert result.language == "javascript"
        assert result.overall_score == 5.5
        assert len(result.security_issues) == 2
        assert len(result.performance_issues) == 1
        assert len(result.style_issues) == 1
        
        # Verify security issues
        assert result.security_issues[0].type == "xss_vulnerability"
        assert result.security_issues[0].severity == "high"
        assert result.security_issues[1].type == "prototype_pollution"
        assert result.security_issues[1].severity == "medium"
    
    @pytest.mark.asyncio
    async def test_detect_vulnerabilities(self, ai_analyzer, sample_python_code):
        """Test vulnerability detection."""
        # Mock Ollama response
        ai_analyzer.ollama_service.generate_text = AsyncMock(return_value=type('Response', (), {
            'content': '''
            {
                "scan_id": "vuln_scan_123",
                "vulnerabilities": [
                    {
                        "id": "CWE-78",
                        "type": "command_injection",
                        "severity": "critical",
                        "line": 5,
                        "description": "Command injection via os.system() call",
                        "impact": "Arbitrary command execution",
                        "remediation": "Use subprocess with shell=False and input validation",
                        "confidence": 0.95,
                        "cvss_score": 9.8
                    },
                    {
                        "id": "CWE-89",
                        "type": "sql_injection",
                        "severity": "high",
                        "line": 8,
                        "description": "SQL injection in query construction",
                        "impact": "Data breach, data manipulation",
                        "remediation": "Use parameterized queries or prepared statements",
                        "confidence": 0.92,
                        "cvss_score": 8.5
                    }
                ],
                "summary": {
                    "total_vulnerabilities": 2,
                    "critical": 1,
                    "high": 1,
                    "medium": 0,
                    "low": 0,
                    "risk_score": 9.2
                }
            }
            ''',
            'success': True
        })())
        
        # Detect vulnerabilities
        result = await ai_analyzer.detect_vulnerabilities(sample_python_code, "python")
        
        # Verify the results
        assert isinstance(result, SecurityAnalysisResult)
        assert result.scan_id == "vuln_scan_123"
        assert len(result.vulnerabilities) == 2
        assert result.total_vulnerabilities == 2
        assert result.critical_vulnerabilities == 1
        assert result.high_vulnerabilities == 1
        assert result.risk_score == 9.2
        
        # Verify individual vulnerabilities
        vuln1 = result.vulnerabilities[0]
        assert vuln1.id == "CWE-78"
        assert vuln1.type == "command_injection"
        assert vuln1.severity == "critical"
        assert vuln1.line == 5
        assert vuln1.cvss_score == 9.8
        assert vuln1.confidence == 0.95
        
        vuln2 = result.vulnerabilities[1]
        assert vuln2.id == "CWE-89"
        assert vuln2.type == "sql_injection"
        assert vuln2.severity == "high"
        assert vuln2.line == 8
        assert vuln2.cvss_score == 8.5
        assert vuln2.confidence == 0.92
    
    @pytest.mark.asyncio
    async def test_review_pull_request(self, ai_analyzer, sample_pull_request_data):
        """Test pull request review."""
        # Mock Ollama response
        ai_analyzer.ollama_service.generate_text = AsyncMock(return_value=type('Response', (), {
            'content': '''
            {
                "review_id": "pr_review_456",
                "overall_assessment": "approve",
                "summary": "Good security improvements with proper password hashing implementation",
                "score": 8.5,
                "comments": [
                    {
                        "file": "auth.py",
                        "line": 6,
                        "type": "improvement",
                        "message": "Excellent switch to bcrypt for password hashing",
                        "confidence": 0.9
                    },
                    {
                        "file": "auth.py",
                        "line": 12,
                        "type": "suggestion",
                        "message": "Consider adding error handling for database operations",
                        "confidence": 0.8
                    },
                    {
                        "file": "config.py",
                        "line": 8,
                        "type": "security",
                        "message": "Good addition of BCRYPT_ROUNDS configuration",
                        "confidence": 0.85
                    }
                ],
                "security_improvements": [
                    "Replaced plain text password comparison with bcrypt hashing",
                    "Added proper salt rounds configuration"
                ],
                "potential_issues": [
                    "Missing error handling for database operations",
                    "Consider adding rate limiting for authentication attempts"
                ],
                "testing_suggestions": [
                    "Add unit tests for the new authentication functions",
                    "Test with invalid credentials",
                    "Verify bcrypt performance with different round values"
                ]
            }
            ''',
            'success': True
        })())
        
        # Review the pull request
        result = await ai_analyzer.review_pull_request(sample_pull_request_data)
        
        # Verify the results
        assert isinstance(result, PullRequestReviewResult)
        assert result.review_id == "pr_review_456"
        assert result.overall_assessment == "approve"
        assert result.summary == "Good security improvements with proper password hashing implementation"
        assert result.score == 8.5
        assert len(result.comments) == 3
        assert len(result.security_improvements) == 2
        assert len(result.potential_issues) == 2
        assert len(result.testing_suggestions) == 3
        
        # Verify comments
        comment1 = result.comments[0]
        assert comment1.file == "auth.py"
        assert comment1.line == 6
        assert comment1.type == "improvement"
        assert comment1.confidence == 0.9
        
        comment2 = result.comments[1]
        assert comment2.file == "auth.py"
        assert comment2.line == 12
        assert comment2.type == "suggestion"
        assert comment2.confidence == 0.8
        
        comment3 = result.comments[2]
        assert comment3.file == "config.py"
        assert comment3.line == 8
        assert comment3.type == "security"
        assert comment3.confidence == 0.85
    
    @pytest.mark.asyncio
    async def test_suggest_improvements(self, ai_analyzer, sample_python_code):
        """Test improvement suggestions."""
        # Mock Ollama response
        ai_analyzer.ollama_service.generate_text = AsyncMock(return_value=type('Response', (), {
            'content': '''
            {
                "suggestions": [
                    {
                        "category": "security",
                        "priority": "high",
                        "description": "Replace os.system() with subprocess.run() for command execution",
                        "example": "subprocess.run(['ls', user_input], capture_output=True, text=True)",
                        "impact": "Prevents command injection vulnerabilities"
                    },
                    {
                        "category": "performance",
                        "priority": "medium",
                        "description": "Use list comprehension instead of append() in loop",
                        "example": "numbers = [i * 2 for i in range(1000000)]",
                        "impact": "Improves performance and readability"
                    },
                    {
                        "category": "style",
                        "priority": "low",
                        "description": "Add docstrings to all functions",
                        "example": "def vulnerable_function(user_input):\\n    '''Process user input safely.'''",
                        "impact": "Improves code documentation and maintainability"
                    }
                ],
                "best_practices": [
                    "Use type hints for better code documentation",
                    "Implement proper error handling with try-except blocks",
                    "Consider using logging instead of print statements",
                    "Follow PEP 8 style guidelines"
                ]
            }
            ''',
            'success': True
        })())
        
        # Get improvement suggestions
        result = await ai_analyzer.suggest_improvements(sample_python_code, "python")
        
        # Verify the results
        assert "suggestions" in result
        assert "best_practices" in result
        assert len(result["suggestions"]) == 3
        assert len(result["best_practices"]) == 4
        
        # Verify suggestions
        security_suggestion = result["suggestions"][0]
        assert security_suggestion["category"] == "security"
        assert security_suggestion["priority"] == "high"
        assert "os.system()" in security_suggestion["description"]
        assert "subprocess.run()" in security_suggestion["description"]
        
        performance_suggestion = result["suggestions"][1]
        assert performance_suggestion["category"] == "performance"
        assert performance_suggestion["priority"] == "medium"
        assert "list comprehension" in performance_suggestion["description"]
        
        style_suggestion = result["suggestions"][2]
        assert style_suggestion["category"] == "style"
        assert style_suggestion["priority"] == "low"
        assert "docstrings" in style_suggestion["description"]
    
    @pytest.mark.asyncio
    async def test_generate_documentation(self, ai_analyzer, sample_python_code):
        """Test documentation generation."""
        # Mock Ollama response
        ai_analyzer.ollama_service.generate_text = AsyncMock(return_value=type('Response', (), {
            'content': '''
            {
                "documentation": {
                    "functions": [
                        {
                            "name": "vulnerable_function",
                            "line": 2,
                            "description": "Processes user input and executes commands",
                            "parameters": [
                                {
                                    "name": "user_input",
                                    "type": "str",
                                    "description": "Input provided by the user"
                                }
                            ],
                            "returns": {
                                "type": "int",
                                "description": "Command execution result"
                            },
                            "raises": [],
                            "examples": [
                                "result = vulnerable_function('test')"
                            ],
                            "notes": [
                                "This function has security vulnerabilities and should be refactored"
                            ]
                        },
                        {
                            "name": "good_function",
                            "line": 18,
                            "description": "Safely processes a parameter",
                            "parameters": [
                                {
                                    "name": "param",
                                    "type": "str",
                                    "description": "Parameter to process"
                                }
                            ],
                            "returns": {
                                "type": "str or None",
                                "description": "Processed parameter or None if input is invalid"
                            },
                            "raises": [],
                            "examples": [
                                "result = good_function('  Hello World  ')"
                            ]
                        }
                    ],
                    "classes": [
                        {
                            "name": "ExampleClass",
                            "line": 25,
                            "description": "Example class demonstrating basic functionality",
                            "methods": [
                                {
                                    "name": "__init__",
                                    "description": "Initialize the class with a name",
                                    "parameters": [
                                        {
                                            "name": "name",
                                            "type": "str",
                                            "description": "Name for the instance"
                                        }
                                    ]
                                },
                                {
                                    "name": "public_method",
                                    "description": "Public method that returns a greeting",
                                    "returns": {
                                        "type": "str",
                                        "description": "Greeting message"
                                    }
                                }
                            ]
                        }
                    ]
                },
                "markdown": "# Code Documentation\\n\\n## Functions\\n\\n### vulnerable_function\\n\\nProcesses user input and executes commands.\\n\\n**Parameters:**\\n- `user_input` (str): Input provided by the user\\n\\n**Returns:**\\n- int: Command execution result\\n\\n**Notes:**\\n- This function has security vulnerabilities and should be refactored\\n\\n### good_function\\n\\nSafely processes a parameter.\\n\\n**Parameters:**\\n- `param` (str): Parameter to process\\n\\n**Returns:**\\n- str or None: Processed parameter or None if input is invalid\\n\\n## Classes\\n\\n### ExampleClass\\n\\nExample class demonstrating basic functionality.\\n\\n#### Methods\\n\\n##### __init__\\n\\nInitialize the class with a name.\\n\\n**Parameters:**\\n- `name` (str): Name for the instance\\n\\n##### public_method\\n\\nPublic method that returns a greeting.\\n\\n**Returns:**\\n- str: Greeting message"
            }
            ''',
            'success': True
        })())
        
        # Generate documentation
        result = await ai_analyzer.generate_documentation(sample_python_code, "python")
        
        # Verify the results
        assert isinstance(result, DocumentationResult)
        assert result.language == "python"
        assert len(result.functions) == 2
        assert len(result.classes) == 1
        assert result.markdown is not None
        
        # Verify function documentation
        vulnerable_func = result.functions[0]
        assert vulnerable_func["name"] == "vulnerable_function"
        assert vulnerable_func["line"] == 2
        assert "Processes user input" in vulnerable_func["description"]
        assert len(vulnerable_func["parameters"]) == 1
        assert vulnerable_func["parameters"][0]["name"] == "user_input"
        assert vulnerable_func["parameters"][0]["type"] == "str"
        assert vulnerable_func["returns"]["type"] == "int"
        assert len(vulnerable_func["notes"]) == 1
        
        good_func = result.functions[1]
        assert good_func["name"] == "good_function"
        assert good_func["line"] == 18
        assert "Safely processes" in good_func["description"]
        assert len(good_func["parameters"]) == 1
        assert good_func["returns"]["type"] == "str or None"
        
        # Verify class documentation
        example_class = result.classes[0]
        assert example_class["name"] == "ExampleClass"
        assert example_class["line"] == 25
        assert "Example class" in example_class["description"]
        assert len(example_class["methods"]) == 2
        
        # Verify markdown generation
        assert "# Code Documentation" in result.markdown
        assert "## Functions" in result.markdown
        assert "## Classes" in result.markdown
        assert "vulnerable_function" in result.markdown
        assert "ExampleClass" in result.markdown
    
    @pytest.mark.asyncio
    async def test_error_handling_ollama_failure(self, ai_analyzer, sample_python_code):
        """Test error handling when Ollama fails."""
        # Mock Ollama failure
        ai_analyzer.ollama_service.generate_text = AsyncMock(side_effect=Exception("Ollama service unavailable"))
        
        # Test that appropriate error is raised
        with pytest.raises(AIAnalysisError):
            await ai_analyzer.analyze_code(sample_python_code, "python")
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_response(self, ai_analyzer, sample_python_code):
        """Test error handling for invalid AI response."""
        # Mock invalid JSON response
        ai_analyzer.ollama_service.generate_text = AsyncMock(return_value=type('Response', (), {
            'content': 'Invalid JSON response',
            'success': True
        })())
        
        # Test that appropriate error is raised
        with pytest.raises(AIAnalysisError):
            await ai_analyzer.analyze_code(sample_python_code, "python")
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, ai_analyzer, sample_python_code):
        """Test timeout handling for long-running analysis."""
        # Mock timeout
        ai_analyzer.ollama_service.generate_text = AsyncMock(side_effect=asyncio.TimeoutError("Analysis timed out"))
        
        # Test that timeout is handled properly
        with pytest.raises(AIAnalysisError):
            await ai_analyzer.analyze_code(sample_python_code, "python")
    
    @pytest.mark.asyncio
    async def test_unsupported_language(self, ai_analyzer):
        """Test handling of unsupported programming languages."""
        unsupported_code = "This is not a supported language"
        
        result = await ai_analyzer.analyze_code(unsupported_code, "unsupported")
        
        # Should return a basic analysis or error
        assert result is not None
        assert result.language == "unsupported"
    
    @pytest.mark.asyncio
    async def test_confidence_filtering(self, ai_analyzer, sample_python_code):
        """Test filtering of low-confidence results."""
        # Mock response with mixed confidence levels
        ai_analyzer.ollama_service.generate_text = AsyncMock(return_value=type('Response', (), {
            'content': '''
            {
                "overall_score": 7.0,
                "security_issues": [
                    {
                        "type": "high_confidence_issue",
                        "severity": "high",
                        "line": 5,
                        "description": "High confidence security issue",
                        "confidence": 0.9
                    },
                    {
                        "type": "low_confidence_issue",
                        "severity": "medium",
                        "line": 10,
                        "description": "Low confidence security issue",
                        "confidence": 0.5
                    }
                ],
                "performance_issues": [],
                "style_issues": [],
                "best_practices": []
            }
            ''',
            'success': True
        })())
        
        # Analyze with confidence filtering
        result = await ai_analyzer.analyze_code(sample_python_code, "python", filter_confidence=True)
        
        # Should only include high-confidence issues
        assert len(result.security_issues) == 1
        assert result.security_issues[0].type == "high_confidence_issue"
        assert result.security_issues[0].confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, ai_analyzer):
        """Test concurrent code analysis requests."""
        code_samples = [
            "def test1(): pass",
            "def test2(): pass",
            "def test3(): pass"
        ]
        
        # Mock responses
        ai_analyzer.ollama_service.generate_text = AsyncMock(return_value=type('Response', (), {
            'content': '''
            {
                "overall_score": 8.0,
                "security_issues": [],
                "performance_issues": [],
                "style_issues": [],
                "best_practices": []
            }
            ''',
            'success': True
        })())
        
        # Analyze concurrently
        tasks = [
            ai_analyzer.analyze_code(code, "python")
            for code in code_samples
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all analyses completed
        assert len(results) == 3
        for result in results:
            assert isinstance(result, CodeAnalysisResult)
            assert result.overall_score == 8.0
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, ai_analyzer):
        """Test getting analyzer metrics."""
        # Set up some test metrics
        ai_analyzer.analysis_count = 50
        ai_analyzer.error_count = 2
        ai_analyzer.total_processing_time = 120.5
        
        metrics = ai_analyzer.get_metrics()
        
        # Verify metrics
        assert metrics["analysis_count"] == 50
        assert metrics["error_count"] == 2
        assert metrics["total_processing_time"] == 120.5
        assert metrics["average_processing_time"] == 2.41  # 120.5 / 50
        assert "success_rate" in metrics
        assert "uptime" in metrics
    
    @pytest.mark.asyncio
    async def test_model_switching(self, ai_analyzer):
        """Test switching between different AI models."""
        # Test switching to different model
        ai_analyzer.set_model("codellama:13b")
        assert ai_analyzer.model == "codellama:13b"
        
        # Test switching back
        ai_analyzer.set_model("codellama:7b")
        assert ai_analyzer.model == "codellama:7b"
    
    @pytest.mark.asyncio
    async def test_custom_prompts(self, ai_analyzer, sample_python_code):
        """Test using custom analysis prompts."""
        custom_prompt = "Focus only on security issues and ignore style problems"
        
        # Mock response
        ai_analyzer.ollama_service.generate_text = AsyncMock(return_value=type('Response', (), {
            'content': '''
            {
                "overall_score": 7.5,
                "security_issues": [
                    {
                        "type": "command_injection",
                        "severity": "high",
                        "line": 5,
                        "description": "Command injection vulnerability",
                        "confidence": 0.9
                    }
                ],
                "performance_issues": [],
                "style_issues": [],
                "best_practices": []
            }
            ''',
            'success': True
        })())
        
        result = await ai_analyzer.analyze_code(
            sample_python_code, 
            "python", 
            custom_prompt=custom_prompt
        )
        
        # Verify that custom prompt was used
        assert len(result.security_issues) == 1
        assert len(result.style_issues) == 0  # Should be ignored per custom prompt
        ai_analyzer.ollama_service.generate_text.assert_called_once()
        call_args = ai_analyzer.ollama_service.generate_text.call_args[0]
        assert custom_prompt in call_args[0]  # Custom prompt should be in the request