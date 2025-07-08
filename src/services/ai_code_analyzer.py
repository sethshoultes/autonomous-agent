"""
AI-powered code analysis service for the Code Agent.

This service provides comprehensive AI-driven code analysis capabilities including
security vulnerability detection, code quality assessment, performance analysis,
style checking, and automated documentation generation using local Ollama models.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from .ollama_service import OllamaService, ProcessingRequest, ProcessingResponse


# Exception classes
class AIAnalysisError(Exception):
    """Base exception for AI analysis errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(message)


class AIModelError(AIAnalysisError):
    """Exception raised when AI model encounters an error."""
    pass


class AIAnalysisTimeoutError(AIAnalysisError):
    """Exception raised when AI analysis times out."""
    pass


class AIAnalysisValidationError(AIAnalysisError):
    """Exception raised when AI analysis validation fails."""
    pass


# Data models
@dataclass
class SecurityIssue:
    """Security vulnerability information."""
    id: str
    type: str
    severity: str
    line: int
    description: str
    impact: str
    remediation: str
    confidence: float
    cvss_score: Optional[float] = None
    cwe: Optional[str] = None
    references: List[str] = field(default_factory=list)


@dataclass
class PerformanceIssue:
    """Performance issue information."""
    type: str
    severity: str
    line: int
    description: str
    impact: str
    recommendation: str
    confidence: float
    complexity: Optional[str] = None
    estimated_improvement: Optional[str] = None


@dataclass
class StyleIssue:
    """Code style issue information."""
    type: str
    severity: str
    line: int
    description: str
    recommendation: str
    confidence: float
    rule: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class ReviewComment:
    """Code review comment."""
    file: str
    line: int
    type: str
    message: str
    confidence: float
    category: str = "general"
    suggestion: Optional[str] = None


@dataclass
class CodeQualityScore:
    """Code quality assessment."""
    overall_score: float
    maintainability: float
    readability: float
    complexity: float
    test_coverage: Optional[float] = None
    documentation: Optional[float] = None


@dataclass
class CodeAnalysisResult:
    """Complete code analysis result."""
    analysis_id: str
    language: str
    overall_score: float
    processing_time: float
    security_issues: List[SecurityIssue]
    performance_issues: List[PerformanceIssue]
    style_issues: List[StyleIssue]
    quality_score: CodeQualityScore
    best_practices: List[str]
    recommendations: List[str]
    summary: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SecurityAnalysisResult:
    """Security analysis result."""
    scan_id: str
    language: str
    vulnerabilities: List[SecurityIssue]
    total_vulnerabilities: int
    critical_vulnerabilities: int
    high_vulnerabilities: int
    medium_vulnerabilities: int
    low_vulnerabilities: int
    risk_score: float
    scan_duration: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PullRequestReviewResult:
    """Pull request review result."""
    review_id: str
    overall_assessment: str
    score: float
    summary: str
    comments: List[ReviewComment]
    security_improvements: List[str]
    potential_issues: List[str]
    testing_suggestions: List[str]
    files_analyzed: int
    review_duration: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DocumentationResult:
    """Documentation generation result."""
    doc_id: str
    language: str
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    modules: List[Dict[str, Any]]
    markdown: str
    coverage: Dict[str, float]
    generation_time: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AICodeAnalyzer:
    """
    AI-powered code analysis service.
    
    Provides comprehensive code analysis using local AI models through Ollama,
    including security scanning, quality assessment, performance analysis,
    and automated documentation generation.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the AI Code Analyzer.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Configuration
        self.model = config.get("model", "codellama:7b")
        self.temperature = config.get("temperature", 0.2)
        self.max_context_length = config.get("max_context_length", 8192)
        self.analysis_timeout = config.get("analysis_timeout", 60)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        
        # Feature flags
        self.security_enabled = config.get("security_enabled", True)
        self.performance_enabled = config.get("performance_enabled", True)
        self.style_enabled = config.get("style_enabled", True)
        self.documentation_enabled = config.get("documentation_enabled", True)
        
        # Initialize Ollama service
        self.ollama_service = OllamaService(config, logger)
        
        # Language-specific configurations
        self.language_configs = {
            "python": {
                "security_patterns": ["sql_injection", "command_injection", "xss", "path_traversal"],
                "style_rules": ["pep8", "naming", "docstrings", "imports"],
                "performance_checks": ["loops", "comprehensions", "data_structures"]
            },
            "javascript": {
                "security_patterns": ["xss", "prototype_pollution", "eval", "cors"],
                "style_rules": ["eslint", "naming", "jsdoc", "semicolons"],
                "performance_checks": ["dom_manipulation", "event_handlers", "memory_leaks"]
            },
            "java": {
                "security_patterns": ["sql_injection", "deserialization", "xxe", "path_traversal"],
                "style_rules": ["checkstyle", "naming", "javadoc", "imports"],
                "performance_checks": ["collections", "streams", "concurrency"]
            },
            "go": {
                "security_patterns": ["sql_injection", "command_injection", "path_traversal"],
                "style_rules": ["gofmt", "naming", "comments", "errors"],
                "performance_checks": ["goroutines", "channels", "memory"]
            }
        }
        
        # Metrics
        self.analysis_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self.start_time = time.time()
        
        # Prompt templates
        self._load_prompt_templates()
    
    def _load_prompt_templates(self) -> None:
        """Load prompt templates for different analysis types."""
        self.prompts = {
            "code_analysis": """
Analyze the following {language} code for security vulnerabilities, performance issues, 
style problems, and overall code quality. Provide a comprehensive analysis in JSON format.

Code to analyze:
```{language}
{code}
```

Please provide analysis in the following JSON structure:
{{
    "overall_score": <float 0-10>,
    "security_issues": [
        {{
            "type": "<vulnerability_type>",
            "severity": "<critical|high|medium|low>",
            "line": <line_number>,
            "description": "<description>",
            "recommendation": "<fix_suggestion>",
            "confidence": <float 0-1>
        }}
    ],
    "performance_issues": [
        {{
            "type": "<issue_type>",
            "severity": "<high|medium|low>",
            "line": <line_number>,
            "description": "<description>",
            "recommendation": "<optimization_suggestion>",
            "confidence": <float 0-1>
        }}
    ],
    "style_issues": [
        {{
            "type": "<style_issue_type>",
            "severity": "<high|medium|low>",
            "line": <line_number>,
            "description": "<description>",
            "recommendation": "<style_fix>",
            "confidence": <float 0-1>
        }}
    ],
    "best_practices": [
        "<best_practice_recommendation>"
    ]
}}

Focus on:
1. Security vulnerabilities (SQL injection, XSS, command injection, etc.)
2. Performance bottlenecks and inefficiencies
3. Code style and formatting issues
4. Best practices and recommendations
""",

            "security_scan": """
Perform a comprehensive security analysis of the following {language} code. 
Focus on identifying vulnerabilities, security weaknesses, and potential attack vectors.

Code to analyze:
```{language}
{code}
```

Provide detailed security analysis in JSON format:
{{
    "scan_id": "<unique_scan_id>",
    "vulnerabilities": [
        {{
            "id": "<CWE_ID>",
            "type": "<vulnerability_type>",
            "severity": "<critical|high|medium|low>",
            "line": <line_number>,
            "description": "<detailed_description>",
            "impact": "<potential_impact>",
            "remediation": "<fix_instructions>",
            "confidence": <float 0-1>,
            "cvss_score": <float 0-10>
        }}
    ],
    "summary": {{
        "total_vulnerabilities": <count>,
        "critical": <count>,
        "high": <count>,
        "medium": <count>,
        "low": <count>,
        "risk_score": <float 0-10>
    }}
}}

Look for:
1. Injection vulnerabilities (SQL, Command, LDAP, etc.)
2. Authentication and authorization flaws
3. Sensitive data exposure
4. XML external entity (XXE) vulnerabilities
5. Broken access control
6. Security misconfiguration
7. Cross-site scripting (XSS)
8. Insecure deserialization
9. Using components with known vulnerabilities
10. Insufficient logging and monitoring
""",

            "pr_review": """
Review the following pull request changes and provide comprehensive feedback.
Focus on code quality, security, performance, and maintainability.

Pull Request Title: {title}
Description: {description}

Files changed:
{files_summary}

File diffs:
{diffs}

Provide review feedback in JSON format:
{{
    "review_id": "<unique_review_id>",
    "overall_assessment": "<approve|request_changes|comment>",
    "summary": "<brief_summary>",
    "score": <float 0-10>,
    "comments": [
        {{
            "file": "<filename>",
            "line": <line_number>,
            "type": "<improvement|issue|suggestion|security>",
            "message": "<detailed_comment>",
            "confidence": <float 0-1>
        }}
    ],
    "security_improvements": [
        "<security_improvement_description>"
    ],
    "potential_issues": [
        "<potential_issue_description>"
    ],
    "testing_suggestions": [
        "<testing_recommendation>"
    ]
}}

Review criteria:
1. Code quality and maintainability
2. Security considerations
3. Performance implications
4. Test coverage and quality
5. Documentation completeness
6. Adherence to coding standards
7. Breaking changes impact
""",

            "documentation": """
Generate comprehensive documentation for the following {language} code.
Create detailed API documentation with examples and usage information.

Code to document:
```{language}
{code}
```

Generate documentation in JSON format:
{{
    "documentation": {{
        "functions": [
            {{
                "name": "<function_name>",
                "line": <line_number>,
                "description": "<detailed_description>",
                "parameters": [
                    {{
                        "name": "<param_name>",
                        "type": "<param_type>",
                        "description": "<param_description>"
                    }}
                ],
                "returns": {{
                    "type": "<return_type>",
                    "description": "<return_description>"
                }},
                "raises": [
                    {{
                        "exception": "<exception_type>",
                        "description": "<when_raised>"
                    }}
                ],
                "examples": [
                    "<usage_example>"
                ]
            }}
        ],
        "classes": [
            {{
                "name": "<class_name>",
                "line": <line_number>,
                "description": "<class_description>",
                "methods": [
                    {{
                        "name": "<method_name>",
                        "description": "<method_description>",
                        "parameters": [...],
                        "returns": {{...}}
                    }}
                ]
            }}
        ]
    }},
    "markdown": "<generated_markdown_documentation>"
}}

Include:
1. Function and method documentation
2. Class and interface documentation
3. Parameter and return value descriptions
4. Usage examples
5. Exception handling information
6. Code organization and structure
""",

            "improvement_suggestions": """
Analyze the following {language} code and provide specific improvement suggestions.
Focus on code quality, performance, security, and maintainability enhancements.

Code to improve:
```{language}
{code}
```

Provide suggestions in JSON format:
{{
    "suggestions": [
        {{
            "category": "<security|performance|style|maintainability>",
            "priority": "<high|medium|low>",
            "description": "<improvement_description>",
            "example": "<code_example_or_fix>",
            "impact": "<expected_impact>"
        }}
    ],
    "best_practices": [
        "<best_practice_recommendation>"
    ]
}}

Categories to consider:
1. Security improvements
2. Performance optimizations
3. Code style and formatting
4. Error handling improvements
5. Code organization and structure
6. Documentation enhancements
7. Testing recommendations
8. Dependency management
"""
        }
    
    async def analyze_code(
        self, 
        code: str, 
        language: str,
        analysis_type: str = "comprehensive",
        custom_prompt: Optional[str] = None,
        filter_confidence: bool = True
    ) -> CodeAnalysisResult:
        """
        Perform comprehensive code analysis.
        
        Args:
            code: Code to analyze
            language: Programming language
            analysis_type: Type of analysis ("comprehensive", "security", "performance", "style")
            custom_prompt: Custom analysis prompt
            filter_confidence: Filter results by confidence threshold
            
        Returns:
            CodeAnalysisResult object
        """
        start_time = time.time()
        analysis_id = str(uuid.uuid4())
        
        try:
            # Prepare analysis prompt
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = self.prompts["code_analysis"].format(
                    language=language,
                    code=code
                )
            
            # Create processing request
            request = ProcessingRequest(
                prompt=prompt,
                model=self.model,
                task_type="code_analysis",
                options={
                    "temperature": self.temperature,
                    "max_tokens": self.max_context_length
                }
            )
            
            # Perform AI analysis
            response = await asyncio.wait_for(
                self.ollama_service.generate_text(
                    prompt=request.prompt,
                    model=request.model,
                    options=request.options
                ),
                timeout=self.analysis_timeout
            )
            
            if not response.success:
                raise AIModelError(f"AI analysis failed: {response.error}")
            
            # Parse response
            analysis_data = self._parse_analysis_response(response.content)
            
            # Create result objects
            security_issues = [
                SecurityIssue(
                    id=f"sec_{i}",
                    type=issue["type"],
                    severity=issue["severity"],
                    line=issue["line"],
                    description=issue["description"],
                    impact=issue.get("impact", "Unknown"),
                    remediation=issue["recommendation"],
                    confidence=issue["confidence"]
                )
                for i, issue in enumerate(analysis_data.get("security_issues", []))
                if not filter_confidence or issue["confidence"] >= self.confidence_threshold
            ]
            
            performance_issues = [
                PerformanceIssue(
                    type=issue["type"],
                    severity=issue["severity"],
                    line=issue["line"],
                    description=issue["description"],
                    impact=issue.get("impact", "Unknown"),
                    recommendation=issue["recommendation"],
                    confidence=issue["confidence"]
                )
                for issue in analysis_data.get("performance_issues", [])
                if not filter_confidence or issue["confidence"] >= self.confidence_threshold
            ]
            
            style_issues = [
                StyleIssue(
                    type=issue["type"],
                    severity=issue["severity"],
                    line=issue["line"],
                    description=issue["description"],
                    recommendation=issue["recommendation"],
                    confidence=issue["confidence"]
                )
                for issue in analysis_data.get("style_issues", [])
                if not filter_confidence or issue["confidence"] >= self.confidence_threshold
            ]
            
            quality_score = CodeQualityScore(
                overall_score=analysis_data.get("overall_score", 7.0),
                maintainability=analysis_data.get("maintainability", 7.0),
                readability=analysis_data.get("readability", 7.0),
                complexity=analysis_data.get("complexity", 5.0)
            )
            
            processing_time = time.time() - start_time
            
            # Update metrics
            self.analysis_count += 1
            self.total_processing_time += processing_time
            
            return CodeAnalysisResult(
                analysis_id=analysis_id,
                language=language,
                overall_score=analysis_data.get("overall_score", 7.0),
                processing_time=processing_time,
                security_issues=security_issues,
                performance_issues=performance_issues,
                style_issues=style_issues,
                quality_score=quality_score,
                best_practices=analysis_data.get("best_practices", []),
                recommendations=analysis_data.get("recommendations", []),
                summary=analysis_data.get("summary", "Code analysis completed")
            )
            
        except asyncio.TimeoutError:
            self.error_count += 1
            raise AIAnalysisTimeoutError(f"Analysis timed out after {self.analysis_timeout} seconds")
        except json.JSONDecodeError as e:
            self.error_count += 1
            raise AIAnalysisValidationError(f"Invalid AI response format: {e}")
        except Exception as e:
            self.error_count += 1
            raise AIAnalysisError(f"Code analysis failed: {e}") from e
    
    async def detect_vulnerabilities(
        self, 
        code: str, 
        language: str,
        scan_type: str = "full",
        include_dependencies: bool = True
    ) -> SecurityAnalysisResult:
        """
        Perform security vulnerability detection.
        
        Args:
            code: Code to scan
            language: Programming language
            scan_type: Type of scan ("full", "quick", "targeted")
            include_dependencies: Include dependency vulnerability scanning
            
        Returns:
            SecurityAnalysisResult object
        """
        start_time = time.time()
        scan_id = str(uuid.uuid4())
        
        try:
            # Prepare security scan prompt
            prompt = self.prompts["security_scan"].format(
                language=language,
                code=code
            )
            
            # Create processing request
            request = ProcessingRequest(
                prompt=prompt,
                model=self.model,
                task_type="security_analysis",
                options={
                    "temperature": 0.1,  # Lower temperature for security analysis
                    "max_tokens": self.max_context_length
                }
            )
            
            # Perform AI analysis
            response = await asyncio.wait_for(
                self.ollama_service.generate_text(
                    prompt=request.prompt,
                    model=request.model,
                    options=request.options
                ),
                timeout=self.analysis_timeout
            )
            
            if not response.success:
                raise AIModelError(f"Security analysis failed: {response.error}")
            
            # Parse response
            scan_data = self._parse_analysis_response(response.content)
            
            # Create vulnerability objects
            vulnerabilities = [
                SecurityIssue(
                    id=vuln.get("id", f"vuln_{i}"),
                    type=vuln["type"],
                    severity=vuln["severity"],
                    line=vuln["line"],
                    description=vuln["description"],
                    impact=vuln.get("impact", "Unknown"),
                    remediation=vuln.get("remediation", "No specific remediation provided"),
                    confidence=vuln["confidence"],
                    cvss_score=vuln.get("cvss_score"),
                    cwe=vuln.get("cwe")
                )
                for i, vuln in enumerate(scan_data.get("vulnerabilities", []))
            ]
            
            # Calculate severity counts
            severity_counts = {
                "critical": len([v for v in vulnerabilities if v.severity == "critical"]),
                "high": len([v for v in vulnerabilities if v.severity == "high"]),
                "medium": len([v for v in vulnerabilities if v.severity == "medium"]),
                "low": len([v for v in vulnerabilities if v.severity == "low"])
            }
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(vulnerabilities)
            
            scan_duration = time.time() - start_time
            
            return SecurityAnalysisResult(
                scan_id=scan_id,
                language=language,
                vulnerabilities=vulnerabilities,
                total_vulnerabilities=len(vulnerabilities),
                critical_vulnerabilities=severity_counts["critical"],
                high_vulnerabilities=severity_counts["high"],
                medium_vulnerabilities=severity_counts["medium"],
                low_vulnerabilities=severity_counts["low"],
                risk_score=risk_score,
                scan_duration=scan_duration,
                recommendations=scan_data.get("recommendations", [])
            )
            
        except asyncio.TimeoutError:
            self.error_count += 1
            raise AIAnalysisTimeoutError(f"Security scan timed out after {self.analysis_timeout} seconds")
        except Exception as e:
            self.error_count += 1
            raise AIAnalysisError(f"Security analysis failed: {e}") from e
    
    async def review_pull_request(self, pr_data: Dict[str, Any]) -> PullRequestReviewResult:
        """
        Perform AI-powered pull request review.
        
        Args:
            pr_data: Pull request data including files and diffs
            
        Returns:
            PullRequestReviewResult object
        """
        start_time = time.time()
        review_id = str(uuid.uuid4())
        
        try:
            # Prepare file summaries and diffs
            files_summary = []
            diffs = []
            
            for file_data in pr_data.get("files", []):
                files_summary.append(f"- {file_data['filename']} (+{file_data.get('additions', 0)} -{file_data.get('deletions', 0)})")
                if file_data.get("patch"):
                    diffs.append(f"File: {file_data['filename']}\n{file_data['patch']}")
            
            # Prepare review prompt
            prompt = self.prompts["pr_review"].format(
                title=pr_data.get("title", "Pull Request"),
                description=pr_data.get("body", "No description provided"),
                files_summary="\n".join(files_summary),
                diffs="\n\n".join(diffs[:10])  # Limit to first 10 files
            )
            
            # Create processing request
            request = ProcessingRequest(
                prompt=prompt,
                model=self.model,
                task_type="code_review",
                options={
                    "temperature": self.temperature,
                    "max_tokens": self.max_context_length
                }
            )
            
            # Perform AI analysis
            response = await asyncio.wait_for(
                self.ollama_service.generate_text(
                    prompt=request.prompt,
                    model=request.model,
                    options=request.options
                ),
                timeout=self.analysis_timeout * 2  # Longer timeout for PR reviews
            )
            
            if not response.success:
                raise AIModelError(f"PR review failed: {response.error}")
            
            # Parse response
            review_data = self._parse_analysis_response(response.content)
            
            # Create review comments
            comments = [
                ReviewComment(
                    file=comment["file"],
                    line=comment["line"],
                    type=comment["type"],
                    message=comment["message"],
                    confidence=comment["confidence"],
                    category=comment.get("category", "general")
                )
                for comment in review_data.get("comments", [])
            ]
            
            review_duration = time.time() - start_time
            
            return PullRequestReviewResult(
                review_id=review_id,
                overall_assessment=review_data.get("overall_assessment", "comment"),
                score=review_data.get("score", 7.0),
                summary=review_data.get("summary", "Pull request reviewed"),
                comments=comments,
                security_improvements=review_data.get("security_improvements", []),
                potential_issues=review_data.get("potential_issues", []),
                testing_suggestions=review_data.get("testing_suggestions", []),
                files_analyzed=len(pr_data.get("files", [])),
                review_duration=review_duration
            )
            
        except asyncio.TimeoutError:
            self.error_count += 1
            raise AIAnalysisTimeoutError(f"PR review timed out after {self.analysis_timeout * 2} seconds")
        except Exception as e:
            self.error_count += 1
            raise AIAnalysisError(f"PR review failed: {e}") from e
    
    async def generate_documentation(
        self, 
        code: Union[str, Dict[str, str]], 
        language: str,
        doc_type: str = "api",
        format_type: str = "markdown"
    ) -> DocumentationResult:
        """
        Generate comprehensive code documentation.
        
        Args:
            code: Code to document (string or dict of file_path: content)
            language: Programming language
            doc_type: Type of documentation ("api", "user", "developer")
            format_type: Output format ("markdown", "html", "rst")
            
        Returns:
            DocumentationResult object
        """
        start_time = time.time()
        doc_id = str(uuid.uuid4())
        
        try:
            # Handle multiple files
            if isinstance(code, dict):
                # Combine multiple files for analysis
                combined_code = "\n\n".join([
                    f"# File: {path}\n{content}"
                    for path, content in code.items()
                ])
            else:
                combined_code = code
            
            # Prepare documentation prompt
            prompt = self.prompts["documentation"].format(
                language=language,
                code=combined_code
            )
            
            # Create processing request
            request = ProcessingRequest(
                prompt=prompt,
                model=self.model,
                task_type="documentation",
                options={
                    "temperature": 0.3,  # Slightly higher for creative documentation
                    "max_tokens": self.max_context_length
                }
            )
            
            # Perform AI analysis
            response = await asyncio.wait_for(
                self.ollama_service.generate_text(
                    prompt=request.prompt,
                    model=request.model,
                    options=request.options
                ),
                timeout=self.analysis_timeout * 2  # Longer timeout for documentation
            )
            
            if not response.success:
                raise AIModelError(f"Documentation generation failed: {response.error}")
            
            # Parse response
            doc_data = self._parse_analysis_response(response.content)
            
            documentation = doc_data.get("documentation", {})
            
            generation_time = time.time() - start_time
            
            # Calculate coverage
            coverage = {
                "functions": len(documentation.get("functions", [])),
                "classes": len(documentation.get("classes", [])),
                "modules": len(documentation.get("modules", []))
            }
            
            return DocumentationResult(
                doc_id=doc_id,
                language=language,
                functions=documentation.get("functions", []),
                classes=documentation.get("classes", []),
                modules=documentation.get("modules", []),
                markdown=doc_data.get("markdown", ""),
                coverage=coverage,
                generation_time=generation_time
            )
            
        except asyncio.TimeoutError:
            self.error_count += 1
            raise AIAnalysisTimeoutError(f"Documentation generation timed out after {self.analysis_timeout * 2} seconds")
        except Exception as e:
            self.error_count += 1
            raise AIAnalysisError(f"Documentation generation failed: {e}") from e
    
    async def suggest_improvements(
        self, 
        code: str, 
        language: str,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate code improvement suggestions.
        
        Args:
            code: Code to analyze
            language: Programming language
            focus_areas: Specific areas to focus on
            
        Returns:
            Dictionary containing improvement suggestions
        """
        try:
            # Prepare improvement prompt
            prompt = self.prompts["improvement_suggestions"].format(
                language=language,
                code=code
            )
            
            # Create processing request
            request = ProcessingRequest(
                prompt=prompt,
                model=self.model,
                task_type="improvement_analysis",
                options={
                    "temperature": self.temperature,
                    "max_tokens": self.max_context_length
                }
            )
            
            # Perform AI analysis
            response = await asyncio.wait_for(
                self.ollama_service.generate_text(
                    prompt=request.prompt,
                    model=request.model,
                    options=request.options
                ),
                timeout=self.analysis_timeout
            )
            
            if not response.success:
                raise AIModelError(f"Improvement analysis failed: {response.error}")
            
            # Parse and return response
            return self._parse_analysis_response(response.content)
            
        except asyncio.TimeoutError:
            self.error_count += 1
            raise AIAnalysisTimeoutError(f"Improvement analysis timed out after {self.analysis_timeout} seconds")
        except Exception as e:
            self.error_count += 1
            raise AIAnalysisError(f"Improvement analysis failed: {e}") from e
    
    async def analyze_issue(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a GitHub issue using AI.
        
        Args:
            issue_data: Issue data from GitHub
            
        Returns:
            Analysis result with categorization and suggestions
        """
        try:
            # Prepare issue analysis prompt
            prompt = f"""
Analyze the following GitHub issue and provide categorization and recommendations.

Issue Title: {issue_data.get('title', 'No title')}
Issue Body: {issue_data.get('body', 'No description')}
Labels: {', '.join(issue_data.get('labels', []))}

Provide analysis in JSON format:
{{
    "category": "<bug|feature|enhancement|documentation|question>",
    "priority": "<critical|high|medium|low>",
    "complexity": "<low|medium|high>",
    "tags": ["<relevant_tag>"],
    "suggested_assignees": ["<team_or_person>"],
    "estimated_effort": "<hours_or_story_points>",
    "related_components": ["<component_name>"],
    "recommendations": ["<recommendation>"]
}}
"""
            
            # Create processing request
            request = ProcessingRequest(
                prompt=prompt,
                model=self.model,
                task_type="issue_analysis",
                options={
                    "temperature": 0.3,
                    "max_tokens": 2048
                }
            )
            
            # Perform AI analysis
            response = await self.ollama_service.generate_text(
                prompt=request.prompt,
                model=request.model,
                options=request.options
            )
            
            if not response.success:
                raise AIModelError(f"Issue analysis failed: {response.error}")
            
            return self._parse_analysis_response(response.content)
            
        except Exception as e:
            self.error_count += 1
            raise AIAnalysisError(f"Issue analysis failed: {e}") from e
    
    async def analyze_commits(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a series of commits for patterns and issues.
        
        Args:
            commits: List of commit data
            
        Returns:
            Commit analysis result
        """
        try:
            # Prepare commit analysis
            commit_messages = [commit.get("message", "") for commit in commits]
            combined_messages = "\n".join(commit_messages)
            
            prompt = f"""
Analyze the following commit messages for patterns, quality, and potential issues:

Commits:
{combined_messages}

Provide analysis in JSON format:
{{
    "analysis_id": "<unique_id>",
    "commits_analyzed": {len(commits)},
    "message_quality": {{
        "average_score": <float 0-10>,
        "issues": ["<issue_description>"]
    }},
    "patterns": {{
        "common_types": ["<commit_type>"],
        "breaking_changes": <count>,
        "hot_files": ["<frequently_modified_file>"]
    }},
    "recommendations": ["<recommendation>"]
}}
"""
            
            # Create processing request
            request = ProcessingRequest(
                prompt=prompt,
                model=self.model,
                task_type="commit_analysis",
                options={
                    "temperature": 0.3,
                    "max_tokens": 2048
                }
            )
            
            # Perform AI analysis
            response = await self.ollama_service.generate_text(
                prompt=request.prompt,
                model=request.model,
                options=request.options
            )
            
            if not response.success:
                raise AIModelError(f"Commit analysis failed: {response.error}")
            
            return self._parse_analysis_response(response.content)
            
        except Exception as e:
            self.error_count += 1
            raise AIAnalysisError(f"Commit analysis failed: {e}") from e
    
    def set_model(self, model: str) -> None:
        """
        Set the AI model to use for analysis.
        
        Args:
            model: Model name
        """
        self.model = model
        self.logger.info(f"AI model changed to: {model}")
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse AI analysis response.
        
        Args:
            response: Raw AI response
            
        Returns:
            Parsed response data
            
        Raises:
            AIAnalysisValidationError: If response format is invalid
        """
        try:
            # Extract JSON from response (handle cases where AI adds extra text)
            import re
            
            # Look for JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
            
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse AI response: {e}")
            self.logger.debug(f"Raw response: {response}")
            raise AIAnalysisValidationError(f"Invalid JSON response from AI: {e}")
    
    def _calculate_risk_score(self, vulnerabilities: List[SecurityIssue]) -> float:
        """
        Calculate overall risk score based on vulnerabilities.
        
        Args:
            vulnerabilities: List of security issues
            
        Returns:
            Risk score (0-10)
        """
        if not vulnerabilities:
            return 0.0
        
        severity_weights = {
            "critical": 10.0,
            "high": 7.5,
            "medium": 5.0,
            "low": 2.5
        }
        
        total_weight = 0.0
        max_possible_weight = 0.0
        
        for vuln in vulnerabilities:
            weight = severity_weights.get(vuln.severity, 2.5)
            confidence_adjusted_weight = weight * vuln.confidence
            total_weight += confidence_adjusted_weight
            max_possible_weight += weight
        
        if max_possible_weight == 0:
            return 0.0
        
        # Normalize to 0-10 scale
        risk_score = (total_weight / max_possible_weight) * 10.0
        
        return min(10.0, risk_score)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get analyzer metrics.
        
        Returns:
            Dictionary containing metrics
        """
        uptime = time.time() - self.start_time
        
        return {
            "analysis_count": self.analysis_count,
            "error_count": self.error_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time / max(1, self.analysis_count),
            "success_rate": (self.analysis_count - self.error_count) / max(1, self.analysis_count),
            "uptime": uptime,
            "model": self.model,
            "features_enabled": {
                "security": self.security_enabled,
                "performance": self.performance_enabled,
                "style": self.style_enabled,
                "documentation": self.documentation_enabled
            }
        }