"""
Code Agent implementation for autonomous development assistance.

This agent provides comprehensive GitHub integration and AI-powered code review
capabilities, including repository monitoring, automated pull request analysis,
security vulnerability detection, and development workflow automation.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Callable
import uuid

from .base import BaseAgent, AgentMessage, AgentState
from .exceptions import AgentError, AgentStateError
from ..services.github_service import GitHubService, GitHubError
from ..services.ai_code_analyzer import AICodeAnalyzer, AIAnalysisError
from ..services.ollama_service import OllamaService
from ..models.code_agent_models import (
    CodeReviewRequest,
    CodeReviewResult,
    VulnerabilityReport,
    RepositoryMonitoringConfig,
    WebhookEvent,
    WorkflowAutomationConfig,
)


class CodeAgent(BaseAgent):
    """
    Code Agent for autonomous development assistance.
    
    Provides GitHub integration, AI-powered code review, repository monitoring,
    and development workflow automation capabilities.
    """
    
    def __init__(
        self,
        agent_id: str,
        config: Dict[str, Any],
        logger: logging.Logger,
        message_broker: Any
    ):
        """
        Initialize the Code Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration dictionary
            logger: Logger instance
            message_broker: Message broker for communication
        """
        super().__init__(agent_id, config, logger, message_broker)
        
        # Initialize services
        self.github_service = GitHubService(config.get("github", {}), logger)
        self.ai_analyzer = AICodeAnalyzer(config.get("ai_analysis", {}), logger)
        self.ollama_service = OllamaService(config, logger)
        
        # Configuration
        self.github_config = config.get("github", {})
        self.ai_config = config.get("ai_analysis", {})
        self.monitoring_config = config.get("repository_monitoring", {})
        self.review_config = config.get("code_review", {})
        self.workflow_config = config.get("workflow_automation", {})
        
        # State management
        self.monitored_repositories: List[str] = []
        self.active_reviews: Dict[str, CodeReviewResult] = {}
        self.webhook_handlers: Dict[str, Callable] = {}
        self.repository_states: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self._repository_monitoring_task: Optional[asyncio.Task] = None
        self._webhook_processing_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics.update({
            "pull_requests_reviewed": 0,
            "vulnerabilities_detected": 0,
            "documentation_generated": 0,
            "repositories_monitored": 0,
            "webhooks_processed": 0,
            "workflow_automations_triggered": 0,
        })
        
        # Event queues
        self.webhook_queue: asyncio.Queue = asyncio.Queue()
        self.review_queue: asyncio.Queue = asyncio.Queue()
        
        # Initialize webhook handlers
        self._setup_webhook_handlers()
    
    async def _initialize(self) -> None:
        """Initialize agent-specific resources."""
        try:
            # Validate configuration
            if not self._validate_configuration(self.config):
                raise AgentError("Invalid configuration provided")
            
            # Connect to services
            await self.github_service.connect()
            await self.ollama_service.connect()
            
            # Load monitored repositories
            await self._load_monitored_repositories()
            
            # Setup webhook handlers
            await self._setup_webhook_handlers()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info(f"Code Agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Code Agent: {e}")
            raise AgentError(f"Code Agent initialization failed: {e}") from e
    
    async def _cleanup(self) -> None:
        """Cleanup agent-specific resources."""
        try:
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Disconnect from services
            await self.github_service.disconnect()
            await self.ollama_service.disconnect()
            
            # Clear state
            self.monitored_repositories.clear()
            self.active_reviews.clear()
            self.repository_states.clear()
            
            self.logger.info(f"Code Agent {self.agent_id} cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup Code Agent: {e}")
            raise AgentError(f"Code Agent cleanup failed: {e}") from e
    
    async def _health_check(self) -> bool:
        """Perform agent-specific health check."""
        try:
            # Check service health
            github_health = await self.github_service.health_check()
            ollama_health = await self.ollama_service.health_check()
            
            github_ok = github_health.get("status") == "healthy"
            ollama_ok = ollama_health.get("status") == "healthy"
            
            # Check background tasks
            monitoring_ok = (
                self._repository_monitoring_task is None or
                not self._repository_monitoring_task.done()
            )
            
            webhook_ok = (
                self._webhook_processing_task is None or
                not self._webhook_processing_task.done()
            )
            
            return github_ok and ollama_ok and monitoring_ok and webhook_ok
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message.
        
        Args:
            message: Message to process
            
        Returns:
            Optional response message
        """
        try:
            message_type = message.message_type
            payload = message.payload
            
            if message_type == "github_webhook":
                return await self._handle_webhook_message(message)
            elif message_type == "code_review_request":
                return await self._handle_code_review_request(message)
            elif message_type == "vulnerability_scan_request":
                return await self._handle_vulnerability_scan_request(message)
            elif message_type == "documentation_request":
                return await self._handle_documentation_request(message)
            elif message_type == "repository_monitoring_request":
                return await self._handle_monitoring_request(message)
            else:
                self.logger.warning(f"Unknown message type: {message_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing message {message.id}: {e}")
            return AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                recipient=message.sender,
                message_type="error",
                payload={"error": str(e), "original_message_id": message.id}
            )
    
    async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        try:
            task_type = task.get("type")
            
            if task_type == "code_review":
                return await self._execute_code_review_task(task)
            elif task_type == "vulnerability_scan":
                return await self._execute_vulnerability_scan_task(task)
            elif task_type == "generate_documentation":
                return await self._execute_documentation_task(task)
            elif task_type == "monitor_repository":
                return await self._execute_monitoring_task(task)
            elif task_type == "workflow_automation":
                return await self._execute_workflow_automation_task(task)
            else:
                raise AgentError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"Error executing task: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_type": task.get("type", "unknown")
            }
    
    async def _execute_code_review_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a code review task."""
        try:
            repository = task["repository"]
            pull_request = task["pull_request"]
            review_type = task.get("review_type", "full")
            focus_areas = task.get("focus_areas", ["security", "performance", "style"])
            
            # Get pull request data
            pr_data = await self.github_service.get_pull_request(repository, pull_request)
            pr_files = await self.github_service.get_pull_request_files(repository, pull_request)
            
            # Prepare review request
            review_request = CodeReviewRequest(
                repository=repository,
                pull_request=pull_request,
                pr_data=pr_data,
                files=pr_files,
                review_type=review_type,
                focus_areas=focus_areas
            )
            
            # Perform AI analysis
            review_result = await self.ai_analyzer.review_pull_request(review_request.to_dict())
            
            # Create GitHub review
            if self.review_config.get("auto_review", True):
                await self._create_github_review(repository, pull_request, review_result)
            
            # Update metrics
            self.metrics["pull_requests_reviewed"] += 1
            
            # Store active review
            review_id = f"{repository}#{pull_request}"
            self.active_reviews[review_id] = review_result
            
            return {
                "success": True,
                "review_id": review_result.get("review_id"),
                "repository": repository,
                "pull_request": pull_request,
                "files_analyzed": len(pr_files),
                "issues_found": len(review_result.get("comments", [])),
                "overall_assessment": review_result.get("overall_assessment"),
                "score": review_result.get("score"),
                "comments": review_result.get("comments", []),
                "security_issues": review_result.get("security_issues", []),
                "performance_issues": review_result.get("performance_issues", []),
                "style_issues": review_result.get("style_issues", []),
                "recommendations": review_result.get("recommendations", [])
            }
            
        except Exception as e:
            self.logger.error(f"Code review task failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "repository": task.get("repository"),
                "pull_request": task.get("pull_request")
            }
    
    async def _execute_vulnerability_scan_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a vulnerability scan task."""
        try:
            repository = task["repository"]
            branch = task.get("branch", "main")
            scan_type = task.get("scan_type", "full")
            include_dependencies = task.get("include_dependencies", True)
            
            # Get repository files
            files = await self.github_service.get_repository_files(repository, branch)
            
            # Perform vulnerability scan
            scan_result = await self.ai_analyzer.detect_vulnerabilities(
                files, 
                scan_type=scan_type,
                include_dependencies=include_dependencies
            )
            
            # Create vulnerability report
            if scan_result.get("vulnerabilities"):
                await self._create_vulnerability_report(repository, scan_result)
            
            # Update metrics
            vuln_count = scan_result.get("total_vulnerabilities", 0)
            self.metrics["vulnerabilities_detected"] += vuln_count
            
            return {
                "success": True,
                "scan_id": scan_result.get("scan_id"),
                "repository": repository,
                "branch": branch,
                "scan_type": scan_type,
                "total_vulnerabilities": scan_result.get("total_vulnerabilities", 0),
                "critical_vulnerabilities": scan_result.get("critical_vulnerabilities", 0),
                "high_vulnerabilities": scan_result.get("high_vulnerabilities", 0),
                "medium_vulnerabilities": scan_result.get("medium_vulnerabilities", 0),
                "low_vulnerabilities": scan_result.get("low_vulnerabilities", 0),
                "vulnerabilities": scan_result.get("vulnerabilities", []),
                "dependencies": scan_result.get("dependencies", {}),
                "risk_score": scan_result.get("risk_score", 0),
                "recommendations": scan_result.get("recommendations", [])
            }
            
        except Exception as e:
            self.logger.error(f"Vulnerability scan task failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "repository": task.get("repository"),
                "branch": task.get("branch")
            }
    
    async def _execute_documentation_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a documentation generation task."""
        try:
            repository = task["repository"]
            files = task.get("files", [])
            doc_type = task.get("doc_type", "api")
            format_type = task.get("format", "markdown")
            
            # Get file contents
            file_contents = {}
            for file_path in files:
                content = await self.github_service.get_file_content(repository, file_path)
                file_contents[file_path] = content
            
            # Generate documentation
            doc_result = await self.ai_analyzer.generate_documentation(
                file_contents,
                doc_type=doc_type,
                format_type=format_type
            )
            
            # Create documentation files if requested
            if task.get("create_files", False):
                await self._create_documentation_files(repository, doc_result)
            
            # Update metrics
            self.metrics["documentation_generated"] += 1
            
            return {
                "success": True,
                "doc_id": doc_result.get("doc_id"),
                "repository": repository,
                "files_processed": len(files),
                "doc_type": doc_type,
                "format": format_type,
                "documentation": doc_result.get("documentation", {}),
                "markdown_output": doc_result.get("markdown_output", ""),
                "summary": doc_result.get("summary", ""),
                "coverage": doc_result.get("coverage", {})
            }
            
        except Exception as e:
            self.logger.error(f"Documentation task failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "repository": task.get("repository"),
                "files": task.get("files", [])
            }
    
    async def _execute_monitoring_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a repository monitoring task."""
        try:
            repository = task["repository"]
            action = task.get("action", "add")
            
            if action == "add":
                if repository not in self.monitored_repositories:
                    self.monitored_repositories.append(repository)
                    await self._setup_repository_monitoring(repository)
                    self.metrics["repositories_monitored"] += 1
                    
                return {
                    "success": True,
                    "action": "added",
                    "repository": repository,
                    "total_monitored": len(self.monitored_repositories)
                }
                
            elif action == "remove":
                if repository in self.monitored_repositories:
                    self.monitored_repositories.remove(repository)
                    await self._cleanup_repository_monitoring(repository)
                    self.metrics["repositories_monitored"] -= 1
                    
                return {
                    "success": True,
                    "action": "removed",
                    "repository": repository,
                    "total_monitored": len(self.monitored_repositories)
                }
                
            elif action == "status":
                status = await self._get_repository_status(repository)
                return {
                    "success": True,
                    "action": "status",
                    "repository": repository,
                    "status": status
                }
                
            else:
                raise AgentError(f"Unknown monitoring action: {action}")
                
        except Exception as e:
            self.logger.error(f"Monitoring task failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "repository": task.get("repository"),
                "action": task.get("action")
            }
    
    async def _execute_workflow_automation_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow automation task."""
        try:
            repository = task["repository"]
            workflow_type = task["workflow_type"]
            trigger_event = task.get("trigger_event")
            
            if workflow_type == "auto_merge":
                result = await self._handle_auto_merge(repository, task)
            elif workflow_type == "auto_deploy":
                result = await self._handle_auto_deploy(repository, task)
            elif workflow_type == "ci_integration":
                result = await self._handle_ci_integration(repository, task)
            elif workflow_type == "notification":
                result = await self._handle_notification(repository, task)
            else:
                raise AgentError(f"Unknown workflow type: {workflow_type}")
            
            # Update metrics
            self.metrics["workflow_automations_triggered"] += 1
            
            return {
                "success": True,
                "workflow_type": workflow_type,
                "repository": repository,
                "trigger_event": trigger_event,
                "result": result
            }
            
        except Exception as e:
            self.logger.error(f"Workflow automation task failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "repository": task.get("repository"),
                "workflow_type": task.get("workflow_type")
            }
    
    async def _handle_webhook_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle GitHub webhook messages."""
        try:
            payload = message.payload
            event_type = payload.get("event")
            action = payload.get("action")
            data = payload.get("data", {})
            
            # Queue webhook for processing
            webhook_event = WebhookEvent(
                event_type=event_type,
                action=action,
                data=data,
                timestamp=datetime.now(timezone.utc)
            )
            
            await self.webhook_queue.put(webhook_event)
            
            # Update metrics
            self.metrics["webhooks_processed"] += 1
            
            return AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                recipient=message.sender,
                message_type="webhook_processed",
                payload={
                    "event": event_type,
                    "action": action,
                    "processed_at": datetime.now(timezone.utc).isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling webhook message: {e}")
            return None
    
    async def _handle_code_review_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle code review request messages."""
        try:
            payload = message.payload
            repository = payload["repository"]
            pull_request = payload["pull_request"]
            
            # Execute code review
            task = {
                "type": "code_review",
                "repository": repository,
                "pull_request": pull_request,
                "review_type": payload.get("review_type", "full"),
                "focus_areas": payload.get("focus_areas", ["security", "performance", "style"])
            }
            
            result = await self._execute_task(task)
            
            return AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                recipient=message.sender,
                message_type="code_review_result",
                payload=result
            )
            
        except Exception as e:
            self.logger.error(f"Error handling code review request: {e}")
            return None
    
    async def _handle_vulnerability_scan_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle vulnerability scan request messages."""
        try:
            payload = message.payload
            repository = payload["repository"]
            
            # Execute vulnerability scan
            task = {
                "type": "vulnerability_scan",
                "repository": repository,
                "branch": payload.get("branch", "main"),
                "scan_type": payload.get("scan_type", "full"),
                "include_dependencies": payload.get("include_dependencies", True)
            }
            
            result = await self._execute_task(task)
            
            return AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                recipient=message.sender,
                message_type="vulnerability_scan_result",
                payload=result
            )
            
        except Exception as e:
            self.logger.error(f"Error handling vulnerability scan request: {e}")
            return None
    
    async def _handle_documentation_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle documentation generation request messages."""
        try:
            payload = message.payload
            repository = payload["repository"]
            
            # Execute documentation generation
            task = {
                "type": "generate_documentation",
                "repository": repository,
                "files": payload.get("files", []),
                "doc_type": payload.get("doc_type", "api"),
                "format": payload.get("format", "markdown"),
                "create_files": payload.get("create_files", False)
            }
            
            result = await self._execute_task(task)
            
            return AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                recipient=message.sender,
                message_type="documentation_result",
                payload=result
            )
            
        except Exception as e:
            self.logger.error(f"Error handling documentation request: {e}")
            return None
    
    async def _handle_monitoring_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle repository monitoring request messages."""
        try:
            payload = message.payload
            repository = payload["repository"]
            
            # Execute monitoring task
            task = {
                "type": "monitor_repository",
                "repository": repository,
                "action": payload.get("action", "add")
            }
            
            result = await self._execute_task(task)
            
            return AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                recipient=message.sender,
                message_type="monitoring_result",
                payload=result
            )
            
        except Exception as e:
            self.logger.error(f"Error handling monitoring request: {e}")
            return None
    
    def _validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        try:
            # Check required GitHub configuration
            github_config = config.get("github", {})
            if not github_config.get("token"):
                self.logger.error("GitHub token not provided in configuration")
                return False
            
            # Check AI analysis configuration
            ai_config = config.get("ai_analysis", {})
            if not ai_config.get("enabled", True):
                self.logger.warning("AI analysis is disabled")
            
            # Check monitoring configuration
            monitoring_config = config.get("repository_monitoring", {})
            if not monitoring_config.get("enabled", True):
                self.logger.warning("Repository monitoring is disabled")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    async def _load_monitored_repositories(self) -> None:
        """Load repositories to monitor from configuration."""
        try:
            repositories = self.monitoring_config.get("repositories", [])
            for repo in repositories:
                if repo not in self.monitored_repositories:
                    self.monitored_repositories.append(repo)
                    await self._setup_repository_monitoring(repo)
            
            self.metrics["repositories_monitored"] = len(self.monitored_repositories)
            self.logger.info(f"Loaded {len(self.monitored_repositories)} repositories for monitoring")
            
        except Exception as e:
            self.logger.error(f"Failed to load monitored repositories: {e}")
    
    async def _setup_webhook_handlers(self) -> None:
        """Setup webhook event handlers."""
        self.webhook_handlers = {
            "pull_request": self._handle_pull_request_webhook,
            "issues": self._handle_issue_webhook,
            "push": self._handle_push_webhook,
            "repository": self._handle_repository_webhook,
            "release": self._handle_release_webhook,
        }
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks."""
        if self.monitoring_config.get("enabled", True):
            self._repository_monitoring_task = asyncio.create_task(
                self._repository_monitoring_loop()
            )
        
        self._webhook_processing_task = asyncio.create_task(
            self._webhook_processing_loop()
        )
        
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop()
        )
    
    async def _stop_background_tasks(self) -> None:
        """Stop background tasks."""
        tasks = [
            self._repository_monitoring_task,
            self._webhook_processing_task,
            self._cleanup_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def _repository_monitoring_loop(self) -> None:
        """Main repository monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                for repository in self.monitored_repositories:
                    await self._monitor_repository(repository)
                
                # Wait for next monitoring cycle
                interval = self.monitoring_config.get("polling_interval", 300)
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=interval
                )
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in repository monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _webhook_processing_loop(self) -> None:
        """Main webhook processing loop."""
        while not self._shutdown_event.is_set():
            try:
                # Get webhook event from queue
                webhook_event = await asyncio.wait_for(
                    self.webhook_queue.get(),
                    timeout=1.0
                )
                
                # Process webhook event
                await self._process_webhook_event(webhook_event)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in webhook processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_cleanup()
                
                # Wait for next cleanup cycle
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=3600  # Cleanup every hour
                )
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_repository(self, repository: str) -> None:
        """Monitor a specific repository."""
        try:
            # Check for new commits
            commits = await self.github_service.get_commits(repository, limit=10)
            
            # Check for new pull requests
            pull_requests = await self.github_service.get_pull_requests(repository, state="open")
            
            # Check for new issues
            issues = await self.github_service.get_issues(repository, state="open")
            
            # Update repository state
            current_state = {
                "last_check": datetime.now(timezone.utc).isoformat(),
                "commits": len(commits),
                "pull_requests": len(pull_requests),
                "issues": len(issues)
            }
            
            # Compare with previous state
            previous_state = self.repository_states.get(repository, {})
            
            # Process changes
            await self._process_repository_changes(repository, previous_state, current_state)
            
            # Update state
            self.repository_states[repository] = current_state
            
        except Exception as e:
            self.logger.error(f"Error monitoring repository {repository}: {e}")
    
    async def _process_repository_changes(
        self, 
        repository: str, 
        previous_state: Dict[str, Any], 
        current_state: Dict[str, Any]
    ) -> None:
        """Process changes in repository state."""
        try:
            # Check for new pull requests
            prev_prs = previous_state.get("pull_requests", 0)
            curr_prs = current_state.get("pull_requests", 0)
            
            if curr_prs > prev_prs:
                # New pull request detected
                await self._handle_new_pull_requests(repository)
            
            # Check for new issues
            prev_issues = previous_state.get("issues", 0)
            curr_issues = current_state.get("issues", 0)
            
            if curr_issues > prev_issues:
                # New issue detected
                await self._handle_new_issues(repository)
            
        except Exception as e:
            self.logger.error(f"Error processing repository changes for {repository}: {e}")
    
    async def _process_webhook_event(self, webhook_event: WebhookEvent) -> None:
        """Process a webhook event."""
        try:
            handler = self.webhook_handlers.get(webhook_event.event_type)
            if handler:
                await handler(webhook_event)
            else:
                self.logger.warning(f"No handler for webhook event: {webhook_event.event_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing webhook event: {e}")
    
    async def _handle_pull_request_webhook(self, webhook_event: WebhookEvent) -> None:
        """Handle pull request webhook events."""
        try:
            action = webhook_event.action
            data = webhook_event.data
            
            if action in ["opened", "synchronize"]:
                # Automatically review pull request
                repository = data.get("repository", {}).get("full_name")
                pr_number = data.get("pull_request", {}).get("number")
                
                if repository and pr_number:
                    await self._queue_code_review(repository, pr_number)
            
        except Exception as e:
            self.logger.error(f"Error handling pull request webhook: {e}")
    
    async def _handle_issue_webhook(self, webhook_event: WebhookEvent) -> None:
        """Handle issue webhook events."""
        try:
            action = webhook_event.action
            data = webhook_event.data
            
            if action == "opened":
                # Analyze new issue
                repository = data.get("repository", {}).get("full_name")
                issue_number = data.get("issue", {}).get("number")
                
                if repository and issue_number:
                    await self._analyze_issue(repository, issue_number)
            
        except Exception as e:
            self.logger.error(f"Error handling issue webhook: {e}")
    
    async def _handle_push_webhook(self, webhook_event: WebhookEvent) -> None:
        """Handle push webhook events."""
        try:
            data = webhook_event.data
            repository = data.get("repository", {}).get("full_name")
            
            if repository:
                # Trigger security scan for push to main branch
                ref = data.get("ref", "")
                if ref == "refs/heads/main":
                    await self._queue_security_scan(repository)
            
        except Exception as e:
            self.logger.error(f"Error handling push webhook: {e}")
    
    async def _handle_repository_webhook(self, webhook_event: WebhookEvent) -> None:
        """Handle repository webhook events."""
        try:
            action = webhook_event.action
            data = webhook_event.data
            
            if action == "created":
                # New repository created
                repository = data.get("repository", {}).get("full_name")
                if repository:
                    await self._setup_new_repository(repository)
            
        except Exception as e:
            self.logger.error(f"Error handling repository webhook: {e}")
    
    async def _handle_release_webhook(self, webhook_event: WebhookEvent) -> None:
        """Handle release webhook events."""
        try:
            action = webhook_event.action
            data = webhook_event.data
            
            if action == "published":
                # New release published
                repository = data.get("repository", {}).get("full_name")
                if repository:
                    await self._handle_new_release(repository, data.get("release", {}))
            
        except Exception as e:
            self.logger.error(f"Error handling release webhook: {e}")
    
    async def _queue_code_review(self, repository: str, pr_number: int) -> None:
        """Queue a code review for processing."""
        try:
            review_request = {
                "repository": repository,
                "pull_request": pr_number,
                "review_type": "automated",
                "priority": "normal"
            }
            
            await self.review_queue.put(review_request)
            
        except Exception as e:
            self.logger.error(f"Error queueing code review: {e}")
    
    async def _perform_cleanup(self) -> None:
        """Perform periodic cleanup."""
        try:
            # Clean up old reviews
            current_time = datetime.now(timezone.utc)
            expired_reviews = []
            
            for review_id, review_data in self.active_reviews.items():
                review_time = review_data.get("created_at")
                if review_time:
                    age = current_time - datetime.fromisoformat(review_time)
                    if age.total_seconds() > 86400:  # 24 hours
                        expired_reviews.append(review_id)
            
            for review_id in expired_reviews:
                del self.active_reviews[review_id]
            
            # Clean up webhook queue if too full
            if self.webhook_queue.qsize() > 1000:
                while self.webhook_queue.qsize() > 500:
                    try:
                        self.webhook_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            
            self.logger.debug(f"Cleanup completed: removed {len(expired_reviews)} old reviews")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    # Additional helper methods would be implemented here for:
    # - _create_github_review
    # - _create_vulnerability_report
    # - _create_documentation_files
    # - _setup_repository_monitoring
    # - _cleanup_repository_monitoring
    # - _get_repository_status
    # - _handle_auto_merge
    # - _handle_auto_deploy
    # - _handle_ci_integration
    # - _handle_notification
    # - _handle_new_pull_requests
    # - _handle_new_issues
    # - _analyze_issue
    # - _queue_security_scan
    # - _setup_new_repository
    # - _handle_new_release
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics including Code Agent specific metrics."""
        base_metrics = super().get_metrics()
        
        # Add Code Agent specific metrics
        base_metrics.update({
            "active_reviews": len(self.active_reviews),
            "monitored_repositories": len(self.monitored_repositories),
            "webhook_queue_size": self.webhook_queue.qsize(),
            "review_queue_size": self.review_queue.qsize(),
        })
        
        return base_metrics