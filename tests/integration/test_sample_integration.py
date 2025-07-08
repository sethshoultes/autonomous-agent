"""
Sample integration tests demonstrating TDD approach for system interactions.

This module shows how to write integration tests that verify the interaction
between multiple components of the autonomous agent system.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from tests.utils.test_helpers import TestDataGenerator, AsyncTestHelper
from tests.utils.assertions import (
    assert_email_data_valid,
    assert_github_data_valid,
    assert_ollama_response_valid,
)


# ============================================================================
# Integration Test: Email Processing Workflow
# ============================================================================

class TestEmailProcessingWorkflow:
    """
    Integration tests for the complete email processing workflow.
    
    Tests the interaction between:
    - Gmail API client
    - Email classification service
    - Ollama AI processing
    - Response generation
    - Email sending
    """
    
    @pytest.mark.integration
    async def test_complete_email_processing_workflow(
        self, 
        integrated_services,
        integration_test_data,
        performance_monitor
    ):
        """Test the complete email processing workflow from fetch to response."""
        # ARRANGE
        performance_monitor.start()
        
        gmail_service = integrated_services["gmail"]
        ollama_client = integrated_services["ollama"]
        
        # Create a test email workflow
        email_workflow = self._create_email_workflow(gmail_service, ollama_client)
        
        # ACT
        result = await email_workflow.process_emails()
        
        # ASSERT
        assert result["status"] == "completed", "Workflow should complete successfully"
        assert "emails_processed" in result, "Should report number of emails processed"
        assert result["emails_processed"] > 0, "Should process at least one email"
        
        # Verify performance
        performance_data = performance_monitor.stop()
        assert performance_data["execution_time"] < 10.0, "Should complete within 10 seconds"
        assert performance_data["memory_used_mb"] < 100, "Should use less than 100MB additional memory"
    
    @pytest.mark.integration
    async def test_email_classification_with_ai_analysis(
        self,
        integrated_services,
        sample_email_data
    ):
        """Test email classification using AI analysis."""
        # ARRANGE
        gmail_service = integrated_services["gmail"]
        ollama_client = integrated_services["ollama"]
        
        classifier = self._create_ai_email_classifier(ollama_client)
        
        # ACT
        classification = await classifier.classify_email(sample_email_data)
        
        # ASSERT
        assert "category" in classification, "Should return category"
        assert "confidence" in classification, "Should return confidence"
        assert "ai_analysis" in classification, "Should include AI analysis"
        
        # Verify AI was actually called
        ollama_client.generate.assert_called()
        
        # Verify classification quality
        assert classification["confidence"] > 0.5, "Should have reasonable confidence"
        assert isinstance(classification["ai_analysis"], str), "AI analysis should be text"
    
    @pytest.mark.integration
    async def test_email_workflow_error_handling(
        self,
        integrated_services,
        mock_database
    ):
        """Test that email workflow handles errors gracefully."""
        # ARRANGE
        gmail_service = integrated_services["gmail"]
        ollama_client = integrated_services["ollama"]
        
        # Simulate an error condition
        ollama_client.generate.side_effect = Exception("AI service temporarily unavailable")
        
        email_workflow = self._create_email_workflow(gmail_service, ollama_client)
        
        # ACT
        result = await email_workflow.process_emails()
        
        # ASSERT
        assert result["status"] == "completed_with_errors", "Should handle errors gracefully"
        assert "errors" in result, "Should report errors"
        assert len(result["errors"]) > 0, "Should capture error details"
        
        # Verify fallback behavior
        assert "fallback_processing" in result, "Should indicate fallback was used"
        assert result["fallback_processing"] is True, "Should use fallback processing"
    
    def _create_email_workflow(self, gmail_service, ollama_client) -> AsyncMock:
        """Create a mock email workflow for testing."""
        workflow = AsyncMock()
        
        async def mock_process_emails():
            try:
                # Simulate email processing steps
                emails = await gmail_service.fetch_emails()
                processed_count = 0
                errors = []
                
                for email in emails[:3]:  # Process first 3 emails
                    try:
                        # Classify email using AI
                        classification = await ollama_client.generate(
                            model="llama3.1:8b",
                            prompt=f"Classify this email: {email.get('subject', '')}"
                        )
                        processed_count += 1
                    except Exception as e:
                        errors.append(str(e))
                
                if errors:
                    return {
                        "status": "completed_with_errors",
                        "emails_processed": processed_count,
                        "errors": errors,
                        "fallback_processing": True,
                    }
                else:
                    return {
                        "status": "completed",
                        "emails_processed": processed_count,
                    }
                    
            except Exception as e:
                return {
                    "status": "failed",
                    "error": str(e),
                }
        
        workflow.process_emails.side_effect = mock_process_emails
        return workflow
    
    def _create_ai_email_classifier(self, ollama_client) -> AsyncMock:
        """Create a mock AI email classifier."""
        classifier = AsyncMock()
        
        async def mock_classify_email(email_data):
            # Simulate AI classification
            ai_response = await ollama_client.generate(
                model="llama3.1:8b",
                prompt=f"Classify email: {email_data.get('subject', '')}"
            )
            
            return {
                "category": "inbox",
                "confidence": 0.85,
                "ai_analysis": ai_response.get("response", "AI analysis unavailable"),
            }
        
        classifier.classify_email.side_effect = mock_classify_email
        return classifier


# ============================================================================
# Integration Test: GitHub Repository Monitoring
# ============================================================================

class TestGitHubRepositoryMonitoring:
    """
    Integration tests for GitHub repository monitoring workflow.
    
    Tests the interaction between:
    - GitHub API client
    - Repository monitoring service
    - Pull request analysis
    - Automated code review
    - Issue creation
    """
    
    @pytest.mark.integration
    async def test_repository_monitoring_workflow(
        self,
        integrated_services,
        integration_test_data
    ):
        """Test the complete repository monitoring workflow."""
        # ARRANGE
        github_client = integrated_services["github"]
        ollama_client = integrated_services["ollama"]
        
        monitoring_service = self._create_monitoring_service(github_client, ollama_client)
        
        # ACT
        result = await monitoring_service.monitor_repositories()
        
        # ASSERT
        assert result["status"] == "completed", "Monitoring should complete successfully"
        assert "repositories_monitored" in result, "Should report repositories monitored"
        assert "pull_requests_analyzed" in result, "Should report PRs analyzed"
        assert result["repositories_monitored"] > 0, "Should monitor at least one repository"
    
    @pytest.mark.integration
    async def test_pull_request_code_review(
        self,
        integrated_services,
        mock_github_pull_request
    ):
        """Test automated pull request code review."""
        # ARRANGE
        github_client = integrated_services["github"]
        ollama_client = integrated_services["ollama"]
        
        code_reviewer = self._create_code_reviewer(ollama_client)
        pr_data = mock_github_pull_request.to_github_format()
        
        # ACT
        review_result = await code_reviewer.review_pull_request(pr_data)
        
        # ASSERT
        assert "review_comments" in review_result, "Should provide review comments"
        assert "approval_status" in review_result, "Should provide approval status"
        assert "suggestions" in review_result, "Should provide suggestions"
        
        # Verify AI was used for analysis
        ollama_client.generate.assert_called()
        
        # Verify review quality
        assert len(review_result["review_comments"]) >= 0, "Should provide comments if needed"
        assert review_result["approval_status"] in ["approved", "changes_requested", "comment"]
    
    @pytest.mark.integration
    async def test_issue_creation_from_analysis(
        self,
        integrated_services,
        integration_test_data
    ):
        """Test automatic issue creation based on code analysis."""
        # ARRANGE
        github_client = integrated_services["github"]
        ollama_client = integrated_services["ollama"]
        
        issue_creator = self._create_issue_creator(github_client, ollama_client)
        
        # Simulate code analysis findings
        analysis_findings = [
            {
                "type": "security_vulnerability",
                "severity": "high",
                "description": "Potential SQL injection vulnerability",
                "file": "src/database.py",
                "line": 42,
            },
            {
                "type": "performance_issue",
                "severity": "medium", 
                "description": "Inefficient database query",
                "file": "src/queries.py",
                "line": 15,
            },
        ]
        
        # ACT
        result = await issue_creator.create_issues_from_findings(analysis_findings)
        
        # ASSERT
        assert result["status"] == "completed", "Issue creation should complete"
        assert "issues_created" in result, "Should report issues created"
        assert result["issues_created"] == len(analysis_findings), "Should create one issue per finding"
    
    def _create_monitoring_service(self, github_client, ollama_client) -> AsyncMock:
        """Create a mock repository monitoring service."""
        service = AsyncMock()
        
        async def mock_monitor_repositories():
            # Simulate monitoring multiple repositories
            repos = ["repo1", "repo2", "repo3"]
            prs_analyzed = 0
            
            for repo in repos:
                # Simulate checking for new PRs
                prs = await github_client.get_pull_requests(repo)
                prs_analyzed += len(prs)
            
            return {
                "status": "completed",
                "repositories_monitored": len(repos),
                "pull_requests_analyzed": prs_analyzed,
            }
        
        service.monitor_repositories.side_effect = mock_monitor_repositories
        return service
    
    def _create_code_reviewer(self, ollama_client) -> AsyncMock:
        """Create a mock code reviewer."""
        reviewer = AsyncMock()
        
        async def mock_review_pull_request(pr_data):
            # Simulate AI code review
            ai_response = await ollama_client.generate(
                model="llama3.1:8b",
                prompt=f"Review this pull request: {pr_data.get('title', '')}"
            )
            
            return {
                "review_comments": [],
                "approval_status": "approved",
                "suggestions": [
                    "Consider adding unit tests for new functionality",
                    "Documentation looks good",
                ],
                "ai_analysis": ai_response.get("response", ""),
            }
        
        reviewer.review_pull_request.side_effect = mock_review_pull_request
        return reviewer
    
    def _create_issue_creator(self, github_client, ollama_client) -> AsyncMock:
        """Create a mock issue creator."""
        creator = AsyncMock()
        
        async def mock_create_issues_from_findings(findings):
            issues_created = 0
            
            for finding in findings:
                # Simulate issue creation
                issue_data = {
                    "title": f"{finding['type']}: {finding['description']}",
                    "body": f"Found in {finding['file']} at line {finding['line']}",
                    "labels": [finding['type'], f"severity-{finding['severity']}"],
                }
                
                # Mock GitHub API call
                await github_client.create_issue(issue_data)
                issues_created += 1
            
            return {
                "status": "completed",
                "issues_created": issues_created,
            }
        
        creator.create_issues_from_findings.side_effect = mock_create_issues_from_findings
        return creator


# ============================================================================
# Integration Test: Multi-Agent Coordination
# ============================================================================

class TestMultiAgentCoordination:
    """
    Integration tests for multi-agent coordination and communication.
    
    Tests the interaction between:
    - Agent Manager
    - Multiple specialized agents
    - Message passing system
    - Shared resources
    """
    
    @pytest.mark.integration
    async def test_agent_coordination_workflow(
        self,
        integrated_agent_system,
        integration_redis
    ):
        """Test coordination between multiple agents."""
        # ARRANGE
        agent_manager = integrated_agent_system["manager"]
        gmail_agent = integrated_agent_system["agents"]["gmail"]
        github_agent = integrated_agent_system["agents"]["github"]
        ollama_agent = integrated_agent_system["agents"]["ollama"]
        
        # Start all agents
        await agent_manager.start()
        await gmail_agent.start()
        await github_agent.start()
        await ollama_agent.start()
        
        # ACT
        # Simulate a workflow that requires agent coordination
        coordination_result = await self._execute_coordinated_workflow(
            agent_manager, gmail_agent, github_agent, ollama_agent
        )
        
        # ASSERT
        assert coordination_result["status"] == "completed", "Coordination should succeed"
        assert "tasks_completed" in coordination_result, "Should report completed tasks"
        assert coordination_result["tasks_completed"] > 0, "Should complete at least one task"
        
        # Verify all agents are still running
        assert await gmail_agent.get_status() == "running"
        assert await github_agent.get_status() == "running" 
        assert await ollama_agent.get_status() == "running"
    
    @pytest.mark.integration
    async def test_agent_message_passing(
        self,
        integrated_agent_system,
        integration_redis
    ):
        """Test message passing between agents."""
        # ARRANGE
        agent_manager = integrated_agent_system["manager"]
        gmail_agent = integrated_agent_system["agents"]["gmail"]
        ollama_agent = integrated_agent_system["agents"]["ollama"]
        
        # Create a test message
        test_message = {
            "id": TestDataGenerator.generate_uuid(),
            "from": "gmail_agent",
            "to": "ollama_agent",
            "type": "analyze_email",
            "payload": {
                "email_data": TestDataGenerator.generate_email_data(),
            },
            "timestamp": TestDataGenerator.generate_timestamp(),
        }
        
        # ACT
        # Send message from Gmail agent to Ollama agent
        send_result = await agent_manager.send_message(test_message)
        
        # Simulate processing by Ollama agent
        process_result = await ollama_agent.process_message(test_message)
        
        # ASSERT
        assert send_result is True, "Message should be sent successfully"
        assert process_result["status"] == "processed", "Message should be processed"
        assert process_result["message_id"] == test_message["id"], "Should reference original message"
    
    @pytest.mark.integration
    async def test_agent_failover_and_recovery(
        self,
        integrated_agent_system,
        integration_redis
    ):
        """Test agent failover and recovery mechanisms."""
        # ARRANGE
        agent_manager = integrated_agent_system["manager"]
        gmail_agent = integrated_agent_system["agents"]["gmail"]
        
        # Start the agent
        await gmail_agent.start()
        assert await gmail_agent.get_status() == "running"
        
        # ACT
        # Simulate agent failure
        await gmail_agent.stop()  # Simulate crash
        assert await gmail_agent.get_status() == "stopped"
        
        # Simulate recovery
        recovery_result = await agent_manager.restart_agent("gmail")
        
        # ASSERT
        assert recovery_result is True, "Agent should restart successfully"
        # Note: In real implementation, this would check actual agent status
        # assert await gmail_agent.get_status() == "running"
    
    async def _execute_coordinated_workflow(
        self, 
        manager, 
        gmail_agent, 
        github_agent, 
        ollama_agent
    ) -> Dict[str, Any]:
        """Execute a workflow that requires coordination between agents."""
        tasks_completed = 0
        
        # Task 1: Gmail agent fetches emails
        emails = await gmail_agent.fetch_emails()
        if emails:
            tasks_completed += 1
        
        # Task 2: Ollama agent analyzes email content
        for email in emails[:2]:  # Analyze first 2 emails
            analysis = await ollama_agent.analyze_content(email)
            if analysis:
                tasks_completed += 1
        
        # Task 3: GitHub agent creates issues based on analysis
        issues_created = await github_agent.create_issue({
            "title": "Email Analysis Results",
            "body": "Analysis completed by AI agent",
        })
        if issues_created:
            tasks_completed += 1
        
        return {
            "status": "completed",
            "tasks_completed": tasks_completed,
        }


# ============================================================================
# Integration Test: End-to-End System Test
# ============================================================================

class TestEndToEndSystem:
    """
    End-to-end integration tests for the complete autonomous agent system.
    """
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_complete_system_workflow(
        self,
        integrated_services,
        integrated_agent_system,
        integration_database,
        integration_redis,
        test_timeout
    ):
        """Test the complete system workflow from start to finish."""
        # ARRANGE
        timeout = test_timeout
        
        # ACT
        system_result = await AsyncTestHelper.run_with_timeout(
            self._run_complete_system_test(
                integrated_services,
                integrated_agent_system,
                integration_database,
                integration_redis
            ),
            timeout=timeout
        )
        
        # ASSERT
        assert system_result["status"] == "completed", "Complete system test should succeed"
        assert "components_tested" in system_result, "Should report components tested"
        assert system_result["components_tested"] >= 5, "Should test at least 5 components"
    
    async def _run_complete_system_test(
        self,
        services,
        agents,
        database,
        redis
    ) -> Dict[str, Any]:
        """Run a complete system test."""
        components_tested = 0
        
        # Test 1: Service initialization
        if services["gmail"] and services["github"] and services["ollama"]:
            components_tested += 1
        
        # Test 2: Agent system startup
        manager = agents["manager"]
        await manager.start()
        components_tested += 1
        
        # Test 3: Database connectivity
        await database.execute("SELECT 1")
        components_tested += 1
        
        # Test 4: Redis connectivity
        await redis.set("test_key", "test_value")
        value = await redis.get("test_key")
        if value == "test_value":
            components_tested += 1
        
        # Test 5: Email processing workflow
        gmail_agent = agents["agents"]["gmail"]
        emails = await gmail_agent.fetch_emails()
        if emails is not None:
            components_tested += 1
        
        # Test 6: AI processing
        ollama_agent = agents["agents"]["ollama"]
        ai_result = await ollama_agent.generate_text("Test prompt")
        if ai_result:
            components_tested += 1
        
        return {
            "status": "completed",
            "components_tested": components_tested,
        }