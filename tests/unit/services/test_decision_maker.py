"""
Tests for Decision Maker component following TDD principles.

This module tests the decision-making algorithms with context awareness,
confidence calculation, and learning integration.
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import MagicMock, AsyncMock

from src.services.intelligence_engine import (
    DecisionMaker,
    DecisionContext,
    Decision,
    DecisionError,
    PriorityScoring,
    ContextAwareness,
    IntelligentPrioritization,
)
from tests.mocks.intelligence_mocks import MockDecisionMaker, mock_intelligence_layer


class TestDecisionMaker:
    """Test suite for DecisionMaker component."""
    
    @pytest.fixture
    def mock_ollama_service(self):
        """Create mock Ollama service."""
        service = MagicMock()
        service.chat_completion = AsyncMock(return_value={
            "content": "High priority email detected based on urgent keywords",
            "success": True
        })
        return service
    
    @pytest.fixture
    def mock_context_awareness(self):
        """Create mock context awareness."""
        context = MagicMock()
        context.get_context = MagicMock(return_value={
            "user_patterns": {"priority_keywords": ["urgent", "asap"]},
            "historical_decisions": [],
            "current_workload": {"high": 3, "medium": 5, "low": 2}
        })
        return context
    
    @pytest.fixture
    def decision_maker(self, mock_ollama_service, mock_context_awareness):
        """Create DecisionMaker instance."""
        return DecisionMaker(
            ollama_service=mock_ollama_service,
            context_awareness=mock_context_awareness
        )
    
    @pytest.mark.asyncio
    async def test_decision_maker_initialization(self, decision_maker):
        """Test DecisionMaker initialization."""
        assert decision_maker.ollama_service is not None
        assert decision_maker.context_awareness is not None
        assert decision_maker.decision_history == []
        assert decision_maker.confidence_threshold == 0.7
        assert decision_maker.priority_scorer is not None
        
    @pytest.mark.asyncio
    async def test_email_triage_decision_urgent(self, decision_maker):
        """Test email triage decision for urgent email."""
        context = DecisionContext(
            agent_id="gmail_agent",
            task_type="email_triage",
            input_data={
                "email": {
                    "subject": "URGENT: Production server down",
                    "sender": "ops@company.com",
                    "body": "Need immediate attention - server offline",
                    "received_at": datetime.now(timezone.utc).isoformat()
                }
            },
            constraints={"max_processing_time": 30}
        )
        
        decision = await decision_maker.make_decision(context)
        
        assert isinstance(decision, Decision)
        assert decision.action == "high_priority"
        assert decision.confidence > 0.8
        assert "urgent" in decision.reasoning.lower()
        assert decision.metadata["priority_score"] > 0.8
        assert decision.metadata["confidence_factors"]["urgency_keywords"] > 0
        
    @pytest.mark.asyncio
    async def test_email_triage_decision_normal(self, decision_maker):
        """Test email triage decision for normal email."""
        context = DecisionContext(
            agent_id="gmail_agent",
            task_type="email_triage",
            input_data={
                "email": {
                    "subject": "Weekly newsletter",
                    "sender": "newsletter@company.com",
                    "body": "This week's updates and news",
                    "received_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        
        decision = await decision_maker.make_decision(context)
        
        assert isinstance(decision, Decision)
        assert decision.action == "normal_priority"
        assert decision.confidence > 0.5
        assert decision.metadata["priority_score"] < 0.7
        
    @pytest.mark.asyncio
    async def test_email_triage_decision_meeting(self, decision_maker):
        """Test email triage decision for meeting invitation."""
        context = DecisionContext(
            agent_id="gmail_agent",
            task_type="email_triage",
            input_data={
                "email": {
                    "subject": "Meeting: Project Review Tomorrow",
                    "sender": "manager@company.com",
                    "body": "Please confirm your attendance for tomorrow's meeting",
                    "received_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        
        decision = await decision_maker.make_decision(context)
        
        assert isinstance(decision, Decision)
        assert decision.action == "schedule_priority"
        assert decision.confidence > 0.7
        assert "meeting" in decision.reasoning.lower()
        
    @pytest.mark.asyncio
    async def test_research_optimization_decision(self, decision_maker):
        """Test research optimization decision."""
        context = DecisionContext(
            agent_id="research_agent",
            task_type="research_optimization",
            input_data={
                "query": "machine learning trends 2024",
                "user_preferences": {
                    "focus": "technical",
                    "depth": "detailed",
                    "sources": "academic"
                },
                "resource_constraints": {
                    "time_limit": 3600,
                    "budget": 100
                }
            }
        )
        
        decision = await decision_maker.make_decision(context)
        
        assert isinstance(decision, Decision)
        assert decision.action == "trend_analysis"
        assert decision.confidence > 0.6
        assert decision.metadata["research_strategy"] is not None
        assert decision.metadata["source_prioritization"] is not None
        
    @pytest.mark.asyncio
    async def test_code_review_prioritization(self, decision_maker):
        """Test code review prioritization decision."""
        context = DecisionContext(
            agent_id="code_agent",
            task_type="code_review_prioritization",
            input_data={
                "pull_request": {
                    "title": "Fix critical security vulnerability",
                    "author": "security_team",
                    "changes": 45,
                    "files_modified": ["auth.py", "security.py"],
                    "labels": ["security", "critical"]
                },
                "current_queue": {
                    "total_prs": 15,
                    "urgent_prs": 3,
                    "reviewer_capacity": 0.7
                }
            }
        )
        
        decision = await decision_maker.make_decision(context)
        
        assert isinstance(decision, Decision)
        assert decision.action == "immediate_review"
        assert decision.confidence > 0.9
        assert "security" in decision.reasoning.lower()
        assert decision.metadata["risk_assessment"] == "high"
        
    @pytest.mark.asyncio
    async def test_task_scheduling_decision(self, decision_maker):
        """Test task scheduling decision."""
        context = DecisionContext(
            agent_id="scheduler_agent",
            task_type="task_scheduling",
            input_data={
                "task": {
                    "type": "data_processing",
                    "priority": "medium",
                    "estimated_duration": 1800,
                    "dependencies": ["data_validation"]
                },
                "schedule": {
                    "current_load": 0.6,
                    "upcoming_tasks": 3,
                    "available_slots": ["14:00", "16:00", "18:00"]
                }
            }
        )
        
        decision = await decision_maker.make_decision(context)
        
        assert isinstance(decision, Decision)
        assert decision.action in ["schedule_now", "schedule_later", "queue_task"]
        assert decision.confidence > 0.5
        assert decision.metadata["recommended_slot"] is not None
        
    @pytest.mark.asyncio
    async def test_confidence_calculation_high_certainty(self, decision_maker):
        """Test confidence calculation with high certainty inputs."""
        context = DecisionContext(
            agent_id="test_agent",
            task_type="high_certainty_task",
            input_data={
                "clear_indicators": True,
                "historical_success": 0.95,
                "pattern_match": 0.9,
                "context_relevance": 0.85
            }
        )
        
        decision = await decision_maker.make_decision(context)
        
        assert decision.confidence > 0.85
        assert decision.metadata["confidence_factors"]["pattern_match"] > 0.8
        assert decision.metadata["confidence_factors"]["historical_success"] > 0.9
        
    @pytest.mark.asyncio
    async def test_confidence_calculation_low_certainty(self, decision_maker):
        """Test confidence calculation with low certainty inputs."""
        context = DecisionContext(
            agent_id="test_agent",
            task_type="low_certainty_task",
            input_data={
                "ambiguous_signals": True,
                "limited_history": 0.3,
                "pattern_match": 0.4,
                "context_relevance": 0.2
            }
        )
        
        decision = await decision_maker.make_decision(context)
        
        assert decision.confidence < 0.6
        assert decision.metadata["uncertainty_factors"] is not None
        assert decision.metadata["recommendation"] == "gather_more_data"
        
    @pytest.mark.asyncio
    async def test_decision_history_tracking(self, decision_maker):
        """Test decision history tracking."""
        contexts = [
            DecisionContext(
                agent_id="test_agent",
                task_type="test_task",
                input_data={"test": f"data_{i}"}
            ) for i in range(3)
        ]
        
        decisions = []
        for context in contexts:
            decision = await decision_maker.make_decision(context)
            decisions.append(decision)
        
        assert len(decision_maker.decision_history) == 3
        assert all(d.decision_id in [h.decision_id for h in decision_maker.decision_history] 
                  for d in decisions)
        
    @pytest.mark.asyncio
    async def test_context_awareness_integration(self, decision_maker):
        """Test context awareness integration."""
        context = DecisionContext(
            agent_id="gmail_agent",
            task_type="email_triage",
            input_data={
                "email": {
                    "subject": "Meeting request",
                    "sender": "important@client.com"
                }
            }
        )
        
        decision = await decision_maker.make_decision(context)
        
        # Verify context was used
        decision_maker.context_awareness.get_context.assert_called()
        assert decision.metadata["context_used"] is True
        
    @pytest.mark.asyncio
    async def test_learning_integration(self, decision_maker):
        """Test learning integration in decision making."""
        # Make initial decision
        context = DecisionContext(
            agent_id="gmail_agent",
            task_type="email_triage",
            input_data={
                "email": {
                    "subject": "Test email",
                    "sender": "test@example.com"
                }
            }
        )
        
        decision = await decision_maker.make_decision(context)
        
        # Simulate learning feedback
        feedback = {
            "decision_id": decision.decision_id,
            "feedback_type": "positive",
            "user_rating": 5,
            "outcome_metrics": {"accuracy": 0.9}
        }
        
        await decision_maker.learn_from_feedback(feedback)
        
        # Make another similar decision
        similar_context = DecisionContext(
            agent_id="gmail_agent",
            task_type="email_triage",
            input_data={
                "email": {
                    "subject": "Test email 2",
                    "sender": "test2@example.com"
                }
            }
        )
        
        new_decision = await decision_maker.make_decision(similar_context)
        
        # Confidence should be higher due to learning
        assert new_decision.confidence >= decision.confidence
        assert new_decision.metadata["learning_applied"] is True
        
    @pytest.mark.asyncio
    async def test_priority_scoring_email(self, decision_maker):
        """Test priority scoring for email."""
        email_data = {
            "subject": "URGENT: Server maintenance required",
            "sender": "admin@company.com",
            "body": "Critical system maintenance needed ASAP",
            "received_at": datetime.now(timezone.utc).isoformat()
        }
        
        priority_score = await decision_maker.calculate_priority_score(
            task_type="email_triage",
            input_data={"email": email_data}
        )
        
        assert priority_score > 0.8
        assert priority_score <= 1.0
        
    @pytest.mark.asyncio
    async def test_priority_scoring_research(self, decision_maker):
        """Test priority scoring for research task."""
        research_data = {
            "query": "urgent market analysis",
            "deadline": "2024-12-31T23:59:59Z",
            "importance": "high",
            "stakeholders": ["CEO", "board"]
        }
        
        priority_score = await decision_maker.calculate_priority_score(
            task_type="research_optimization",
            input_data=research_data
        )
        
        assert priority_score > 0.7
        assert priority_score <= 1.0
        
    @pytest.mark.asyncio
    async def test_decision_explanation_generation(self, decision_maker):
        """Test decision explanation generation."""
        context = DecisionContext(
            agent_id="gmail_agent",
            task_type="email_triage",
            input_data={
                "email": {
                    "subject": "URGENT: Need help",
                    "sender": "client@important.com",
                    "body": "This is urgent"
                }
            }
        )
        
        decision = await decision_maker.make_decision(context)
        
        assert decision.reasoning is not None
        assert len(decision.reasoning) > 20  # Meaningful explanation
        assert "urgent" in decision.reasoning.lower()
        assert decision.metadata["explanation_confidence"] > 0.7
        
    @pytest.mark.asyncio
    async def test_error_handling_invalid_context(self, decision_maker):
        """Test error handling with invalid context."""
        with pytest.raises(DecisionError):
            await decision_maker.make_decision(None)
            
    @pytest.mark.asyncio
    async def test_error_handling_missing_data(self, decision_maker):
        """Test error handling with missing data."""
        context = DecisionContext(
            agent_id="test_agent",
            task_type="email_triage",
            input_data={}  # Missing email data
        )
        
        with pytest.raises(DecisionError) as exc_info:
            await decision_maker.make_decision(context)
        
        assert "missing required data" in str(exc_info.value).lower()
        
    @pytest.mark.asyncio
    async def test_decision_caching(self, decision_maker):
        """Test decision caching for similar contexts."""
        context1 = DecisionContext(
            agent_id="test_agent",
            task_type="simple_task",
            input_data={"key": "value"}
        )
        
        context2 = DecisionContext(
            agent_id="test_agent",
            task_type="simple_task",
            input_data={"key": "value"}
        )
        
        decision1 = await decision_maker.make_decision(context1)
        decision2 = await decision_maker.make_decision(context2)
        
        # Should use cache for similar contexts
        assert decision2.metadata.get("from_cache") is True
        assert decision1.action == decision2.action
        
    @pytest.mark.asyncio
    async def test_multi_criteria_decision(self, decision_maker):
        """Test multi-criteria decision making."""
        context = DecisionContext(
            agent_id="complex_agent",
            task_type="multi_criteria_task",
            input_data={
                "criteria": {
                    "urgency": 0.8,
                    "importance": 0.7,
                    "complexity": 0.6,
                    "resource_availability": 0.9
                },
                "weights": {
                    "urgency": 0.3,
                    "importance": 0.4,
                    "complexity": 0.2,
                    "resource_availability": 0.1
                }
            }
        )
        
        decision = await decision_maker.make_decision(context)
        
        assert decision.metadata["criteria_scores"] is not None
        assert decision.metadata["weighted_score"] > 0.0
        assert decision.metadata["decision_method"] == "multi_criteria"
        
    @pytest.mark.asyncio
    async def test_adaptive_confidence_threshold(self, decision_maker):
        """Test adaptive confidence threshold adjustment."""
        # Simulate several successful decisions
        for i in range(5):
            context = DecisionContext(
                agent_id="adaptive_agent",
                task_type="adaptive_task",
                input_data={"iteration": i}
            )
            
            decision = await decision_maker.make_decision(context)
            
            # Simulate positive feedback
            feedback = {
                "decision_id": decision.decision_id,
                "feedback_type": "positive",
                "user_rating": 5,
                "outcome_metrics": {"accuracy": 0.9}
            }
            
            await decision_maker.learn_from_feedback(feedback)
        
        # Confidence threshold should adapt
        assert decision_maker.confidence_threshold != 0.7  # Default
        assert decision_maker.adaptation_history is not None