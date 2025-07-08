"""
Comprehensive tests for Intelligence Engine following TDD principles.

This module tests the core intelligence layer including decision-making algorithms,
multi-agent coordination, learning mechanisms, and task planning capabilities.
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from src.services.intelligence_engine import (
    IntelligenceEngine,
    DecisionMaker,
    TaskPlanner,
    LearningSystem,
    AgentCoordinator,
    DecisionContext,
    TaskPlan,
    LearningMetrics,
    CoordinationResult,
    IntelligenceError,
    DecisionError,
    CoordinationError,
    LearningError,
    Decision,
    TaskStep,
    AgentWorkload,
    PerformanceMetrics,
    UserPreferences,
    PatternRecognition,
    WorkflowOptimization,
    PriorityScoring,
    ResourceAllocation,
    ConflictResolution,
    KnowledgeBase,
    AdaptationEngine,
    ContextAwareness,
    MultiStepPlanning,
    WorkflowOrchestration,
    IntelligentPrioritization,
    CrossAgentLearning,
    AutomatedOptimization,
)
from src.config.manager import ConfigManager
from src.logging.manager import LoggingManager
from tests.mocks.ollama_mocks import MockOllamaClient, mock_ollama_api


class TestIntelligenceEngine:
    """Test suite for IntelligenceEngine class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: {
            "intelligence.enabled": True,
            "intelligence.learning_rate": 0.1,
            "intelligence.max_context_length": 8192,
            "intelligence.decision_threshold": 0.7,
            "intelligence.coordination_timeout": 30,
            "intelligence.cache_size": 1000,
            "intelligence.optimization_interval": 300,
            "intelligence.metrics_retention_days": 30,
        }.get(key, default)
        return config
    
    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MagicMock()
    
    @pytest.fixture
    def mock_ollama_service(self):
        """Create mock Ollama service."""
        service = MagicMock()
        service.chat_completion = AsyncMock()
        service.generate_text = AsyncMock()
        service.model_manager = MagicMock()
        service.conversation_manager = MagicMock()
        return service
    
    @pytest.fixture
    def mock_message_broker(self):
        """Create mock message broker."""
        broker = MagicMock()
        broker.publish = AsyncMock()
        broker.subscribe = AsyncMock()
        broker.get_agent_status = AsyncMock()
        return broker
    
    @pytest.fixture
    def intelligence_engine(self, mock_config, mock_logger, mock_ollama_service, mock_message_broker):
        """Create IntelligenceEngine instance."""
        return IntelligenceEngine(
            config=mock_config,
            logger=mock_logger,
            ollama_service=mock_ollama_service,
            message_broker=mock_message_broker
        )
    
    @pytest.mark.asyncio
    async def test_intelligence_engine_initialization(self, intelligence_engine):
        """Test IntelligenceEngine initialization."""
        assert intelligence_engine.config is not None
        assert intelligence_engine.logger is not None
        assert intelligence_engine.ollama_service is not None
        assert intelligence_engine.message_broker is not None
        assert intelligence_engine.decision_maker is not None
        assert intelligence_engine.task_planner is not None
        assert intelligence_engine.learning_system is not None
        assert intelligence_engine.agent_coordinator is not None
        assert intelligence_engine.is_initialized is False
        
    @pytest.mark.asyncio
    async def test_intelligence_engine_initialization_process(self, intelligence_engine):
        """Test IntelligenceEngine initialization process."""
        await intelligence_engine.initialize()
        
        assert intelligence_engine.is_initialized is True
        assert intelligence_engine.knowledge_base is not None
        assert intelligence_engine.context_awareness is not None
        assert intelligence_engine.pattern_recognition is not None
        
    @pytest.mark.asyncio
    async def test_intelligence_engine_shutdown(self, intelligence_engine):
        """Test IntelligenceEngine shutdown process."""
        await intelligence_engine.initialize()
        await intelligence_engine.shutdown()
        
        assert intelligence_engine.is_initialized is False
        
    @pytest.mark.asyncio
    async def test_make_decision_simple_context(self, intelligence_engine):
        """Test making a decision with simple context."""
        await intelligence_engine.initialize()
        
        context = DecisionContext(
            agent_id="test_agent",
            task_type="email_triage",
            input_data={"email": {"subject": "Urgent: Meeting tomorrow", "sender": "boss@company.com"}},
            constraints={"max_processing_time": 30}
        )
        
        decision = await intelligence_engine.make_decision(context)
        
        assert isinstance(decision, Decision)
        assert decision.action is not None
        assert decision.confidence > 0
        assert decision.reasoning is not None
        assert decision.metadata is not None
        
    @pytest.mark.asyncio
    async def test_make_decision_complex_context(self, intelligence_engine):
        """Test making a decision with complex context."""
        await intelligence_engine.initialize()
        
        context = DecisionContext(
            agent_id="research_agent",
            task_type="research_optimization",
            input_data={
                "query": "artificial intelligence trends 2024",
                "user_preferences": {"focus": "technical", "depth": "detailed"},
                "past_searches": ["AI research", "machine learning"],
                "resource_constraints": {"budget": 1000, "time_limit": 3600}
            },
            constraints={"accuracy_threshold": 0.8, "source_diversity": True}
        )
        
        decision = await intelligence_engine.make_decision(context)
        
        assert isinstance(decision, Decision)
        assert decision.action is not None
        assert decision.confidence >= 0.8
        assert "research_strategy" in decision.metadata
        
    @pytest.mark.asyncio
    async def test_create_task_plan_simple(self, intelligence_engine):
        """Test creating a simple task plan."""
        await intelligence_engine.initialize()
        
        task_definition = {
            "goal": "Process incoming emails",
            "constraints": {"max_time": 300, "priority": "high"},
            "resources": {"agents": ["gmail_agent"], "budget": 100}
        }
        
        plan = await intelligence_engine.create_task_plan(task_definition)
        
        assert isinstance(plan, TaskPlan)
        assert len(plan.steps) > 0
        assert plan.estimated_duration > 0
        assert plan.required_resources is not None
        
    @pytest.mark.asyncio
    async def test_create_task_plan_multi_agent(self, intelligence_engine):
        """Test creating a multi-agent task plan."""
        await intelligence_engine.initialize()
        
        task_definition = {
            "goal": "Research and summarize AI trends, then send email report",
            "constraints": {"deadline": "2024-12-31T23:59:59Z", "quality": "high"},
            "resources": {"agents": ["research_agent", "gmail_agent"], "budget": 500}
        }
        
        plan = await intelligence_engine.create_task_plan(task_definition)
        
        assert isinstance(plan, TaskPlan)
        assert len(plan.steps) >= 2
        assert plan.agent_assignments is not None
        assert "research_agent" in plan.agent_assignments
        assert "gmail_agent" in plan.agent_assignments
        
    @pytest.mark.asyncio
    async def test_coordinate_agents_simple(self, intelligence_engine):
        """Test simple agent coordination."""
        await intelligence_engine.initialize()
        
        coordination_request = {
            "task_id": "task_123",
            "agents": ["agent1", "agent2"],
            "coordination_type": "parallel",
            "timeout": 60
        }
        
        result = await intelligence_engine.coordinate_agents(coordination_request)
        
        assert isinstance(result, CoordinationResult)
        assert result.success is True
        assert len(result.agent_results) == 2
        
    @pytest.mark.asyncio
    async def test_coordinate_agents_with_conflict(self, intelligence_engine):
        """Test agent coordination with conflict resolution."""
        await intelligence_engine.initialize()
        
        coordination_request = {
            "task_id": "task_456",
            "agents": ["agent1", "agent2"],
            "coordination_type": "sequential",
            "conflict_resolution": "priority_based",
            "timeout": 120
        }
        
        result = await intelligence_engine.coordinate_agents(coordination_request)
        
        assert isinstance(result, CoordinationResult)
        assert result.conflicts_resolved is not None
        
    @pytest.mark.asyncio
    async def test_learn_from_feedback_positive(self, intelligence_engine):
        """Test learning from positive feedback."""
        await intelligence_engine.initialize()
        
        feedback = {
            "task_id": "task_789",
            "decision_id": "decision_abc",
            "feedback_type": "positive",
            "user_rating": 5,
            "outcome_metrics": {"accuracy": 0.95, "speed": 0.8},
            "user_comment": "Excellent prioritization of emails"
        }
        
        learning_result = await intelligence_engine.learn_from_feedback(feedback)
        
        assert learning_result is not None
        assert learning_result.get("learning_applied") is True
        assert learning_result.get("confidence_adjustment") > 0
        
    @pytest.mark.asyncio
    async def test_learn_from_feedback_negative(self, intelligence_engine):
        """Test learning from negative feedback."""
        await intelligence_engine.initialize()
        
        feedback = {
            "task_id": "task_101",
            "decision_id": "decision_def",
            "feedback_type": "negative",
            "user_rating": 2,
            "outcome_metrics": {"accuracy": 0.3, "speed": 0.9},
            "user_comment": "Missed important email priority"
        }
        
        learning_result = await intelligence_engine.learn_from_feedback(feedback)
        
        assert learning_result is not None
        assert learning_result.get("learning_applied") is True
        assert learning_result.get("confidence_adjustment") < 0
        
    @pytest.mark.asyncio
    async def test_optimize_workflow_email_processing(self, intelligence_engine):
        """Test workflow optimization for email processing."""
        await intelligence_engine.initialize()
        
        workflow_data = {
            "workflow_type": "email_processing",
            "current_performance": {
                "processing_time": 120,
                "accuracy": 0.75,
                "user_satisfaction": 0.6
            },
            "historical_data": [
                {"timestamp": "2024-01-01", "processing_time": 150, "accuracy": 0.7},
                {"timestamp": "2024-01-02", "processing_time": 130, "accuracy": 0.72},
                {"timestamp": "2024-01-03", "processing_time": 125, "accuracy": 0.74}
            ]
        }
        
        optimization_result = await intelligence_engine.optimize_workflow(workflow_data)
        
        assert optimization_result is not None
        assert optimization_result.get("optimizations_found") is True
        assert "recommended_changes" in optimization_result
        
    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, intelligence_engine):
        """Test getting performance metrics."""
        await intelligence_engine.initialize()
        
        metrics = await intelligence_engine.get_performance_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_decisions >= 0
        assert metrics.average_confidence >= 0
        assert metrics.learning_progress >= 0
        assert metrics.coordination_success_rate >= 0
        
    @pytest.mark.asyncio
    async def test_get_agent_insights(self, intelligence_engine):
        """Test getting agent insights."""
        await intelligence_engine.initialize()
        
        insights = await intelligence_engine.get_agent_insights("gmail_agent")
        
        assert isinstance(insights, dict)
        assert "performance" in insights
        assert "learning_progress" in insights
        assert "optimization_suggestions" in insights
        
    @pytest.mark.asyncio
    async def test_update_user_preferences(self, intelligence_engine):
        """Test updating user preferences."""
        await intelligence_engine.initialize()
        
        preferences = {
            "email_priority_factors": {"sender_importance": 0.8, "urgency_keywords": 0.7},
            "research_preferences": {"depth": "detailed", "sources": "academic"},
            "workflow_preferences": {"speed_vs_accuracy": "balanced"}
        }
        
        result = await intelligence_engine.update_user_preferences(preferences)
        
        assert result is True
        
    @pytest.mark.asyncio
    async def test_predict_optimal_actions(self, intelligence_engine):
        """Test predicting optimal actions."""
        await intelligence_engine.initialize()
        
        prediction_context = {
            "current_state": {
                "inbox_size": 50,
                "urgent_emails": 5,
                "available_agents": ["gmail_agent", "research_agent"]
            },
            "historical_patterns": [
                {"timestamp": "2024-01-01", "action": "prioritize_urgent", "outcome": "success"},
                {"timestamp": "2024-01-02", "action": "batch_process", "outcome": "partial_success"}
            ]
        }
        
        predictions = await intelligence_engine.predict_optimal_actions(prediction_context)
        
        assert isinstance(predictions, list)
        assert len(predictions) > 0
        assert all("action" in pred and "confidence" in pred for pred in predictions)
        
    @pytest.mark.asyncio
    async def test_error_handling_invalid_context(self, intelligence_engine):
        """Test error handling with invalid context."""
        await intelligence_engine.initialize()
        
        with pytest.raises(IntelligenceError):
            await intelligence_engine.make_decision(None)
            
    @pytest.mark.asyncio
    async def test_error_handling_coordination_timeout(self, intelligence_engine):
        """Test error handling with coordination timeout."""
        await intelligence_engine.initialize()
        
        coordination_request = {
            "task_id": "task_timeout",
            "agents": ["slow_agent"],
            "coordination_type": "sequential",
            "timeout": 0.1  # Very short timeout
        }
        
        with pytest.raises(CoordinationError):
            await intelligence_engine.coordinate_agents(coordination_request)


class TestDecisionMaker:
    """Test suite for DecisionMaker class."""
    
    @pytest.fixture
    def mock_ollama_service(self):
        """Create mock Ollama service."""
        service = MagicMock()
        service.chat_completion = AsyncMock()
        service.generate_text = AsyncMock()
        return service
    
    @pytest.fixture
    def decision_maker(self, mock_ollama_service):
        """Create DecisionMaker instance."""
        return DecisionMaker(mock_ollama_service)
    
    @pytest.mark.asyncio
    async def test_decision_maker_initialization(self, decision_maker):
        """Test DecisionMaker initialization."""
        assert decision_maker.ollama_service is not None
        assert decision_maker.decision_history == []
        assert decision_maker.confidence_threshold == 0.7
        
    @pytest.mark.asyncio
    async def test_make_decision_email_triage(self, decision_maker):
        """Test decision making for email triage."""
        context = DecisionContext(
            agent_id="gmail_agent",
            task_type="email_triage",
            input_data={
                "email": {
                    "subject": "URGENT: Server down",
                    "sender": "ops@company.com",
                    "body": "Production server is experiencing issues"
                }
            }
        )
        
        decision = await decision_maker.make_decision(context)
        
        assert isinstance(decision, Decision)
        assert decision.action == "high_priority"
        assert decision.confidence > 0.8
        assert "urgency" in decision.reasoning.lower()
        
    @pytest.mark.asyncio
    async def test_make_decision_research_optimization(self, decision_maker):
        """Test decision making for research optimization."""
        context = DecisionContext(
            agent_id="research_agent",
            task_type="research_optimization",
            input_data={
                "query": "climate change solutions",
                "user_preferences": {"focus": "technical", "sources": "academic"},
                "resource_constraints": {"time_limit": 3600}
            }
        )
        
        decision = await decision_maker.make_decision(context)
        
        assert isinstance(decision, Decision)
        assert decision.action is not None
        assert decision.confidence > 0.5
        assert "strategy" in decision.metadata
        
    @pytest.mark.asyncio
    async def test_decision_confidence_calculation(self, decision_maker):
        """Test decision confidence calculation."""
        context = DecisionContext(
            agent_id="test_agent",
            task_type="simple_task",
            input_data={"clear_signal": True, "confidence_factors": [0.9, 0.8, 0.85]}
        )
        
        decision = await decision_maker.make_decision(context)
        
        assert decision.confidence > 0.7
        assert decision.confidence <= 1.0
        
    @pytest.mark.asyncio
    async def test_decision_history_tracking(self, decision_maker):
        """Test decision history tracking."""
        context = DecisionContext(
            agent_id="test_agent",
            task_type="test_task",
            input_data={"test": "data"}
        )
        
        decision1 = await decision_maker.make_decision(context)
        decision2 = await decision_maker.make_decision(context)
        
        assert len(decision_maker.decision_history) == 2
        assert decision_maker.decision_history[0].decision_id == decision1.decision_id
        assert decision_maker.decision_history[1].decision_id == decision2.decision_id


class TestTaskPlanner:
    """Test suite for TaskPlanner class."""
    
    @pytest.fixture
    def task_planner(self):
        """Create TaskPlanner instance."""
        return TaskPlanner()
    
    @pytest.mark.asyncio
    async def test_task_planner_initialization(self, task_planner):
        """Test TaskPlanner initialization."""
        assert task_planner.planning_strategies is not None
        assert task_planner.resource_estimator is not None
        assert task_planner.dependency_resolver is not None
        
    @pytest.mark.asyncio
    async def test_create_simple_plan(self, task_planner):
        """Test creating a simple task plan."""
        task_definition = {
            "goal": "Process 10 emails",
            "constraints": {"max_time": 300},
            "resources": {"agents": ["gmail_agent"]}
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        assert isinstance(plan, TaskPlan)
        assert len(plan.steps) > 0
        assert plan.estimated_duration > 0
        assert plan.required_resources is not None
        
    @pytest.mark.asyncio
    async def test_create_complex_plan(self, task_planner):
        """Test creating a complex multi-step plan."""
        task_definition = {
            "goal": "Research AI trends, create report, and send to stakeholders",
            "constraints": {"deadline": "2024-12-31T23:59:59Z", "quality": "high"},
            "resources": {"agents": ["research_agent", "gmail_agent"], "budget": 500}
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        assert isinstance(plan, TaskPlan)
        assert len(plan.steps) >= 3
        assert plan.agent_assignments is not None
        assert "research_agent" in plan.agent_assignments
        assert "gmail_agent" in plan.agent_assignments
        
    @pytest.mark.asyncio
    async def test_plan_optimization(self, task_planner):
        """Test plan optimization."""
        task_definition = {
            "goal": "Process large email batch",
            "constraints": {"max_time": 600, "accuracy": 0.9},
            "resources": {"agents": ["gmail_agent", "backup_agent"]}
        }
        
        plan = await task_planner.create_plan(task_definition)
        optimized_plan = await task_planner.optimize_plan(plan)
        
        assert optimized_plan.estimated_duration <= plan.estimated_duration
        assert optimized_plan.efficiency_score >= plan.efficiency_score
        
    @pytest.mark.asyncio
    async def test_plan_execution_monitoring(self, task_planner):
        """Test plan execution monitoring."""
        task_definition = {
            "goal": "Monitor task execution",
            "constraints": {"max_time": 300},
            "resources": {"agents": ["test_agent"]}
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        # Simulate execution progress
        progress = await task_planner.monitor_execution(plan.plan_id)
        
        assert progress is not None
        assert "completion_percentage" in progress
        assert "current_step" in progress
        assert "estimated_remaining_time" in progress


class TestLearningSystem:
    """Test suite for LearningSystem class."""
    
    @pytest.fixture
    def learning_system(self):
        """Create LearningSystem instance."""
        return LearningSystem()
    
    @pytest.mark.asyncio
    async def test_learning_system_initialization(self, learning_system):
        """Test LearningSystem initialization."""
        assert learning_system.learning_algorithms is not None
        assert learning_system.knowledge_base is not None
        assert learning_system.adaptation_engine is not None
        
    @pytest.mark.asyncio
    async def test_learn_from_positive_feedback(self, learning_system):
        """Test learning from positive feedback."""
        feedback = {
            "task_id": "task_123",
            "decision_id": "decision_abc",
            "feedback_type": "positive",
            "user_rating": 5,
            "outcome_metrics": {"accuracy": 0.95, "speed": 0.8}
        }
        
        learning_result = await learning_system.learn_from_feedback(feedback)
        
        assert learning_result.get("learning_applied") is True
        assert learning_result.get("confidence_adjustment") > 0
        assert learning_result.get("pattern_updated") is True
        
    @pytest.mark.asyncio
    async def test_learn_from_negative_feedback(self, learning_system):
        """Test learning from negative feedback."""
        feedback = {
            "task_id": "task_456",
            "decision_id": "decision_def",
            "feedback_type": "negative",
            "user_rating": 2,
            "outcome_metrics": {"accuracy": 0.3, "speed": 0.9}
        }
        
        learning_result = await learning_system.learn_from_feedback(feedback)
        
        assert learning_result.get("learning_applied") is True
        assert learning_result.get("confidence_adjustment") < 0
        assert learning_result.get("pattern_updated") is True
        
    @pytest.mark.asyncio
    async def test_pattern_recognition(self, learning_system):
        """Test pattern recognition."""
        historical_data = [
            {"timestamp": "2024-01-01", "task_type": "email_triage", "outcome": "success"},
            {"timestamp": "2024-01-02", "task_type": "email_triage", "outcome": "success"},
            {"timestamp": "2024-01-03", "task_type": "email_triage", "outcome": "failure"},
            {"timestamp": "2024-01-04", "task_type": "research", "outcome": "success"},
        ]
        
        patterns = await learning_system.recognize_patterns(historical_data)
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert all("pattern_type" in p and "confidence" in p for p in patterns)
        
    @pytest.mark.asyncio
    async def test_adaptation_engine(self, learning_system):
        """Test adaptation engine."""
        performance_data = {
            "current_performance": {"accuracy": 0.75, "speed": 0.8},
            "target_performance": {"accuracy": 0.85, "speed": 0.9},
            "constraints": {"max_adaptation_time": 3600}
        }
        
        adaptations = await learning_system.adapt_algorithms(performance_data)
        
        assert isinstance(adaptations, list)
        assert len(adaptations) > 0
        assert all("adaptation_type" in a and "expected_improvement" in a for a in adaptations)
        
    @pytest.mark.asyncio
    async def test_knowledge_base_update(self, learning_system):
        """Test knowledge base update."""
        new_knowledge = {
            "domain": "email_processing",
            "insights": [
                {"type": "priority_pattern", "data": {"sender_domain": "urgent", "confidence": 0.9}},
                {"type": "time_pattern", "data": {"morning_emails": "high_priority", "confidence": 0.7}}
            ]
        }
        
        result = await learning_system.update_knowledge_base(new_knowledge)
        
        assert result is True
        
    @pytest.mark.asyncio
    async def test_get_learning_metrics(self, learning_system):
        """Test getting learning metrics."""
        metrics = await learning_system.get_learning_metrics()
        
        assert isinstance(metrics, LearningMetrics)
        assert metrics.total_learning_events >= 0
        assert metrics.accuracy_improvement >= 0
        assert metrics.adaptation_success_rate >= 0


class TestAgentCoordinator:
    """Test suite for AgentCoordinator class."""
    
    @pytest.fixture
    def mock_message_broker(self):
        """Create mock message broker."""
        broker = MagicMock()
        broker.publish = AsyncMock()
        broker.subscribe = AsyncMock()
        broker.get_agent_status = AsyncMock()
        return broker
    
    @pytest.fixture
    def agent_coordinator(self, mock_message_broker):
        """Create AgentCoordinator instance."""
        return AgentCoordinator(mock_message_broker)
    
    @pytest.mark.asyncio
    async def test_agent_coordinator_initialization(self, agent_coordinator):
        """Test AgentCoordinator initialization."""
        assert agent_coordinator.message_broker is not None
        assert agent_coordinator.active_coordinations == {}
        assert agent_coordinator.conflict_resolver is not None
        
    @pytest.mark.asyncio
    async def test_coordinate_agents_parallel(self, agent_coordinator):
        """Test parallel agent coordination."""
        coordination_request = {
            "task_id": "task_123",
            "agents": ["agent1", "agent2"],
            "coordination_type": "parallel",
            "timeout": 60
        }
        
        result = await agent_coordinator.coordinate_agents(coordination_request)
        
        assert isinstance(result, CoordinationResult)
        assert result.success is True
        assert len(result.agent_results) == 2
        assert result.coordination_type == "parallel"
        
    @pytest.mark.asyncio
    async def test_coordinate_agents_sequential(self, agent_coordinator):
        """Test sequential agent coordination."""
        coordination_request = {
            "task_id": "task_456",
            "agents": ["agent1", "agent2"],
            "coordination_type": "sequential",
            "timeout": 120
        }
        
        result = await agent_coordinator.coordinate_agents(coordination_request)
        
        assert isinstance(result, CoordinationResult)
        assert result.success is True
        assert result.coordination_type == "sequential"
        
    @pytest.mark.asyncio
    async def test_conflict_resolution(self, agent_coordinator):
        """Test conflict resolution."""
        conflicts = [
            {
                "type": "resource_conflict",
                "agents": ["agent1", "agent2"],
                "resource": "processing_slot",
                "priority": ["agent1", "agent2"]
            },
            {
                "type": "timing_conflict",
                "agents": ["agent2", "agent3"],
                "schedule": "2024-01-01T10:00:00Z"
            }
        ]
        
        resolutions = await agent_coordinator.resolve_conflicts(conflicts)
        
        assert isinstance(resolutions, list)
        assert len(resolutions) == 2
        assert all("resolution_type" in r and "affected_agents" in r for r in resolutions)
        
    @pytest.mark.asyncio
    async def test_workload_balancing(self, agent_coordinator):
        """Test workload balancing."""
        agent_workloads = {
            "agent1": AgentWorkload(agent_id="agent1", current_tasks=5, capacity=10),
            "agent2": AgentWorkload(agent_id="agent2", current_tasks=8, capacity=10),
            "agent3": AgentWorkload(agent_id="agent3", current_tasks=2, capacity=10)
        }
        
        new_task = {"task_id": "task_789", "priority": "high", "estimated_duration": 300}
        
        assignment = await agent_coordinator.balance_workload(agent_workloads, new_task)
        
        assert assignment is not None
        assert assignment.get("assigned_agent") == "agent3"  # Least loaded
        assert assignment.get("load_balance_score") > 0
        
    @pytest.mark.asyncio
    async def test_get_coordination_status(self, agent_coordinator):
        """Test getting coordination status."""
        # First start a coordination
        coordination_request = {
            "task_id": "task_status",
            "agents": ["agent1", "agent2"],
            "coordination_type": "parallel",
            "timeout": 60
        }
        
        await agent_coordinator.coordinate_agents(coordination_request)
        
        status = await agent_coordinator.get_coordination_status("task_status")
        
        assert status is not None
        assert "coordination_id" in status
        assert "progress" in status
        assert "agent_statuses" in status