"""
Mock configurations for Intelligence Engine testing.

This module provides comprehensive mocks for intelligence layer components,
following the TDD approach and ensuring complete isolation from external dependencies.
"""

import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field

import pytest


@dataclass
class MockDecision:
    """Mock decision result."""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action: str = "mock_action"
    confidence: float = 0.8
    reasoning: str = "Mock reasoning for decision"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MockTaskStep:
    """Mock task step."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = "Mock task step"
    agent_id: str = "mock_agent"
    estimated_duration: int = 60
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "agent_id": self.agent_id,
            "estimated_duration": self.estimated_duration,
            "dependencies": self.dependencies,
            "status": self.status
        }


@dataclass
class MockTaskPlan:
    """Mock task plan."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = "Mock goal"
    steps: List[MockTaskStep] = field(default_factory=list)
    estimated_duration: int = 300
    required_resources: Dict[str, Any] = field(default_factory=dict)
    agent_assignments: Dict[str, str] = field(default_factory=dict)
    efficiency_score: float = 0.8
    
    def __post_init__(self):
        """Initialize with default steps if none provided."""
        if not self.steps:
            self.steps = [
                MockTaskStep(description="Step 1", agent_id="agent1"),
                MockTaskStep(description="Step 2", agent_id="agent2")
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "estimated_duration": self.estimated_duration,
            "required_resources": self.required_resources,
            "agent_assignments": self.agent_assignments,
            "efficiency_score": self.efficiency_score
        }


@dataclass
class MockCoordinationResult:
    """Mock coordination result."""
    coordination_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    success: bool = True
    coordination_type: str = "parallel"
    agent_results: Dict[str, Any] = field(default_factory=dict)
    conflicts_resolved: Optional[List[Dict[str, Any]]] = None
    duration: float = 30.0
    
    def __post_init__(self):
        """Initialize with default agent results if none provided."""
        if not self.agent_results:
            self.agent_results = {
                "agent1": {"status": "completed", "result": "success"},
                "agent2": {"status": "completed", "result": "success"}
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "coordination_id": self.coordination_id,
            "success": self.success,
            "coordination_type": self.coordination_type,
            "agent_results": self.agent_results,
            "conflicts_resolved": self.conflicts_resolved,
            "duration": self.duration
        }


@dataclass
class MockLearningMetrics:
    """Mock learning metrics."""
    total_learning_events: int = 100
    accuracy_improvement: float = 0.15
    adaptation_success_rate: float = 0.85
    pattern_recognition_accuracy: float = 0.78
    knowledge_base_size: int = 1000
    learning_rate: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_learning_events": self.total_learning_events,
            "accuracy_improvement": self.accuracy_improvement,
            "adaptation_success_rate": self.adaptation_success_rate,
            "pattern_recognition_accuracy": self.pattern_recognition_accuracy,
            "knowledge_base_size": self.knowledge_base_size,
            "learning_rate": self.learning_rate
        }


@dataclass
class MockPerformanceMetrics:
    """Mock performance metrics."""
    total_decisions: int = 500
    average_confidence: float = 0.82
    learning_progress: float = 0.75
    coordination_success_rate: float = 0.90
    optimization_improvements: float = 0.25
    user_satisfaction: float = 0.88
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_decisions": self.total_decisions,
            "average_confidence": self.average_confidence,
            "learning_progress": self.learning_progress,
            "coordination_success_rate": self.coordination_success_rate,
            "optimization_improvements": self.optimization_improvements,
            "user_satisfaction": self.user_satisfaction
        }


@dataclass
class MockAgentWorkload:
    """Mock agent workload."""
    agent_id: str = "mock_agent"
    current_tasks: int = 3
    capacity: int = 10
    utilization: float = 0.3
    performance_score: float = 0.85
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "current_tasks": self.current_tasks,
            "capacity": self.capacity,
            "utilization": self.utilization,
            "performance_score": self.performance_score
        }


class MockDecisionMaker:
    """Mock decision maker with realistic behavior."""
    
    def __init__(self):
        self.decision_history = []
        self.confidence_threshold = 0.7
        self.decision_patterns = {
            "email_triage": {
                "urgent_keywords": ["urgent", "asap", "emergency"],
                "important_senders": ["boss", "client", "support"],
                "confidence_boost": 0.2
            },
            "research_optimization": {
                "quality_factors": ["academic", "peer-reviewed", "recent"],
                "speed_factors": ["cached", "indexed", "fast"],
                "confidence_boost": 0.15
            }
        }
    
    async def make_decision(self, context) -> MockDecision:
        """Make a mock decision based on context."""
        task_type = context.get("task_type", "generic")
        input_data = context.get("input_data", {})
        
        # Calculate confidence based on task type and input
        confidence = self._calculate_confidence(task_type, input_data)
        
        # Determine action based on task type
        action = self._determine_action(task_type, input_data)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(task_type, action, confidence)
        
        decision = MockDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            metadata={"task_type": task_type, "input_processed": True}
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _calculate_confidence(self, task_type: str, input_data: Dict[str, Any]) -> float:
        """Calculate decision confidence."""
        base_confidence = 0.7
        
        if task_type == "email_triage":
            email = input_data.get("email", {})
            subject = email.get("subject", "").lower()
            sender = email.get("sender", "").lower()
            
            # Check for urgent keywords
            urgent_keywords = self.decision_patterns["email_triage"]["urgent_keywords"]
            if any(keyword in subject for keyword in urgent_keywords):
                base_confidence += 0.2
            
            # Check for important senders
            important_senders = self.decision_patterns["email_triage"]["important_senders"]
            if any(sender_keyword in sender for sender_keyword in important_senders):
                base_confidence += 0.15
        
        elif task_type == "research_optimization":
            query = input_data.get("query", "").lower()
            preferences = input_data.get("user_preferences", {})
            
            # Check for quality indicators
            if preferences.get("focus") == "technical":
                base_confidence += 0.1
            if preferences.get("sources") == "academic":
                base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _determine_action(self, task_type: str, input_data: Dict[str, Any]) -> str:
        """Determine action based on task type and input."""
        if task_type == "email_triage":
            email = input_data.get("email", {})
            subject = email.get("subject", "").lower()
            
            if "urgent" in subject or "asap" in subject:
                return "high_priority"
            elif "meeting" in subject:
                return "schedule_priority"
            else:
                return "normal_priority"
        
        elif task_type == "research_optimization":
            query = input_data.get("query", "")
            if "trends" in query.lower():
                return "trend_analysis"
            elif "solutions" in query.lower():
                return "solution_search"
            else:
                return "general_research"
        
        return "default_action"
    
    def _generate_reasoning(self, task_type: str, action: str, confidence: float) -> str:
        """Generate reasoning for decision."""
        if task_type == "email_triage":
            if action == "high_priority":
                return f"Email contains urgent keywords and high-priority sender (confidence: {confidence:.2f})"
            elif action == "schedule_priority":
                return f"Email relates to scheduling and requires timely response (confidence: {confidence:.2f})"
            else:
                return f"Email classified as normal priority based on content analysis (confidence: {confidence:.2f})"
        
        elif task_type == "research_optimization":
            if action == "trend_analysis":
                return f"Query indicates trend analysis requirement, optimizing for recent sources (confidence: {confidence:.2f})"
            elif action == "solution_search":
                return f"Query seeks solutions, focusing on practical and implementable results (confidence: {confidence:.2f})"
            else:
                return f"General research approach with balanced coverage (confidence: {confidence:.2f})"
        
        return f"Decision made using {task_type} analysis (confidence: {confidence:.2f})"


class MockTaskPlanner:
    """Mock task planner with realistic behavior."""
    
    def __init__(self):
        self.planning_strategies = ["sequential", "parallel", "hybrid"]
        self.resource_estimator = MockResourceEstimator()
        self.dependency_resolver = MockDependencyResolver()
        self.active_plans = {}
    
    async def create_plan(self, task_definition: Dict[str, Any]) -> MockTaskPlan:
        """Create a mock task plan."""
        goal = task_definition.get("goal", "Mock goal")
        resources = task_definition.get("resources", {})
        constraints = task_definition.get("constraints", {})
        
        # Generate steps based on goal
        steps = self._generate_steps(goal, resources, constraints)
        
        # Calculate estimated duration
        estimated_duration = sum(step.estimated_duration for step in steps)
        
        # Create agent assignments
        agent_assignments = self._create_agent_assignments(steps, resources)
        
        plan = MockTaskPlan(
            goal=goal,
            steps=steps,
            estimated_duration=estimated_duration,
            required_resources=resources,
            agent_assignments=agent_assignments,
            efficiency_score=0.8
        )
        
        self.active_plans[plan.plan_id] = plan
        return plan
    
    def _generate_steps(self, goal: str, resources: Dict[str, Any], constraints: Dict[str, Any]) -> List[MockTaskStep]:
        """Generate task steps based on goal."""
        steps = []
        
        if "email" in goal.lower():
            steps.extend([
                MockTaskStep(description="Fetch emails", agent_id="gmail_agent", estimated_duration=30),
                MockTaskStep(description="Classify emails", agent_id="gmail_agent", estimated_duration=60),
                MockTaskStep(description="Prioritize emails", agent_id="gmail_agent", estimated_duration=30)
            ])
        
        if "research" in goal.lower():
            steps.extend([
                MockTaskStep(description="Query optimization", agent_id="research_agent", estimated_duration=45),
                MockTaskStep(description="Search execution", agent_id="research_agent", estimated_duration=120),
                MockTaskStep(description="Result analysis", agent_id="research_agent", estimated_duration=90)
            ])
        
        if "report" in goal.lower():
            steps.extend([
                MockTaskStep(description="Generate report", agent_id="report_agent", estimated_duration=180),
                MockTaskStep(description="Review report", agent_id="report_agent", estimated_duration=60)
            ])
        
        if "send" in goal.lower():
            steps.append(
                MockTaskStep(description="Send communication", agent_id="gmail_agent", estimated_duration=30)
            )
        
        # If no specific steps generated, create generic steps
        if not steps:
            steps = [
                MockTaskStep(description="Initialize task", agent_id="generic_agent", estimated_duration=30),
                MockTaskStep(description="Execute task", agent_id="generic_agent", estimated_duration=120),
                MockTaskStep(description="Finalize task", agent_id="generic_agent", estimated_duration=30)
            ]
        
        return steps
    
    def _create_agent_assignments(self, steps: List[MockTaskStep], resources: Dict[str, Any]) -> Dict[str, str]:
        """Create agent assignments for steps."""
        assignments = {}
        available_agents = resources.get("agents", ["generic_agent"])
        
        for step in steps:
            if step.agent_id in available_agents:
                assignments[step.step_id] = step.agent_id
            else:
                # Assign to first available agent
                assignments[step.step_id] = available_agents[0]
        
        return assignments
    
    async def optimize_plan(self, plan: MockTaskPlan) -> MockTaskPlan:
        """Optimize a task plan."""
        # Simulate optimization by reducing duration and improving efficiency
        optimized_steps = []
        for step in plan.steps:
            optimized_step = MockTaskStep(
                step_id=step.step_id,
                description=step.description,
                agent_id=step.agent_id,
                estimated_duration=int(step.estimated_duration * 0.9),  # 10% improvement
                dependencies=step.dependencies,
                status=step.status
            )
            optimized_steps.append(optimized_step)
        
        optimized_plan = MockTaskPlan(
            plan_id=plan.plan_id,
            goal=plan.goal,
            steps=optimized_steps,
            estimated_duration=int(plan.estimated_duration * 0.9),
            required_resources=plan.required_resources,
            agent_assignments=plan.agent_assignments,
            efficiency_score=min(plan.efficiency_score + 0.1, 1.0)
        )
        
        self.active_plans[optimized_plan.plan_id] = optimized_plan
        return optimized_plan
    
    async def monitor_execution(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Monitor plan execution."""
        if plan_id not in self.active_plans:
            return None
        
        plan = self.active_plans[plan_id]
        
        return {
            "plan_id": plan_id,
            "completion_percentage": 65.0,
            "current_step": 2,
            "total_steps": len(plan.steps),
            "estimated_remaining_time": 120,
            "status": "in_progress"
        }


class MockResourceEstimator:
    """Mock resource estimator."""
    
    def estimate_resources(self, task_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate required resources."""
        return {
            "cpu_time": 300,
            "memory": 512,
            "storage": 100,
            "network": 50
        }


class MockDependencyResolver:
    """Mock dependency resolver."""
    
    def resolve_dependencies(self, steps: List[MockTaskStep]) -> List[MockTaskStep]:
        """Resolve step dependencies."""
        return steps


class MockLearningSystem:
    """Mock learning system with realistic behavior."""
    
    def __init__(self):
        self.learning_algorithms = ["gradient_descent", "reinforcement", "pattern_matching"]
        self.knowledge_base = {}
        self.adaptation_engine = MockAdaptationEngine()
        self.learning_history = []
    
    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from feedback."""
        feedback_type = feedback.get("feedback_type", "neutral")
        user_rating = feedback.get("user_rating", 3)
        outcome_metrics = feedback.get("outcome_metrics", {})
        
        # Calculate learning adjustments
        if feedback_type == "positive" or user_rating >= 4:
            confidence_adjustment = 0.05
            pattern_updated = True
        elif feedback_type == "negative" or user_rating <= 2:
            confidence_adjustment = -0.05
            pattern_updated = True
        else:
            confidence_adjustment = 0.0
            pattern_updated = False
        
        learning_result = {
            "learning_applied": True,
            "confidence_adjustment": confidence_adjustment,
            "pattern_updated": pattern_updated,
            "feedback_type": feedback_type,
            "improvement_areas": ["accuracy", "speed"] if feedback_type == "negative" else []
        }
        
        self.learning_history.append(feedback)
        return learning_result
    
    async def recognize_patterns(self, historical_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recognize patterns in historical data."""
        patterns = []
        
        # Analyze success patterns
        success_rate = sum(1 for item in historical_data if item.get("outcome") == "success") / len(historical_data)
        if success_rate > 0.8:
            patterns.append({
                "pattern_type": "high_success_rate",
                "confidence": success_rate,
                "description": "Consistent high success rate observed"
            })
        
        # Analyze task type patterns
        task_types = {}
        for item in historical_data:
            task_type = item.get("task_type", "unknown")
            if task_type not in task_types:
                task_types[task_type] = {"total": 0, "successes": 0}
            task_types[task_type]["total"] += 1
            if item.get("outcome") == "success":
                task_types[task_type]["successes"] += 1
        
        for task_type, stats in task_types.items():
            if stats["total"] >= 3:  # Minimum data points
                success_rate = stats["successes"] / stats["total"]
                patterns.append({
                    "pattern_type": "task_performance",
                    "confidence": success_rate,
                    "description": f"Task type '{task_type}' has {success_rate:.1%} success rate",
                    "task_type": task_type
                })
        
        return patterns
    
    async def adapt_algorithms(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adapt algorithms based on performance."""
        current_performance = performance_data.get("current_performance", {})
        target_performance = performance_data.get("target_performance", {})
        
        adaptations = []
        
        # Check accuracy adaptation
        current_accuracy = current_performance.get("accuracy", 0.7)
        target_accuracy = target_performance.get("accuracy", 0.8)
        if current_accuracy < target_accuracy:
            adaptations.append({
                "adaptation_type": "accuracy_improvement",
                "expected_improvement": target_accuracy - current_accuracy,
                "method": "threshold_adjustment",
                "estimated_time": 300
            })
        
        # Check speed adaptation
        current_speed = current_performance.get("speed", 0.6)
        target_speed = target_performance.get("speed", 0.8)
        if current_speed < target_speed:
            adaptations.append({
                "adaptation_type": "speed_improvement",
                "expected_improvement": target_speed - current_speed,
                "method": "parallel_processing",
                "estimated_time": 600
            })
        
        return adaptations
    
    async def update_knowledge_base(self, new_knowledge: Dict[str, Any]) -> bool:
        """Update knowledge base with new insights."""
        domain = new_knowledge.get("domain", "general")
        insights = new_knowledge.get("insights", [])
        
        if domain not in self.knowledge_base:
            self.knowledge_base[domain] = {}
        
        for insight in insights:
            insight_type = insight.get("type", "unknown")
            self.knowledge_base[domain][insight_type] = insight.get("data", {})
        
        return True
    
    async def get_learning_metrics(self) -> MockLearningMetrics:
        """Get learning metrics."""
        return MockLearningMetrics(
            total_learning_events=len(self.learning_history),
            accuracy_improvement=0.15,
            adaptation_success_rate=0.85,
            pattern_recognition_accuracy=0.78,
            knowledge_base_size=len(self.knowledge_base),
            learning_rate=0.1
        )


class MockAdaptationEngine:
    """Mock adaptation engine."""
    
    def __init__(self):
        self.adaptation_strategies = ["incremental", "aggressive", "conservative"]
        self.adaptation_history = []
    
    def adapt(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform adaptation."""
        return {
            "adaptation_applied": True,
            "strategy": "incremental",
            "expected_improvement": 0.1
        }


class MockAgentCoordinator:
    """Mock agent coordinator with realistic behavior."""
    
    def __init__(self, message_broker):
        self.message_broker = message_broker
        self.active_coordinations = {}
        self.conflict_resolver = MockConflictResolver()
    
    async def coordinate_agents(self, coordination_request: Dict[str, Any]) -> MockCoordinationResult:
        """Coordinate agents."""
        task_id = coordination_request.get("task_id", "unknown")
        agents = coordination_request.get("agents", [])
        coordination_type = coordination_request.get("coordination_type", "parallel")
        
        # Simulate coordination
        agent_results = {}
        for agent in agents:
            agent_results[agent] = {
                "status": "completed",
                "result": "success",
                "duration": 30.0
            }
        
        result = MockCoordinationResult(
            coordination_type=coordination_type,
            agent_results=agent_results,
            success=True,
            duration=30.0
        )
        
        self.active_coordinations[task_id] = result
        return result
    
    async def resolve_conflicts(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve conflicts."""
        resolutions = []
        
        for conflict in conflicts:
            resolution = {
                "conflict_id": str(uuid.uuid4()),
                "resolution_type": "priority_based",
                "affected_agents": conflict.get("agents", []),
                "resolution_time": 5.0,
                "success": True
            }
            resolutions.append(resolution)
        
        return resolutions
    
    async def balance_workload(self, agent_workloads: Dict[str, MockAgentWorkload], new_task: Dict[str, Any]) -> Dict[str, Any]:
        """Balance workload across agents."""
        # Find agent with lowest utilization
        min_utilization = float('inf')
        best_agent = None
        
        for agent_id, workload in agent_workloads.items():
            if workload.utilization < min_utilization:
                min_utilization = workload.utilization
                best_agent = agent_id
        
        return {
            "assigned_agent": best_agent,
            "load_balance_score": 0.85,
            "expected_completion_time": 300,
            "utilization_after_assignment": min_utilization + 0.1
        }
    
    async def get_coordination_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get coordination status."""
        if task_id not in self.active_coordinations:
            return None
        
        coordination = self.active_coordinations[task_id]
        
        return {
            "coordination_id": coordination.coordination_id,
            "task_id": task_id,
            "progress": 100.0,
            "status": "completed",
            "agent_statuses": {
                agent: {"status": "completed", "progress": 100.0}
                for agent in coordination.agent_results.keys()
            }
        }


class MockConflictResolver:
    """Mock conflict resolver."""
    
    def __init__(self):
        self.resolution_strategies = ["priority_based", "time_based", "resource_based"]
    
    def resolve_conflict(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a single conflict."""
        return {
            "resolution_type": "priority_based",
            "success": True,
            "resolution_time": 5.0
        }


class MockIntelligenceEngine:
    """Mock intelligence engine that integrates all components."""
    
    def __init__(self, config, logger, ollama_service, message_broker):
        self.config = config
        self.logger = logger
        self.ollama_service = ollama_service
        self.message_broker = message_broker
        
        self.decision_maker = MockDecisionMaker()
        self.task_planner = MockTaskPlanner()
        self.learning_system = MockLearningSystem()
        self.agent_coordinator = MockAgentCoordinator(message_broker)
        
        self.is_initialized = False
        self.knowledge_base = {}
        self.context_awareness = MockContextAwareness()
        self.pattern_recognition = MockPatternRecognition()
    
    async def initialize(self):
        """Initialize the intelligence engine."""
        self.is_initialized = True
        self.knowledge_base = {"initialized": True}
        self.context_awareness = MockContextAwareness()
        self.pattern_recognition = MockPatternRecognition()
    
    async def shutdown(self):
        """Shutdown the intelligence engine."""
        self.is_initialized = False
    
    async def make_decision(self, context) -> MockDecision:
        """Make a decision."""
        if not self.is_initialized:
            raise ValueError("Intelligence engine not initialized")
        
        return await self.decision_maker.make_decision(context)
    
    async def create_task_plan(self, task_definition: Dict[str, Any]) -> MockTaskPlan:
        """Create a task plan."""
        return await self.task_planner.create_plan(task_definition)
    
    async def coordinate_agents(self, coordination_request: Dict[str, Any]) -> MockCoordinationResult:
        """Coordinate agents."""
        return await self.agent_coordinator.coordinate_agents(coordination_request)
    
    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from feedback."""
        return await self.learning_system.learn_from_feedback(feedback)
    
    async def optimize_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize workflow."""
        return {
            "optimizations_found": True,
            "recommended_changes": [
                {"type": "batching", "expected_improvement": 0.2},
                {"type": "caching", "expected_improvement": 0.15}
            ],
            "estimated_improvement": 0.35
        }
    
    async def get_performance_metrics(self) -> MockPerformanceMetrics:
        """Get performance metrics."""
        return MockPerformanceMetrics()
    
    async def get_agent_insights(self, agent_id: str) -> Dict[str, Any]:
        """Get agent insights."""
        return {
            "agent_id": agent_id,
            "performance": {
                "accuracy": 0.85,
                "speed": 0.78,
                "reliability": 0.92
            },
            "learning_progress": {
                "learning_rate": 0.1,
                "adaptations_applied": 5,
                "improvement_trend": "positive"
            },
            "optimization_suggestions": [
                {"type": "resource_allocation", "priority": "high"},
                {"type": "workflow_optimization", "priority": "medium"}
            ]
        }
    
    async def update_user_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        return True
    
    async def predict_optimal_actions(self, prediction_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict optimal actions."""
        return [
            {"action": "prioritize_urgent", "confidence": 0.9},
            {"action": "batch_process", "confidence": 0.7},
            {"action": "delegate_tasks", "confidence": 0.6}
        ]


class MockContextAwareness:
    """Mock context awareness component."""
    
    def __init__(self):
        self.context_history = []
        self.current_context = {}
    
    def get_context(self, context_type: str) -> Dict[str, Any]:
        """Get context information."""
        return {
            "context_type": context_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "relevant_data": {"mock": "context"}
        }


class MockPatternRecognition:
    """Mock pattern recognition component."""
    
    def __init__(self):
        self.recognized_patterns = []
        self.pattern_confidence = 0.8
    
    def recognize_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recognize patterns in data."""
        return [
            {
                "pattern_type": "temporal",
                "confidence": 0.85,
                "description": "Regular daily pattern detected"
            },
            {
                "pattern_type": "behavioral",
                "confidence": 0.78,
                "description": "User preference pattern identified"
            }
        ]


# Pytest fixtures for intelligence mocks
@pytest.fixture
def mock_decision():
    """Provide a mock decision."""
    return MockDecision()


@pytest.fixture
def mock_task_plan():
    """Provide a mock task plan."""
    return MockTaskPlan()


@pytest.fixture
def mock_coordination_result():
    """Provide a mock coordination result."""
    return MockCoordinationResult()


@pytest.fixture
def mock_learning_metrics():
    """Provide mock learning metrics."""
    return MockLearningMetrics()


@pytest.fixture
def mock_performance_metrics():
    """Provide mock performance metrics."""
    return MockPerformanceMetrics()


@pytest.fixture
def mock_decision_maker():
    """Provide a mock decision maker."""
    return MockDecisionMaker()


@pytest.fixture
def mock_task_planner():
    """Provide a mock task planner."""
    return MockTaskPlanner()


@pytest.fixture
def mock_learning_system():
    """Provide a mock learning system."""
    return MockLearningSystem()


@pytest.fixture
def mock_agent_coordinator():
    """Provide a mock agent coordinator."""
    return MockAgentCoordinator(MagicMock())


@pytest.fixture
def mock_intelligence_engine():
    """Provide a mock intelligence engine."""
    return MockIntelligenceEngine(
        config=MagicMock(),
        logger=MagicMock(),
        ollama_service=MagicMock(),
        message_broker=MagicMock()
    )


# Context managers for intelligence mocking
class MockIntelligenceContext:
    """Context manager for mocking intelligence components."""
    
    def __init__(self, intelligence_engine: MockIntelligenceEngine):
        self.intelligence_engine = intelligence_engine
        self.patches = []
    
    def __enter__(self):
        # Mock intelligence engine components
        engine_patch = patch("src.services.intelligence_engine.IntelligenceEngine")
        mock_engine = engine_patch.start()
        mock_engine.return_value = self.intelligence_engine
        self.patches.append(engine_patch)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for patch_obj in self.patches:
            patch_obj.stop()


# Decorator for intelligence mocking
def mock_intelligence_layer(engine: Optional[MockIntelligenceEngine] = None):
    """Decorator to mock intelligence layer for a test function."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            mock_engine = engine or MockIntelligenceEngine(
                config=MagicMock(),
                logger=MagicMock(),
                ollama_service=MagicMock(),
                message_broker=MagicMock()
            )
            with MockIntelligenceContext(mock_engine):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# Sample data generators for testing
def generate_sample_decisions(count: int = 5) -> List[MockDecision]:
    """Generate sample decisions for testing."""
    decisions = []
    actions = ["high_priority", "normal_priority", "delegate", "schedule", "archive"]
    
    for i in range(count):
        decision = MockDecision(
            action=actions[i % len(actions)],
            confidence=0.7 + (i * 0.05),
            reasoning=f"Mock reasoning for decision {i+1}",
            metadata={"decision_number": i+1}
        )
        decisions.append(decision)
    
    return decisions


def generate_sample_task_plans(count: int = 3) -> List[MockTaskPlan]:
    """Generate sample task plans for testing."""
    plans = []
    goals = ["Process emails", "Research topic", "Generate report"]
    
    for i in range(count):
        plan = MockTaskPlan(
            goal=goals[i % len(goals)],
            estimated_duration=300 + (i * 60),
            efficiency_score=0.7 + (i * 0.1)
        )
        plans.append(plan)
    
    return plans


def generate_sample_feedback(count: int = 10) -> List[Dict[str, Any]]:
    """Generate sample feedback for testing."""
    feedback_list = []
    feedback_types = ["positive", "negative", "neutral"]
    
    for i in range(count):
        feedback = {
            "task_id": f"task_{i+1}",
            "decision_id": f"decision_{i+1}",
            "feedback_type": feedback_types[i % len(feedback_types)],
            "user_rating": (i % 5) + 1,
            "outcome_metrics": {
                "accuracy": 0.5 + (i * 0.05),
                "speed": 0.6 + (i * 0.03)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        feedback_list.append(feedback)
    
    return feedback_list