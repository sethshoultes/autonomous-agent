"""
Advanced Intelligence Layer for the autonomous agent system.

This module provides sophisticated AI decision-making, multi-agent coordination,
learning and adaptation mechanisms, and automated workflow orchestration.
Built on top of the existing Ollama integration for enhanced AI capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator, Callable
from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

from .ollama_service import OllamaService, ProcessingRequest, ProcessingResponse
from ..agents.base import AgentMessage
from ..core.exceptions import CoreError


# Custom Exceptions
class IntelligenceError(CoreError):
    """Base exception for intelligence layer errors."""
    pass


class DecisionError(IntelligenceError):
    """Exception raised during decision making."""
    pass


class CoordinationError(IntelligenceError):
    """Exception raised during agent coordination."""
    pass


class LearningError(IntelligenceError):
    """Exception raised during learning operations."""
    pass


class TaskPlanningError(IntelligenceError):
    """Exception raised during task planning."""
    pass


# Core Data Models
@dataclass
class DecisionContext:
    """Context for decision making."""
    agent_id: str
    task_type: str
    input_data: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    historical_context: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Decision:
    """Represents a decision made by the intelligence engine."""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "context_hash": self.context_hash
        }


@dataclass
class TaskStep:
    """Represents a step in a task plan."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    agent_id: str = ""
    estimated_duration: int = 0
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    priority: int = 5
    resources: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "agent_id": self.agent_id,
            "estimated_duration": self.estimated_duration,
            "dependencies": self.dependencies,
            "status": self.status,
            "priority": self.priority,
            "resources": self.resources
        }


@dataclass
class TaskPlan:
    """Represents a complete task plan."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    steps: List[TaskStep] = field(default_factory=list)
    estimated_duration: int = 0
    required_resources: Dict[str, Any] = field(default_factory=dict)
    agent_assignments: Dict[str, str] = field(default_factory=dict)
    efficiency_score: float = 0.0
    coordination_type: str = "sequential"
    priority: str = "medium"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Additional planning attributes
    required_agents: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    sequential_groups: List[List[str]] = field(default_factory=list)
    workflow_type: str = "simple"
    stakeholder_involvement: bool = False
    approval_gates: List[Dict[str, Any]] = field(default_factory=list)
    optimization_applied: bool = False
    optimization_methods: List[str] = field(default_factory=list)
    resource_allocation: Dict[str, Any] = field(default_factory=dict)
    adaptation_applied: bool = False
    adaptation_reason: str = ""
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    version: int = 1
    template_used: Optional[str] = None
    template_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "estimated_duration": self.estimated_duration,
            "required_resources": self.required_resources,
            "agent_assignments": self.agent_assignments,
            "efficiency_score": self.efficiency_score,
            "coordination_type": self.coordination_type,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "required_agents": self.required_agents,
            "parallel_groups": self.parallel_groups,
            "sequential_groups": self.sequential_groups,
            "workflow_type": self.workflow_type,
            "stakeholder_involvement": self.stakeholder_involvement,
            "approval_gates": self.approval_gates,
            "optimization_applied": self.optimization_applied,
            "optimization_methods": self.optimization_methods,
            "resource_allocation": self.resource_allocation,
            "adaptation_applied": self.adaptation_applied,
            "adaptation_reason": self.adaptation_reason,
            "adaptation_history": self.adaptation_history,
            "version": self.version,
            "template_used": self.template_used,
            "template_parameters": self.template_parameters
        }


@dataclass
class CoordinationResult:
    """Result of agent coordination."""
    coordination_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    success: bool = False
    coordination_type: str = "parallel"
    agent_results: Dict[str, Any] = field(default_factory=dict)
    conflicts_resolved: Optional[List[Dict[str, Any]]] = None
    duration: float = 0.0
    efficiency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "coordination_id": self.coordination_id,
            "success": self.success,
            "coordination_type": self.coordination_type,
            "agent_results": self.agent_results,
            "conflicts_resolved": self.conflicts_resolved,
            "duration": self.duration,
            "efficiency": self.efficiency
        }


@dataclass
class LearningMetrics:
    """Metrics for learning system performance."""
    total_learning_events: int = 0
    accuracy_improvement: float = 0.0
    adaptation_success_rate: float = 0.0
    pattern_recognition_accuracy: float = 0.0
    knowledge_base_size: int = 0
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
class PerformanceMetrics:
    """Overall performance metrics."""
    total_decisions: int = 0
    average_confidence: float = 0.0
    learning_progress: float = 0.0
    coordination_success_rate: float = 0.0
    optimization_improvements: float = 0.0
    user_satisfaction: float = 0.0
    
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
class AgentWorkload:
    """Represents agent workload information."""
    agent_id: str
    current_tasks: int = 0
    capacity: int = 10
    utilization: float = 0.0
    performance_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "current_tasks": self.current_tasks,
            "capacity": self.capacity,
            "utilization": self.utilization,
            "performance_score": self.performance_score
        }


@dataclass
class UserPreferences:
    """User preferences for intelligent behavior."""
    email_priority_factors: Dict[str, float] = field(default_factory=dict)
    research_preferences: Dict[str, Any] = field(default_factory=dict)
    workflow_preferences: Dict[str, Any] = field(default_factory=dict)
    notification_preferences: Dict[str, Any] = field(default_factory=dict)
    learning_preferences: Dict[str, Any] = field(default_factory=dict)


# Core Intelligence Components
class PriorityScoring:
    """Calculates priority scores for various tasks."""
    
    def __init__(self):
        self.scoring_models = {}
        self.feature_extractors = {}
        self._initialize_scoring_models()
    
    def _initialize_scoring_models(self):
        """Initialize scoring models for different task types."""
        # Email priority scoring
        self.scoring_models["email_triage"] = {
            "urgency_keywords": ["urgent", "asap", "emergency", "critical", "immediate"],
            "importance_keywords": ["important", "priority", "deadline", "meeting"],
            "sender_weights": {
                "boss": 0.9,
                "client": 0.8,
                "support": 0.7,
                "team": 0.6,
                "external": 0.5
            },
            "time_weights": {
                "morning": 0.8,
                "afternoon": 0.6,
                "evening": 0.4,
                "night": 0.2
            }
        }
        
        # Research priority scoring
        self.scoring_models["research_optimization"] = {
            "quality_indicators": ["academic", "peer-reviewed", "recent", "authoritative"],
            "urgency_factors": ["deadline", "urgent", "priority"],
            "complexity_factors": ["analysis", "comprehensive", "detailed"],
            "stakeholder_weights": {
                "CEO": 0.9,
                "board": 0.8,
                "department": 0.6,
                "team": 0.4
            }
        }
    
    async def calculate_priority_score(self, task_type: str, input_data: Dict[str, Any]) -> float:
        """Calculate priority score for a task."""
        if task_type not in self.scoring_models:
            return 0.5  # Default medium priority
        
        if task_type == "email_triage":
            return await self._score_email_priority(input_data)
        elif task_type == "research_optimization":
            return await self._score_research_priority(input_data)
        elif task_type == "code_review_prioritization":
            return await self._score_code_review_priority(input_data)
        
        return 0.5
    
    async def _score_email_priority(self, input_data: Dict[str, Any]) -> float:
        """Score email priority."""
        email = input_data.get("email", {})
        subject = email.get("subject", "").lower()
        sender = email.get("sender", "").lower()
        body = email.get("body", "").lower()
        
        score = 0.5  # Base score
        model = self.scoring_models["email_triage"]
        
        # Check urgency keywords
        urgency_score = 0.0
        for keyword in model["urgency_keywords"]:
            if keyword in subject or keyword in body:
                urgency_score = min(urgency_score + 0.2, 0.4)
        
        # Check importance keywords
        importance_score = 0.0
        for keyword in model["importance_keywords"]:
            if keyword in subject or keyword in body:
                importance_score = min(importance_score + 0.1, 0.2)
        
        # Check sender importance
        sender_score = 0.0
        for sender_type, weight in model["sender_weights"].items():
            if sender_type in sender:
                sender_score = weight * 0.3
                break
        
        # Time-based scoring
        time_score = 0.1  # Default
        current_hour = datetime.now().hour
        if 6 <= current_hour < 12:  # Morning
            time_score = model["time_weights"]["morning"] * 0.1
        elif 12 <= current_hour < 18:  # Afternoon
            time_score = model["time_weights"]["afternoon"] * 0.1
        elif 18 <= current_hour < 22:  # Evening
            time_score = model["time_weights"]["evening"] * 0.1
        else:  # Night
            time_score = model["time_weights"]["night"] * 0.1
        
        final_score = min(score + urgency_score + importance_score + sender_score + time_score, 1.0)
        return final_score
    
    async def _score_research_priority(self, input_data: Dict[str, Any]) -> float:
        """Score research task priority."""
        query = input_data.get("query", "").lower()
        deadline = input_data.get("deadline")
        stakeholders = input_data.get("stakeholders", [])
        importance = input_data.get("importance", "medium").lower()
        
        score = 0.5  # Base score
        model = self.scoring_models["research_optimization"]
        
        # Check urgency factors
        urgency_score = 0.0
        for factor in model["urgency_factors"]:
            if factor in query:
                urgency_score = min(urgency_score + 0.15, 0.3)
        
        # Deadline proximity
        deadline_score = 0.0
        if deadline:
            try:
                deadline_dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                time_until_deadline = (deadline_dt - datetime.now(timezone.utc)).total_seconds()
                if time_until_deadline < 3600:  # Less than 1 hour
                    deadline_score = 0.3
                elif time_until_deadline < 86400:  # Less than 1 day
                    deadline_score = 0.2
                elif time_until_deadline < 604800:  # Less than 1 week
                    deadline_score = 0.1
            except:
                pass
        
        # Stakeholder importance
        stakeholder_score = 0.0
        for stakeholder in stakeholders:
            stakeholder_lower = str(stakeholder).lower()
            for stakeholder_type, weight in model["stakeholder_weights"].items():
                if stakeholder_type.lower() in stakeholder_lower:
                    stakeholder_score = max(stakeholder_score, weight * 0.2)
        
        # Explicit importance
        importance_score = 0.0
        if importance == "high":
            importance_score = 0.2
        elif importance == "critical":
            importance_score = 0.3
        elif importance == "low":
            importance_score = -0.1
        
        final_score = min(score + urgency_score + deadline_score + stakeholder_score + importance_score, 1.0)
        return max(final_score, 0.0)
    
    async def _score_code_review_priority(self, input_data: Dict[str, Any]) -> float:
        """Score code review priority."""
        pr = input_data.get("pull_request", {})
        title = pr.get("title", "").lower()
        labels = [label.lower() for label in pr.get("labels", [])]
        changes = pr.get("changes", 0)
        files_modified = pr.get("files_modified", [])
        
        score = 0.5  # Base score
        
        # Critical keywords
        critical_keywords = ["security", "critical", "hotfix", "urgent", "vulnerability"]
        for keyword in critical_keywords:
            if keyword in title or keyword in labels:
                score = min(score + 0.3, 1.0)
        
        # File importance
        critical_files = ["security", "auth", "payment", "config"]
        for file_path in files_modified:
            file_lower = str(file_path).lower()
            for critical in critical_files:
                if critical in file_lower:
                    score = min(score + 0.2, 1.0)
        
        # Change size impact
        if changes > 500:
            score = min(score + 0.1, 1.0)
        elif changes < 50:
            score = max(score - 0.1, 0.0)
        
        return score


class ContextAwareness:
    """Provides context awareness for intelligent decision making."""
    
    def __init__(self):
        self.context_cache = {}
        self.context_history = deque(maxlen=1000)
        self.pattern_recognizer = PatternRecognition()
    
    async def get_context(self, context_type: str, agent_id: str = None) -> Dict[str, Any]:
        """Get contextual information."""
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context_type": context_type
        }
        
        if context_type == "user_patterns":
            context.update(await self._get_user_patterns())
        elif context_type == "system_state":
            context.update(await self._get_system_state())
        elif context_type == "agent_performance":
            context.update(await self._get_agent_performance(agent_id))
        elif context_type == "workload_distribution":
            context.update(await self._get_workload_distribution())
        
        return context
    
    async def _get_user_patterns(self) -> Dict[str, Any]:
        """Get user behavioral patterns."""
        return {
            "active_hours": {"start": 9, "end": 17},
            "priority_keywords": ["urgent", "asap", "critical"],
            "preferred_communication_style": "concise",
            "typical_response_time": 1800,  # 30 minutes
            "workload_preferences": {"batch_processing": True, "real_time": False}
        }
    
    async def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state."""
        return {
            "active_agents": 5,
            "system_load": 0.65,
            "available_resources": {"cpu": 0.8, "memory": 0.7, "storage": 0.9},
            "ongoing_tasks": 12,
            "error_rate": 0.02
        }
    
    async def _get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get agent performance context."""
        return {
            "agent_id": agent_id,
            "success_rate": 0.92,
            "average_response_time": 45.0,
            "current_load": 0.6,
            "specializations": ["email_processing", "text_analysis"],
            "recent_performance_trend": "improving"
        }
    
    async def _get_workload_distribution(self) -> Dict[str, Any]:
        """Get workload distribution context."""
        return {
            "total_active_tasks": 25,
            "high_priority_tasks": 3,
            "medium_priority_tasks": 15,
            "low_priority_tasks": 7,
            "agent_utilization": {
                "gmail_agent": 0.8,
                "research_agent": 0.6,
                "code_agent": 0.4
            }
        }


class PatternRecognition:
    """Recognizes patterns in data for intelligent insights."""
    
    def __init__(self):
        self.pattern_models = {}
        self.pattern_cache = {}
        self.learning_history = deque(maxlen=10000)
    
    async def recognize_patterns(self, data: List[Dict[str, Any]], pattern_type: str = "general") -> List[Dict[str, Any]]:
        """Recognize patterns in historical data."""
        patterns = []
        
        if pattern_type == "temporal":
            patterns.extend(await self._recognize_temporal_patterns(data))
        elif pattern_type == "behavioral":
            patterns.extend(await self._recognize_behavioral_patterns(data))
        elif pattern_type == "performance":
            patterns.extend(await self._recognize_performance_patterns(data))
        else:
            patterns.extend(await self._recognize_general_patterns(data))
        
        return patterns
    
    async def _recognize_temporal_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recognize temporal patterns."""
        patterns = []
        
        # Analyze by hour of day
        hour_distribution = defaultdict(int)
        for item in data:
            if "timestamp" in item:
                try:
                    timestamp = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
                    hour_distribution[timestamp.hour] += 1
                except:
                    continue
        
        if hour_distribution:
            peak_hour = max(hour_distribution, key=hour_distribution.get)
            peak_count = hour_distribution[peak_hour]
            total_count = sum(hour_distribution.values())
            
            if peak_count > total_count * 0.3:  # More than 30% of activity in one hour
                patterns.append({
                    "pattern_type": "temporal_peak",
                    "confidence": peak_count / total_count,
                    "description": f"Peak activity at hour {peak_hour}",
                    "data": {"peak_hour": peak_hour, "concentration": peak_count / total_count}
                })
        
        # Analyze by day of week
        if len(data) > 7:
            day_distribution = defaultdict(int)
            for item in data:
                if "timestamp" in item:
                    try:
                        timestamp = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
                        day_distribution[timestamp.weekday()] += 1
                    except:
                        continue
            
            if day_distribution:
                weekday_total = sum(day_distribution[i] for i in range(5))  # Mon-Fri
                weekend_total = sum(day_distribution[i] for i in range(5, 7))  # Sat-Sun
                total = weekday_total + weekend_total
                
                if total > 0:
                    weekday_ratio = weekday_total / total
                    if weekday_ratio > 0.8:
                        patterns.append({
                            "pattern_type": "weekday_preference",
                            "confidence": weekday_ratio,
                            "description": "Strong weekday activity pattern",
                            "data": {"weekday_ratio": weekday_ratio}
                        })
        
        return patterns
    
    async def _recognize_behavioral_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recognize behavioral patterns."""
        patterns = []
        
        # Analyze task type preferences
        task_type_counts = defaultdict(int)
        success_rates = defaultdict(list)
        
        for item in data:
            task_type = item.get("task_type", "unknown")
            outcome = item.get("outcome", "unknown")
            
            task_type_counts[task_type] += 1
            success_rates[task_type].append(1 if outcome == "success" else 0)
        
        for task_type, count in task_type_counts.items():
            if count >= 5:  # Minimum sample size
                success_rate = sum(success_rates[task_type]) / len(success_rates[task_type])
                
                patterns.append({
                    "pattern_type": "task_performance",
                    "confidence": min(count / 20, 1.0),  # Confidence increases with sample size
                    "description": f"Task type '{task_type}' success pattern",
                    "data": {
                        "task_type": task_type,
                        "success_rate": success_rate,
                        "sample_size": count
                    }
                })
        
        return patterns
    
    async def _recognize_performance_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recognize performance patterns."""
        patterns = []
        
        # Analyze performance trends
        performance_data = []
        for item in data:
            if "performance_score" in item and "timestamp" in item:
                try:
                    timestamp = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
                    performance_data.append((timestamp, item["performance_score"]))
                except:
                    continue
        
        if len(performance_data) >= 10:
            performance_data.sort(key=lambda x: x[0])  # Sort by timestamp
            scores = [score for _, score in performance_data]
            
            # Calculate trend
            x = np.arange(len(scores))
            z = np.polyfit(x, scores, 1)
            trend_slope = z[0]
            
            if abs(trend_slope) > 0.01:  # Significant trend
                trend_direction = "improving" if trend_slope > 0 else "declining"
                patterns.append({
                    "pattern_type": "performance_trend",
                    "confidence": min(abs(trend_slope) * 10, 1.0),
                    "description": f"Performance {trend_direction} over time",
                    "data": {
                        "trend_slope": trend_slope,
                        "trend_direction": trend_direction,
                        "sample_size": len(scores)
                    }
                })
        
        return patterns
    
    async def _recognize_general_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recognize general patterns."""
        patterns = []
        
        # Overall success rate pattern
        outcomes = [item.get("outcome") for item in data if "outcome" in item]
        if outcomes:
            success_rate = outcomes.count("success") / len(outcomes)
            
            if success_rate > 0.8:
                patterns.append({
                    "pattern_type": "high_success_rate",
                    "confidence": success_rate,
                    "description": "Consistently high success rate",
                    "data": {"success_rate": success_rate, "sample_size": len(outcomes)}
                })
            elif success_rate < 0.6:
                patterns.append({
                    "pattern_type": "low_success_rate",
                    "confidence": 1 - success_rate,
                    "description": "Concerning low success rate",
                    "data": {"success_rate": success_rate, "sample_size": len(outcomes)}
                })
        
        return patterns


# Main Intelligence Components
class DecisionMaker:
    """Advanced decision making with context awareness and learning."""
    
    def __init__(self, ollama_service: OllamaService, context_awareness: ContextAwareness = None):
        self.ollama_service = ollama_service
        self.context_awareness = context_awareness or ContextAwareness()
        self.priority_scorer = PriorityScoring()
        
        self.decision_history = []
        self.confidence_threshold = 0.7
        self.learning_adjustments = {}
        self.decision_cache = {}
        self.adaptation_history = []
    
    async def make_decision(self, context: DecisionContext) -> Decision:
        """Make an intelligent decision based on context."""
        if context is None:
            raise DecisionError("Decision context cannot be None")
        
        try:
            # Check cache for similar decisions
            context_hash = self._hash_context(context)
            if context_hash in self.decision_cache:
                cached_decision = self.decision_cache[context_hash]
                cached_decision.metadata["from_cache"] = True
                return cached_decision
            
            # Get additional context
            additional_context = await self.context_awareness.get_context(
                "system_state", context.agent_id
            )
            
            # Calculate priority score
            priority_score = await self.priority_scorer.calculate_priority_score(
                context.task_type, context.input_data
            )
            
            # Make decision based on task type
            decision = await self._make_task_specific_decision(context, priority_score, additional_context)
            
            # Apply learning adjustments
            decision = await self._apply_learning_adjustments(decision, context)
            
            # Cache decision
            self.decision_cache[context_hash] = decision
            decision.context_hash = context_hash
            
            # Store in history
            self.decision_history.append(decision)
            
            return decision
            
        except Exception as e:
            raise DecisionError(f"Failed to make decision: {str(e)}")
    
    async def _make_task_specific_decision(self, context: DecisionContext, priority_score: float, additional_context: Dict[str, Any]) -> Decision:
        """Make decision specific to task type."""
        if context.task_type == "email_triage":
            return await self._make_email_triage_decision(context, priority_score, additional_context)
        elif context.task_type == "research_optimization":
            return await self._make_research_decision(context, priority_score, additional_context)
        elif context.task_type == "code_review_prioritization":
            return await self._make_code_review_decision(context, priority_score, additional_context)
        elif context.task_type == "task_scheduling":
            return await self._make_task_scheduling_decision(context, priority_score, additional_context)
        else:
            return await self._make_general_decision(context, priority_score, additional_context)
    
    async def _make_email_triage_decision(self, context: DecisionContext, priority_score: float, additional_context: Dict[str, Any]) -> Decision:
        """Make email triage decision."""
        email = context.input_data.get("email", {})
        if not email:
            raise DecisionError("Email data missing for email triage decision")
        
        subject = email.get("subject", "").lower()
        sender = email.get("sender", "").lower()
        body = email.get("body", "").lower()
        
        # Determine action based on priority score
        if priority_score > 0.8:
            action = "high_priority"
            confidence = 0.9
        elif priority_score > 0.6:
            if "meeting" in subject:
                action = "schedule_priority"
                confidence = 0.8
            else:
                action = "medium_priority"
                confidence = 0.7
        else:
            action = "normal_priority"
            confidence = 0.6
        
        # Generate reasoning
        reasoning_factors = []
        if priority_score > 0.8:
            reasoning_factors.append("high priority score based on urgency indicators")
        if "urgent" in subject or "urgent" in body:
            reasoning_factors.append("urgent keywords detected")
        if any(sender_type in sender for sender_type in ["boss", "client", "support"]):
            reasoning_factors.append("important sender identified")
        if "meeting" in subject:
            reasoning_factors.append("meeting-related content requires scheduling attention")
        
        reasoning = f"Email classified as {action} priority. " + "; ".join(reasoning_factors)
        
        # Calculate confidence factors
        confidence_factors = {
            "priority_score": priority_score,
            "urgency_keywords": 1.0 if "urgent" in subject + body else 0.0,
            "sender_importance": 0.8 if any(s in sender for s in ["boss", "client"]) else 0.3,
            "content_clarity": 0.8,  # Assume good content clarity
            "historical_accuracy": 0.85  # Based on past performance
        }
        
        return Decision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "priority_score": priority_score,
                "confidence_factors": confidence_factors,
                "email_analysis": {
                    "subject_keywords": self._extract_keywords(subject),
                    "sender_classification": self._classify_sender(sender),
                    "urgency_level": "high" if priority_score > 0.8 else "medium" if priority_score > 0.5 else "low"
                },
                "context_used": True
            }
        )
    
    async def _make_research_decision(self, context: DecisionContext, priority_score: float, additional_context: Dict[str, Any]) -> Decision:
        """Make research optimization decision."""
        query = context.input_data.get("query", "")
        user_preferences = context.input_data.get("user_preferences", {})
        resource_constraints = context.input_data.get("resource_constraints", {})
        
        # Determine research strategy
        if "trends" in query.lower():
            action = "trend_analysis"
            strategy = "focus_on_recent_sources"
        elif "solutions" in query.lower():
            action = "solution_search"
            strategy = "prioritize_practical_implementations"
        elif "comparison" in query.lower():
            action = "comparative_analysis"
            strategy = "structured_comparison_approach"
        else:
            action = "general_research"
            strategy = "comprehensive_coverage"
        
        # Calculate confidence based on query clarity and resources
        confidence = 0.7
        if len(query) > 20:  # Detailed query
            confidence += 0.1
        if user_preferences.get("focus"):
            confidence += 0.1
        if resource_constraints.get("time_limit", 0) > 1800:  # Sufficient time
            confidence += 0.1
        
        reasoning = f"Research strategy '{action}' selected based on query analysis. Strategy: {strategy}"
        
        return Decision(
            action=action,
            confidence=min(confidence, 1.0),
            reasoning=reasoning,
            metadata={
                "priority_score": priority_score,
                "research_strategy": strategy,
                "source_prioritization": self._determine_source_priority(user_preferences),
                "estimated_depth": "detailed" if resource_constraints.get("time_limit", 0) > 1800 else "summary",
                "quality_threshold": user_preferences.get("quality", "medium")
            }
        )
    
    async def _make_code_review_decision(self, context: DecisionContext, priority_score: float, additional_context: Dict[str, Any]) -> Decision:
        """Make code review prioritization decision."""
        pr = context.input_data.get("pull_request", {})
        title = pr.get("title", "").lower()
        labels = [label.lower() for label in pr.get("labels", [])]
        
        # Determine review priority
        if priority_score > 0.9 or "security" in title or "security" in labels:
            action = "immediate_review"
            confidence = 0.95
            risk_assessment = "high"
        elif priority_score > 0.7 or "critical" in title:
            action = "priority_review"
            confidence = 0.85
            risk_assessment = "medium"
        else:
            action = "standard_review"
            confidence = 0.7
            risk_assessment = "low"
        
        reasoning = f"Code review prioritized as {action} based on "
        reasoning_factors = []
        
        if "security" in title or "security" in labels:
            reasoning_factors.append("security implications")
        if "critical" in title:
            reasoning_factors.append("critical system impact")
        if priority_score > 0.8:
            reasoning_factors.append("high priority score")
        
        reasoning += ", ".join(reasoning_factors) if reasoning_factors else "standard criteria"
        
        return Decision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "priority_score": priority_score,
                "risk_assessment": risk_assessment,
                "estimated_review_time": self._estimate_review_time(pr),
                "reviewer_recommendations": self._recommend_reviewers(pr),
                "automated_checks_status": "pending"
            }
        )
    
    async def _make_task_scheduling_decision(self, context: DecisionContext, priority_score: float, additional_context: Dict[str, Any]) -> Decision:
        """Make task scheduling decision."""
        task = context.input_data.get("task", {})
        schedule = context.input_data.get("schedule", {})
        
        current_load = schedule.get("current_load", 0.5)
        available_slots = schedule.get("available_slots", [])
        
        # Determine scheduling action
        if priority_score > 0.8 and current_load < 0.8:
            action = "schedule_now"
            confidence = 0.9
            recommended_slot = "immediate"
        elif priority_score > 0.6 and available_slots:
            action = "schedule_next_slot"
            confidence = 0.8
            recommended_slot = available_slots[0]
        else:
            action = "queue_task"
            confidence = 0.7
            recommended_slot = "when_capacity_available"
        
        reasoning = f"Task scheduled with action '{action}' based on priority score {priority_score:.2f} and current system load {current_load:.2f}"
        
        return Decision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "priority_score": priority_score,
                "recommended_slot": recommended_slot,
                "load_consideration": current_load,
                "estimated_completion": self._estimate_completion_time(task),
                "resource_requirements": task.get("estimated_duration", 0)
            }
        )
    
    async def _make_general_decision(self, context: DecisionContext, priority_score: float, additional_context: Dict[str, Any]) -> Decision:
        """Make general decision for unknown task types."""
        # Use AI model for general decision making
        prompt = f"""
        Make a decision for the following context:
        Task Type: {context.task_type}
        Input Data: {json.dumps(context.input_data, indent=2)}
        Priority Score: {priority_score}
        Constraints: {json.dumps(context.constraints, indent=2)}
        
        Please provide:
        1. Recommended action
        2. Confidence level (0-1)
        3. Reasoning for the decision
        
        Respond in JSON format.
        """
        
        try:
            response = await self.ollama_service.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3}  # Lower temperature for more consistent decisions
            )
            
            if response.success:
                # Parse AI response
                ai_decision = json.loads(response.content)
                
                return Decision(
                    action=ai_decision.get("action", "default_action"),
                    confidence=ai_decision.get("confidence", 0.5),
                    reasoning=ai_decision.get("reasoning", "AI-generated decision"),
                    metadata={
                        "priority_score": priority_score,
                        "ai_generated": True,
                        "model_used": "ollama",
                        "context_used": True
                    }
                )
            else:
                # Fallback decision
                return Decision(
                    action="default_action",
                    confidence=0.5,
                    reasoning="Fallback decision due to AI model unavailability",
                    metadata={
                        "priority_score": priority_score,
                        "fallback_used": True
                    }
                )
                
        except Exception as e:
            # Fallback decision
            return Decision(
                action="default_action",
                confidence=0.4,
                reasoning=f"Fallback decision due to error: {str(e)}",
                metadata={
                    "priority_score": priority_score,
                    "error_fallback": True,
                    "error": str(e)
                }
            )
    
    def _hash_context(self, context: DecisionContext) -> str:
        """Create hash of context for caching."""
        import hashlib
        context_str = f"{context.agent_id}_{context.task_type}_{json.dumps(context.input_data, sort_keys=True)}"
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if len(word) > 3]
        return keywords[:10]  # Return top 10 keywords
    
    def _classify_sender(self, sender: str) -> str:
        """Classify sender type."""
        sender_lower = sender.lower()
        if any(term in sender_lower for term in ["boss", "manager", "ceo", "director"]):
            return "leadership"
        elif any(term in sender_lower for term in ["client", "customer", "external"]):
            return "external"
        elif any(term in sender_lower for term in ["support", "help", "service"]):
            return "support"
        else:
            return "internal"
    
    def _determine_source_priority(self, preferences: Dict[str, Any]) -> Dict[str, float]:
        """Determine source priority based on preferences."""
        base_priority = {
            "academic": 0.9,
            "industry": 0.7,
            "news": 0.5,
            "social": 0.3
        }
        
        if preferences.get("sources") == "academic":
            base_priority["academic"] = 1.0
        elif preferences.get("sources") == "industry":
            base_priority["industry"] = 0.9
        
        return base_priority
    
    def _estimate_review_time(self, pr: Dict[str, Any]) -> int:
        """Estimate code review time in minutes."""
        changes = pr.get("changes", 0)
        files = len(pr.get("files_modified", []))
        
        base_time = 15  # 15 minutes base
        change_time = min(changes * 0.1, 60)  # Max 60 minutes for changes
        file_time = files * 5  # 5 minutes per file
        
        return int(base_time + change_time + file_time)
    
    def _recommend_reviewers(self, pr: Dict[str, Any]) -> List[str]:
        """Recommend reviewers based on PR content."""
        files = pr.get("files_modified", [])
        labels = pr.get("labels", [])
        
        reviewers = []
        
        # Security files need security team
        if any("security" in file.lower() for file in files):
            reviewers.append("security_team")
        
        # Database files need DBA
        if any("db" in file.lower() or "database" in file.lower() for file in files):
            reviewers.append("database_team")
        
        # Always include team lead
        reviewers.append("team_lead")
        
        return reviewers[:3]  # Max 3 reviewers
    
    def _estimate_completion_time(self, task: Dict[str, Any]) -> str:
        """Estimate task completion time."""
        duration = task.get("estimated_duration", 300)  # Default 5 minutes
        
        completion_time = datetime.now(timezone.utc) + timedelta(seconds=duration)
        return completion_time.isoformat()
    
    async def calculate_priority_score(self, task_type: str, input_data: Dict[str, Any]) -> float:
        """Calculate priority score for external use."""
        return await self.priority_scorer.calculate_priority_score(task_type, input_data)
    
    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Learn from decision feedback."""
        decision_id = feedback.get("decision_id")
        feedback_type = feedback.get("feedback_type")
        user_rating = feedback.get("user_rating", 3)
        outcome_metrics = feedback.get("outcome_metrics", {})
        
        # Find the decision in history
        decision = next((d for d in self.decision_history if d.decision_id == decision_id), None)
        if not decision:
            return
        
        # Calculate learning adjustment
        if feedback_type == "positive" or user_rating >= 4:
            adjustment = 0.05  # Small positive adjustment
            self.confidence_threshold = min(self.confidence_threshold + 0.01, 0.9)
        elif feedback_type == "negative" or user_rating <= 2:
            adjustment = -0.05  # Small negative adjustment
            self.confidence_threshold = max(self.confidence_threshold - 0.01, 0.5)
        else:
            adjustment = 0.0
        
        # Store learning adjustment
        context_hash = decision.context_hash
        if context_hash not in self.learning_adjustments:
            self.learning_adjustments[context_hash] = 0.0
        
        self.learning_adjustments[context_hash] += adjustment
        
        # Record adaptation
        self.adaptation_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision_id": decision_id,
            "feedback_type": feedback_type,
            "adjustment": adjustment,
            "new_threshold": self.confidence_threshold
        })
    
    async def _apply_learning_adjustments(self, decision: Decision, context: DecisionContext) -> Decision:
        """Apply learning adjustments to decision."""
        context_hash = self._hash_context(context)
        
        if context_hash in self.learning_adjustments:
            adjustment = self.learning_adjustments[context_hash]
            decision.confidence = max(0.0, min(1.0, decision.confidence + adjustment))
            decision.metadata["learning_applied"] = True
            decision.metadata["confidence_adjustment"] = adjustment
        else:
            decision.metadata["learning_applied"] = False
        
        return decision


class TaskPlanner:
    """Task planning component with multi-step planning capabilities."""
    
    def __init__(self):
        self.active_plans = {}
        self.planning_strategies = ["sequential", "parallel", "hybrid"]
        self.plan_templates = {}
        
    async def create_plan(self, task_definition: Dict[str, Any]) -> TaskPlan:
        """Create a comprehensive task plan."""
        # Implementation moved to separate file for better organization
        from .task_planner import TaskPlanner as DetailedTaskPlanner
        detailed_planner = DetailedTaskPlanner()
        return await detailed_planner.create_plan(task_definition)


class LearningSystem:
    """Learning system component with adaptation capabilities."""
    
    def __init__(self):
        self.learning_history = deque(maxlen=1000)
        self.adaptation_strategies = {}
        
    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from feedback and adapt behavior."""
        # Implementation moved to separate file for better organization
        from .learning_system import LearningSystem as DetailedLearningSystem
        detailed_learning = DetailedLearningSystem()
        return await detailed_learning.learn_from_feedback(feedback)
    
    async def recognize_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recognize patterns in data."""
        patterns = await self.pattern_recognition.recognize_patterns(data)
        return patterns
    
    async def adapt_algorithms(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adapt algorithms based on performance."""
        from .learning_system import LearningSystem as DetailedLearningSystem
        detailed_learning = DetailedLearningSystem()
        return await detailed_learning.adapt_algorithms(performance_data)
    
    async def update_knowledge_base(self, new_knowledge: Dict[str, Any]) -> bool:
        """Update knowledge base."""
        from .learning_system import LearningSystem as DetailedLearningSystem
        detailed_learning = DetailedLearningSystem()
        return await detailed_learning.update_knowledge_base(new_knowledge)
    
    async def get_learning_metrics(self) -> LearningMetrics:
        """Get learning metrics."""
        from .learning_system import LearningSystem as DetailedLearningSystem
        detailed_learning = DetailedLearningSystem()
        return await detailed_learning.get_learning_metrics()


class AgentCoordinator:
    """Agent coordination component."""
    
    def __init__(self, message_broker):
        self.message_broker = message_broker
        self.active_coordinations = {}
        self.conflict_resolver = None
        
    async def coordinate_agents(self, coordination_request: Dict[str, Any]) -> CoordinationResult:
        """Coordinate multiple agents."""
        # Implementation moved to separate file for better organization
        from .agent_coordinator import AgentCoordinator as DetailedCoordinator
        detailed_coordinator = DetailedCoordinator(self.message_broker)
        return await detailed_coordinator.coordinate_agents(coordination_request)
    
    async def resolve_conflicts(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve conflicts between agents."""
        from .agent_coordinator import AgentCoordinator as DetailedCoordinator
        detailed_coordinator = DetailedCoordinator(self.message_broker)
        return await detailed_coordinator.resolve_conflicts(conflicts)
    
    async def balance_workload(self, agent_workloads: Dict[str, AgentWorkload], new_task: Dict[str, Any]) -> Dict[str, Any]:
        """Balance workload across agents."""
        from .agent_coordinator import AgentCoordinator as DetailedCoordinator
        detailed_coordinator = DetailedCoordinator(self.message_broker)
        return await detailed_coordinator.balance_workload(agent_workloads, new_task)
    
    async def get_coordination_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get coordination status."""
        from .agent_coordinator import AgentCoordinator as DetailedCoordinator
        detailed_coordinator = DetailedCoordinator(self.message_broker)
        return await detailed_coordinator.get_coordination_status(task_id)


# Additional Intelligence Components
class MultiStepPlanning:
    """Advanced multi-step planning capabilities."""
    
    def __init__(self):
        self.planning_algorithms = ["breadth_first", "depth_first", "best_first", "a_star"]
        self.plan_cache = {}
    
    async def create_multi_step_plan(self, goal: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed multi-step plan."""
        steps = await self._decompose_goal(goal, constraints)
        dependencies = await self._analyze_dependencies(steps)
        optimized_sequence = await self._optimize_sequence(steps, dependencies)
        
        return {
            "goal": goal,
            "steps": optimized_sequence,
            "estimated_duration": sum(step.get("duration", 0) for step in optimized_sequence),
            "complexity_score": self._calculate_complexity(optimized_sequence),
            "success_probability": self._estimate_success_probability(optimized_sequence)
        }
    
    async def _decompose_goal(self, goal: str, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose a high-level goal into steps."""
        # Analyze goal and create steps
        steps = []
        goal_lower = goal.lower()
        
        if "email" in goal_lower:
            steps.extend([
                {"action": "fetch_emails", "duration": 30, "priority": 1},
                {"action": "classify_emails", "duration": 60, "priority": 2},
                {"action": "process_emails", "duration": 120, "priority": 3}
            ])
        
        if "research" in goal_lower:
            steps.extend([
                {"action": "formulate_query", "duration": 45, "priority": 1},
                {"action": "search_sources", "duration": 180, "priority": 2},
                {"action": "analyze_results", "duration": 120, "priority": 3}
            ])
        
        if "report" in goal_lower:
            steps.extend([
                {"action": "gather_data", "duration": 90, "priority": 1},
                {"action": "create_structure", "duration": 60, "priority": 2},
                {"action": "write_report", "duration": 240, "priority": 3}
            ])
        
        return steps
    
    async def _analyze_dependencies(self, steps: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze dependencies between steps."""
        dependencies = {}
        
        for i, step in enumerate(steps):
            step_id = f"step_{i}"
            dependencies[step_id] = []
            
            # Simple dependency logic based on step priorities
            for j, prev_step in enumerate(steps[:i]):
                if prev_step.get("priority", 0) < step.get("priority", 0):
                    dependencies[step_id].append(f"step_{j}")
        
        return dependencies
    
    async def _optimize_sequence(self, steps: List[Dict[str, Any]], dependencies: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Optimize the sequence of steps."""
        # Simple topological sort for dependency ordering
        visited = set()
        result = []
        
        def visit(step_index: int):
            step_id = f"step_{step_index}"
            if step_index in visited:
                return
            
            visited.add(step_index)
            
            # Visit dependencies first
            for dep_id in dependencies.get(step_id, []):
                dep_index = int(dep_id.split("_")[1])
                visit(dep_index)
            
            if step_index < len(steps):
                result.append(steps[step_index])
        
        for i in range(len(steps)):
            visit(i)
        
        return result
    
    def _calculate_complexity(self, steps: List[Dict[str, Any]]) -> float:
        """Calculate plan complexity score."""
        return len(steps) * 0.1 + sum(step.get("duration", 0) for step in steps) / 3600
    
    def _estimate_success_probability(self, steps: List[Dict[str, Any]]) -> float:
        """Estimate probability of plan success."""
        base_probability = 0.9
        complexity_penalty = len(steps) * 0.02
        return max(0.5, base_probability - complexity_penalty)


class WorkflowOrchestration:
    """Orchestrates complex workflows across multiple agents and systems."""
    
    def __init__(self):
        self.active_workflows = {}
        self.workflow_templates = {}
        self.orchestration_patterns = ["sequential", "parallel", "event_driven", "condition_based"]
    
    async def orchestrate_workflow(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate a complex workflow."""
        workflow_id = str(uuid.uuid4())
        
        # Parse workflow definition
        workflow_type = workflow_definition.get("type", "sequential")
        steps = workflow_definition.get("steps", [])
        conditions = workflow_definition.get("conditions", {})
        
        # Create orchestration plan
        orchestration_plan = await self._create_orchestration_plan(workflow_type, steps, conditions)
        
        # Execute orchestration
        execution_result = await self._execute_orchestration(orchestration_plan)
        
        self.active_workflows[workflow_id] = {
            "definition": workflow_definition,
            "plan": orchestration_plan,
            "result": execution_result,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed" if execution_result.get("success") else "failed"
        }
        
        return {
            "workflow_id": workflow_id,
            "success": execution_result.get("success", False),
            "orchestration_plan": orchestration_plan,
            "execution_result": execution_result
        }
    
    async def _create_orchestration_plan(self, workflow_type: str, steps: List[Dict[str, Any]], conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Create an orchestration plan."""
        return {
            "workflow_type": workflow_type,
            "execution_strategy": "optimized",
            "step_count": len(steps),
            "estimated_duration": sum(step.get("duration", 60) for step in steps),
            "coordination_points": self._identify_coordination_points(steps),
            "fallback_strategies": self._generate_fallback_strategies(steps)
        }
    
    async def _execute_orchestration(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the orchestration plan."""
        # Simulate orchestration execution
        return {
            "success": True,
            "steps_completed": plan.get("step_count", 0),
            "actual_duration": plan.get("estimated_duration", 0) * 0.9,  # 10% efficiency gain
            "coordination_points_hit": len(plan.get("coordination_points", [])),
            "fallbacks_used": 0
        }
    
    def _identify_coordination_points(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify points where coordination is needed."""
        coordination_points = []
        
        for i, step in enumerate(steps):
            if step.get("requires_coordination", False):
                coordination_points.append({
                    "step_index": i,
                    "coordination_type": "sync",
                    "participants": step.get("agents", [])
                })
        
        return coordination_points
    
    def _generate_fallback_strategies(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate fallback strategies for workflow steps."""
        fallbacks = []
        
        for i, step in enumerate(steps):
            if step.get("critical", False):
                fallbacks.append({
                    "step_index": i,
                    "fallback_type": "retry",
                    "max_retries": 3,
                    "alternative_agent": "backup_agent"
                })
        
        return fallbacks


class IntelligentPrioritization:
    """Intelligent prioritization system."""
    
    def __init__(self):
        self.prioritization_models = {}
        self.priority_history = deque(maxlen=1000)
        
    async def prioritize_tasks(self, tasks: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Intelligently prioritize a list of tasks."""
        scored_tasks = []
        
        for task in tasks:
            priority_score = await self._calculate_priority_score(task, context)
            scored_task = task.copy()
            scored_task["priority_score"] = priority_score
            scored_task["priority_reasoning"] = await self._generate_priority_reasoning(task, priority_score)
            scored_tasks.append(scored_task)
        
        # Sort by priority score (descending)
        prioritized_tasks = sorted(scored_tasks, key=lambda x: x["priority_score"], reverse=True)
        
        return prioritized_tasks
    
    async def _calculate_priority_score(self, task: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate priority score for a task."""
        base_score = 0.5
        
        # Urgency factor
        urgency = task.get("urgency", "medium")
        urgency_scores = {"low": 0.3, "medium": 0.5, "high": 0.8, "critical": 1.0}
        base_score += urgency_scores.get(urgency, 0.5) * 0.3
        
        # Importance factor
        importance = task.get("importance", "medium")
        importance_scores = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
        base_score += importance_scores.get(importance, 0.5) * 0.3
        
        # Deadline factor
        deadline = task.get("deadline")
        if deadline:
            try:
                deadline_dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                time_until_deadline = (deadline_dt - datetime.now(timezone.utc)).total_seconds()
                
                if time_until_deadline < 3600:  # Less than 1 hour
                    base_score += 0.3
                elif time_until_deadline < 86400:  # Less than 1 day
                    base_score += 0.2
                elif time_until_deadline < 604800:  # Less than 1 week
                    base_score += 0.1
            except:
                pass
        
        # Context factors
        user_preferences = context.get("user_preferences", {})
        if task.get("category") in user_preferences.get("priority_categories", []):
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    async def _generate_priority_reasoning(self, task: Dict[str, Any], score: float) -> str:
        """Generate reasoning for priority assignment."""
        factors = []
        
        urgency = task.get("urgency", "medium")
        if urgency in ["high", "critical"]:
            factors.append(f"high urgency ({urgency})")
        
        importance = task.get("importance", "medium")
        if importance in ["high", "critical"]:
            factors.append(f"high importance ({importance})")
        
        if task.get("deadline"):
            factors.append("has deadline")
        
        if not factors:
            factors.append("standard prioritization criteria")
        
        return f"Priority score {score:.2f} based on: {', '.join(factors)}"


class CrossAgentLearning:
    """Enables learning across different agents."""
    
    def __init__(self):
        self.shared_knowledge = {}
        self.learning_insights = deque(maxlen=500)
        self.cross_agent_patterns = {}
    
    async def share_learning_insight(self, source_agent: str, insight: Dict[str, Any]) -> bool:
        """Share a learning insight from one agent to others."""
        insight_id = str(uuid.uuid4())
        
        learning_insight = {
            "insight_id": insight_id,
            "source_agent": source_agent,
            "insight_type": insight.get("type", "general"),
            "knowledge": insight.get("knowledge", {}),
            "confidence": insight.get("confidence", 0.5),
            "applicable_domains": insight.get("domains", ["general"]),
            "shared_at": datetime.now(timezone.utc).isoformat()
        }
        
        self.learning_insights.append(learning_insight)
        
        # Store in shared knowledge base
        insight_type = learning_insight["insight_type"]
        if insight_type not in self.shared_knowledge:
            self.shared_knowledge[insight_type] = []
        
        self.shared_knowledge[insight_type].append(learning_insight)
        
        return True
    
    async def get_relevant_insights(self, requesting_agent: str, domain: str) -> List[Dict[str, Any]]:
        """Get relevant insights for an agent in a specific domain."""
        relevant_insights = []
        
        for insight_type, insights in self.shared_knowledge.items():
            for insight in insights:
                if (domain in insight["applicable_domains"] or 
                    "general" in insight["applicable_domains"]):
                    if insight["confidence"] > 0.6:  # Only high-confidence insights
                        relevant_insights.append(insight)
        
        # Sort by confidence and recency
        relevant_insights.sort(
            key=lambda x: (x["confidence"], x["shared_at"]), 
            reverse=True
        )
        
        return relevant_insights[:10]  # Return top 10 insights
    
    async def identify_cross_agent_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns that emerge across multiple agents."""
        patterns = []
        
        # Group insights by type
        insight_groups = defaultdict(list)
        for insight in self.learning_insights:
            insight_groups[insight["insight_type"]].append(insight)
        
        # Look for patterns within each type
        for insight_type, type_insights in insight_groups.items():
            if len(type_insights) >= 3:  # Need at least 3 insights to identify pattern
                # Extract common knowledge patterns
                common_knowledge = self._extract_common_knowledge(type_insights)
                
                if common_knowledge:
                    pattern = {
                        "pattern_type": f"cross_agent_{insight_type}",
                        "confidence": len(type_insights) / 10,  # Simple confidence calculation
                        "description": f"Common {insight_type} pattern across {len(type_insights)} agents",
                        "knowledge": common_knowledge,
                        "contributing_agents": list(set(insight["source_agent"] for insight in type_insights)),
                        "discovered_at": datetime.now(timezone.utc).isoformat()
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _extract_common_knowledge(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common knowledge from multiple insights."""
        # Simple implementation: find knowledge that appears in multiple insights
        knowledge_counts = defaultdict(int)
        
        for insight in insights:
            knowledge = insight.get("knowledge", {})
            for key, value in knowledge.items():
                knowledge_counts[f"{key}:{value}"] += 1
        
        # Return knowledge that appears in at least 50% of insights
        threshold = len(insights) * 0.5
        common_knowledge = {}
        
        for knowledge_item, count in knowledge_counts.items():
            if count >= threshold:
                key, value = knowledge_item.split(":", 1)
                common_knowledge[key] = value
        
        return common_knowledge


class AutomatedOptimization:
    """Automated system optimization based on performance data."""
    
    def __init__(self):
        self.optimization_history = deque(maxlen=200)
        self.optimization_strategies = {}
        self.performance_baselines = {}
    
    async def optimize_system_performance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically optimize system performance."""
        current_metrics = performance_data.get("current_metrics", {})
        
        # Identify optimization opportunities
        opportunities = await self._identify_optimization_opportunities(current_metrics)
        
        # Select and apply optimizations
        applied_optimizations = []
        for opportunity in opportunities[:3]:  # Apply top 3 optimizations
            optimization_result = await self._apply_optimization(opportunity)
            applied_optimizations.append(optimization_result)
        
        # Calculate expected improvement
        expected_improvement = sum(
            opt.get("expected_improvement", 0) for opt in applied_optimizations
        )
        
        return {
            "optimizations_applied": applied_optimizations,
            "expected_improvement": expected_improvement,
            "optimization_timestamp": datetime.now(timezone.utc).isoformat(),
            "performance_before": current_metrics,
            "next_optimization_check": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        }
    
    async def _identify_optimization_opportunities(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities."""
        opportunities = []
        
        # Check response time
        response_time = metrics.get("average_response_time", 0)
        if response_time > 5.0:  # More than 5 seconds
            opportunities.append({
                "type": "response_time_optimization",
                "current_value": response_time,
                "target_value": 3.0,
                "priority": "high",
                "strategy": "caching_and_parallelization"
            })
        
        # Check resource utilization
        cpu_utilization = metrics.get("cpu_utilization", 0)
        if cpu_utilization > 0.8:  # More than 80%
            opportunities.append({
                "type": "cpu_optimization",
                "current_value": cpu_utilization,
                "target_value": 0.6,
                "priority": "medium",
                "strategy": "load_balancing"
            })
        
        # Check error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > 0.05:  # More than 5%
            opportunities.append({
                "type": "reliability_optimization",
                "current_value": error_rate,
                "target_value": 0.02,
                "priority": "high",
                "strategy": "error_handling_improvement"
            })
        
        # Sort by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        opportunities.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)
        
        return opportunities
    
    async def _apply_optimization(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a specific optimization."""
        optimization_type = opportunity["type"]
        strategy = opportunity["strategy"]
        
        # Simulate optimization application
        expected_improvement = (
            (opportunity["current_value"] - opportunity["target_value"]) / 
            opportunity["current_value"]
        )
        
        optimization_result = {
            "optimization_type": optimization_type,
            "strategy": strategy,
            "expected_improvement": expected_improvement,
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "success": True,
            "details": f"Applied {strategy} for {optimization_type}"
        }
        
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Get a report on system optimizations."""
        if not self.optimization_history:
            return {"message": "No optimizations applied yet"}
        
        recent_optimizations = list(self.optimization_history)[-10:]  # Last 10
        
        total_improvement = sum(
            opt.get("expected_improvement", 0) for opt in recent_optimizations
        )
        
        optimization_types = defaultdict(int)
        for opt in recent_optimizations:
            optimization_types[opt["optimization_type"]] += 1
        
        return {
            "total_optimizations": len(self.optimization_history),
            "recent_optimizations": len(recent_optimizations),
            "total_expected_improvement": total_improvement,
            "average_improvement_per_optimization": total_improvement / len(recent_optimizations) if recent_optimizations else 0,
            "optimization_breakdown": dict(optimization_types),
            "most_common_optimization": max(optimization_types.items(), key=lambda x: x[1])[0] if optimization_types else None,
            "report_generated_at": datetime.now(timezone.utc).isoformat()
        }


# Main Intelligence Engine Class
class IntelligenceEngine:
    """
    Main Intelligence Engine that orchestrates all AI capabilities.
    
    This is the central component that coordinates decision making, task planning,
    learning, and agent coordination to provide sophisticated AI-driven automation.
    """
    
    def __init__(self, config: Any, logger: logging.Logger, ollama_service: OllamaService, message_broker: Any):
        """Initialize the Intelligence Engine."""
        self.config = config
        self.logger = logger
        self.ollama_service = ollama_service
        self.message_broker = message_broker
        
        # Core components
        self.decision_maker = DecisionMaker(ollama_service)
        self.task_planner = TaskPlanner()
        self.learning_system = LearningSystem()
        self.agent_coordinator = AgentCoordinator(message_broker)
        
        # Advanced components
        self.multi_step_planning = MultiStepPlanning()
        self.workflow_orchestration = WorkflowOrchestration()
        self.intelligent_prioritization = IntelligentPrioritization()
        self.cross_agent_learning = CrossAgentLearning()
        self.automated_optimization = AutomatedOptimization()
        
        # System state
        self.is_initialized = False
        self.knowledge_base = {}
        self.context_awareness = ContextAwareness()
        self.pattern_recognition = PatternRecognition()
        
        # Performance tracking
        self.start_time = time.time()
        self.total_decisions = 0
        self.total_coordinations = 0
        self.total_learning_events = 0
    
    async def initialize(self) -> None:
        """Initialize the Intelligence Engine."""
        try:
            self.logger.info("Initializing Intelligence Engine...")
            
            # Initialize core components
            await self._initialize_knowledge_base()
            await self._initialize_context_awareness()
            await self._initialize_pattern_recognition()
            
            # Load configuration
            await self._load_intelligence_config()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_initialized = True
            self.logger.info("Intelligence Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Intelligence Engine: {e}")
            raise IntelligenceError(f"Initialization failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the Intelligence Engine."""
        try:
            self.logger.info("Shutting down Intelligence Engine...")
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Save state
            await self._save_intelligence_state()
            
            self.is_initialized = False
            self.logger.info("Intelligence Engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during Intelligence Engine shutdown: {e}")
    
    async def make_decision(self, context: DecisionContext) -> Decision:
        """Make an intelligent decision."""
        if not self.is_initialized:
            raise IntelligenceError("Intelligence Engine not initialized")
        
        try:
            decision = await self.decision_maker.make_decision(context)
            self.total_decisions += 1
            return decision
        except Exception as e:
            raise DecisionError(f"Decision making failed: {e}")
    
    async def create_task_plan(self, task_definition: Dict[str, Any]) -> TaskPlan:
        """Create a comprehensive task plan."""
        if not self.is_initialized:
            raise IntelligenceError("Intelligence Engine not initialized")
        
        return await self.task_planner.create_plan(task_definition)
    
    async def coordinate_agents(self, coordination_request: Dict[str, Any]) -> CoordinationResult:
        """Coordinate multiple agents."""
        if not self.is_initialized:
            raise IntelligenceError("Intelligence Engine not initialized")
        
        result = await self.agent_coordinator.coordinate_agents(coordination_request)
        self.total_coordinations += 1
        return result
    
    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from user feedback."""
        if not self.is_initialized:
            raise IntelligenceError("Intelligence Engine not initialized")
        
        try:
            learning_result = await self.learning_system.learn_from_feedback(feedback)
            self.total_learning_events += 1
            
            # Share learning insights across agents if applicable
            if learning_result.get("learning_applied"):
                await self.cross_agent_learning.share_learning_insight(
                    source_agent="intelligence_engine",
                    insight={
                        "type": "feedback_learning",
                        "knowledge": learning_result,
                        "confidence": 0.8,
                        "domains": [feedback.get("domain", "general")]
                    }
                )
            
            return learning_result
            
        except Exception as e:
            raise LearningError(f"Learning from feedback failed: {e}")
    
    async def optimize_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a workflow."""
        if not self.is_initialized:
            raise IntelligenceError("Intelligence Engine not initialized")
        
        # Use automated optimization
        optimization_result = await self.automated_optimization.optimize_system_performance(workflow_data)
        
        return {
            "optimizations_found": len(optimization_result.get("optimizations_applied", [])) > 0,
            "recommended_changes": optimization_result.get("optimizations_applied", []),
            "estimated_improvement": optimization_result.get("expected_improvement", 0)
        }
    
    async def get_performance_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics."""
        uptime = time.time() - self.start_time
        
        # Calculate success rates
        coordination_success_rate = 0.9  # Placeholder - would calculate from actual data
        
        # Get learning metrics
        learning_metrics = await self.learning_system.get_learning_metrics()
        
        return PerformanceMetrics(
            total_decisions=self.total_decisions,
            average_confidence=0.82,  # Placeholder - would calculate from decision history
            learning_progress=learning_metrics.accuracy_improvement,
            coordination_success_rate=coordination_success_rate,
            optimization_improvements=0.15,  # Placeholder
            user_satisfaction=0.88  # Placeholder
        )
    
    async def get_agent_insights(self, agent_id: str) -> Dict[str, Any]:
        """Get insights about a specific agent."""
        # Get relevant learning insights
        relevant_insights = await self.cross_agent_learning.get_relevant_insights(agent_id, "general")
        
        return {
            "agent_id": agent_id,
            "performance": {
                "accuracy": 0.85,
                "speed": 0.78,
                "reliability": 0.92
            },
            "learning_progress": {
                "learning_rate": 0.1,
                "adaptations_applied": len(relevant_insights),
                "improvement_trend": "positive"
            },
            "optimization_suggestions": [
                {"type": "resource_allocation", "priority": "high"},
                {"type": "workflow_optimization", "priority": "medium"}
            ],
            "relevant_insights": relevant_insights[:5]  # Top 5 insights
        }
    
    async def update_user_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        try:
            # Update learning system with new preferences
            await self.learning_system.learning_history.append({
                "event_type": "preference_update",
                "data": preferences,
                "timestamp": datetime.now(timezone.utc)
            })
            
            return True
        except Exception:
            return False
    
    async def predict_optimal_actions(self, prediction_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict optimal actions for a given context."""
        # Use intelligent prioritization for action prediction
        possible_actions = prediction_context.get("possible_actions", [
            {"action": "prioritize_urgent", "urgency": "high", "importance": "high"},
            {"action": "batch_process", "urgency": "medium", "importance": "medium"},
            {"action": "delegate_tasks", "urgency": "low", "importance": "medium"}
        ])
        
        prioritized_actions = await self.intelligent_prioritization.prioritize_tasks(
            possible_actions, prediction_context
        )
        
        return [
            {
                "action": action["action"],
                "confidence": action["priority_score"],
                "reasoning": action.get("priority_reasoning", "")
            }
            for action in prioritized_actions
        ]
    
    # Internal methods
    async def _initialize_knowledge_base(self) -> None:
        """Initialize the knowledge base."""
        self.knowledge_base = {
            "initialized": True,
            "version": "1.0",
            "domains": {
                "email_processing": {},
                "research": {},
                "task_management": {},
                "user_preferences": {}
            },
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def _initialize_context_awareness(self) -> None:
        """Initialize context awareness."""
        self.context_awareness = ContextAwareness()
    
    async def _initialize_pattern_recognition(self) -> None:
        """Initialize pattern recognition."""
        self.pattern_recognition = PatternRecognition()
    
    async def _load_intelligence_config(self) -> None:
        """Load intelligence configuration."""
        # Load configuration settings
        pass
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks."""
        # Start periodic optimization checks, pattern recognition, etc.
        pass
    
    async def _stop_background_tasks(self) -> None:
        """Stop background tasks."""
        pass
    
    async def _save_intelligence_state(self) -> None:
        """Save current intelligence state."""
        pass