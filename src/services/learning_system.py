"""
Learning System component for the Intelligence Layer.

This module provides user preference learning, performance optimization,
pattern recognition, and adaptive behavior capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

from .intelligence_engine import (
    LearningMetrics, UserPreferences, PerformanceMetrics,
    LearningError, IntelligenceError
)


@dataclass
class LearningEvent:
    """Represents a learning event."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""  # feedback, performance, pattern, adaptation
    source: str = ""  # user, system, agent
    data: Dict[str, Any] = field(default_factory=dict)
    outcome: str = ""  # success, failure, partial
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "data": self.data,
            "outcome": self.outcome,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }


@dataclass
class Pattern:
    """Represents a recognized pattern."""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str = ""
    description: str = ""
    confidence: float = 0.0
    frequency: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_observed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "confidence": self.confidence,
            "frequency": self.frequency,
            "data": self.data,
            "discovered_at": self.discovered_at.isoformat(),
            "last_observed": self.last_observed.isoformat()
        }


@dataclass
class Adaptation:
    """Represents an adaptation made by the system."""
    adaptation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    adaptation_type: str = ""
    target_component: str = ""
    changes_made: Dict[str, Any] = field(default_factory=dict)
    expected_improvement: float = 0.0
    actual_improvement: Optional[float] = None
    success: bool = False
    applied_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "adaptation_id": self.adaptation_id,
            "adaptation_type": self.adaptation_type,
            "target_component": self.target_component,
            "changes_made": self.changes_made,
            "expected_improvement": self.expected_improvement,
            "actual_improvement": self.actual_improvement,
            "success": self.success,
            "applied_at": self.applied_at.isoformat()
        }


class UserPreferenceLearning:
    """Learns and adapts to user preferences."""
    
    def __init__(self):
        self.user_preferences = UserPreferences()
        self.preference_history = deque(maxlen=1000)
        self.preference_models = {}
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        
        self._initialize_preference_models()
    
    def _initialize_preference_models(self):
        """Initialize machine learning models for preference learning."""
        self.preference_models = {
            "email_priority": RandomForestClassifier(n_estimators=50, random_state=42),
            "communication_style": RandomForestClassifier(n_estimators=30, random_state=42),
            "workflow_efficiency": GradientBoostingRegressor(n_estimators=50, random_state=42),
            "content_relevance": RandomForestClassifier(n_estimators=40, random_state=42)
        }
        
        # Initialize with dummy data to allow predictions
        for model_name, model in self.preference_models.items():
            if hasattr(model, 'fit'):
                # Create dummy training data
                X_dummy = np.random.random((10, 5))
                if isinstance(model, RandomForestClassifier):
                    y_dummy = np.random.randint(0, 3, 10)  # 3 classes
                else:
                    y_dummy = np.random.random(10)
                model.fit(X_dummy, y_dummy)
    
    async def learn_from_user_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from user feedback and update preferences."""
        try:
            feedback_type = feedback.get("feedback_type", "general")
            user_rating = feedback.get("user_rating", 3)
            context = feedback.get("context", {})
            
            learning_result = {
                "preference_updates": [],
                "confidence_changes": {},
                "learning_applied": False
            }
            
            # Update preferences based on feedback type
            if feedback_type == "email_priority":
                updates = await self._learn_email_preferences(feedback, user_rating, context)
                learning_result["preference_updates"].extend(updates)
            
            elif feedback_type == "communication_style":
                updates = await self._learn_communication_preferences(feedback, user_rating, context)
                learning_result["preference_updates"].extend(updates)
            
            elif feedback_type == "workflow_efficiency":
                updates = await self._learn_workflow_preferences(feedback, user_rating, context)
                learning_result["preference_updates"].extend(updates)
            
            # Record preference learning event
            learning_event = LearningEvent(
                event_type="preference_learning",
                source="user_feedback",
                data=feedback,
                outcome="success" if user_rating >= 4 else "partial" if user_rating >= 3 else "failure",
                confidence=self._calculate_feedback_confidence(user_rating),
                context=context
            )
            
            self.preference_history.append(learning_event)
            learning_result["learning_applied"] = True
            
            return learning_result
            
        except Exception as e:
            raise LearningError(f"Failed to learn from user feedback: {str(e)}")
    
    async def _learn_email_preferences(self, feedback: Dict[str, Any], rating: int, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Learn email prioritization preferences."""
        updates = []
        
        email_data = context.get("email", {})
        sender = email_data.get("sender", "").lower()
        subject = email_data.get("subject", "").lower()
        priority_assigned = context.get("priority_assigned", "medium")
        
        # Update sender importance weights
        if rating >= 4:  # Positive feedback
            if "boss" in sender or "manager" in sender:
                self.user_preferences.email_priority_factors["leadership_weight"] = min(
                    self.user_preferences.email_priority_factors.get("leadership_weight", 0.8) + 0.1, 1.0
                )
                updates.append({
                    "type": "sender_importance",
                    "change": "increased leadership weight",
                    "new_value": self.user_preferences.email_priority_factors["leadership_weight"]
                })
            
            if "client" in sender or "customer" in sender:
                self.user_preferences.email_priority_factors["client_weight"] = min(
                    self.user_preferences.email_priority_factors.get("client_weight", 0.7) + 0.1, 1.0
                )
                updates.append({
                    "type": "sender_importance",
                    "change": "increased client weight",
                    "new_value": self.user_preferences.email_priority_factors["client_weight"]
                })
        
        elif rating <= 2:  # Negative feedback
            if priority_assigned == "high":
                # Reduce importance of similar patterns
                urgency_weight = self.user_preferences.email_priority_factors.get("urgency_keywords", 0.8)
                self.user_preferences.email_priority_factors["urgency_keywords"] = max(urgency_weight - 0.05, 0.3)
                updates.append({
                    "type": "keyword_importance",
                    "change": "reduced urgency keyword weight",
                    "new_value": self.user_preferences.email_priority_factors["urgency_keywords"]
                })
        
        # Learn time-based preferences
        current_hour = datetime.now().hour
        if rating >= 4:
            time_key = f"preferred_hour_{current_hour}"
            current_preference = self.user_preferences.email_priority_factors.get(time_key, 0.5)
            self.user_preferences.email_priority_factors[time_key] = min(current_preference + 0.05, 1.0)
            updates.append({
                "type": "time_preference",
                "change": f"increased preference for hour {current_hour}",
                "new_value": self.user_preferences.email_priority_factors[time_key]
            })
        
        return updates
    
    async def _learn_communication_preferences(self, feedback: Dict[str, Any], rating: int, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Learn communication style preferences."""
        updates = []
        
        response_style = context.get("response_style", "")
        message_length = context.get("message_length", "medium")
        formality_level = context.get("formality_level", "professional")
        
        if rating >= 4:  # Positive feedback
            # Increase preference for successful style
            style_preferences = self.user_preferences.workflow_preferences.get("communication_style", {})
            
            if response_style:
                current_weight = style_preferences.get(response_style, 0.5)
                style_preferences[response_style] = min(current_weight + 0.1, 1.0)
                updates.append({
                    "type": "response_style",
                    "change": f"increased preference for {response_style} style",
                    "new_value": style_preferences[response_style]
                })
            
            if message_length:
                length_preferences = style_preferences.get("length_preference", {})
                current_weight = length_preferences.get(message_length, 0.5)
                length_preferences[message_length] = min(current_weight + 0.1, 1.0)
                style_preferences["length_preference"] = length_preferences
                updates.append({
                    "type": "message_length",
                    "change": f"increased preference for {message_length} messages",
                    "new_value": length_preferences[message_length]
                })
            
            self.user_preferences.workflow_preferences["communication_style"] = style_preferences
        
        return updates
    
    async def _learn_workflow_preferences(self, feedback: Dict[str, Any], rating: int, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Learn workflow efficiency preferences."""
        updates = []
        
        workflow_type = context.get("workflow_type", "")
        coordination_style = context.get("coordination_style", "sequential")
        automation_level = context.get("automation_level", "medium")
        
        if rating >= 4:  # Positive feedback
            workflow_prefs = self.user_preferences.workflow_preferences
            
            # Update coordination style preference
            if coordination_style:
                coord_key = f"{coordination_style}_preference"
                current_value = workflow_prefs.get(coord_key, 0.5)
                workflow_prefs[coord_key] = min(current_value + 0.1, 1.0)
                updates.append({
                    "type": "coordination_style",
                    "change": f"increased preference for {coordination_style} coordination",
                    "new_value": workflow_prefs[coord_key]
                })
            
            # Update automation preference
            if automation_level:
                auto_key = f"automation_{automation_level}"
                current_value = workflow_prefs.get(auto_key, 0.5)
                workflow_prefs[auto_key] = min(current_value + 0.1, 1.0)
                updates.append({
                    "type": "automation_level",
                    "change": f"increased preference for {automation_level} automation",
                    "new_value": workflow_prefs[auto_key]
                })
        
        elif rating <= 2:  # Negative feedback
            # Reduce preference for unsuccessful patterns
            workflow_prefs = self.user_preferences.workflow_preferences
            
            if coordination_style:
                coord_key = f"{coordination_style}_preference"
                current_value = workflow_prefs.get(coord_key, 0.5)
                workflow_prefs[coord_key] = max(current_value - 0.05, 0.1)
                updates.append({
                    "type": "coordination_style",
                    "change": f"decreased preference for {coordination_style} coordination",
                    "new_value": workflow_prefs[coord_key]
                })
        
        return updates
    
    def _calculate_feedback_confidence(self, rating: int) -> float:
        """Calculate confidence level from user rating."""
        # Convert 1-5 rating to confidence (0-1)
        if rating >= 5:
            return 0.95
        elif rating >= 4:
            return 0.8
        elif rating >= 3:
            return 0.6
        elif rating >= 2:
            return 0.4
        else:
            return 0.2
    
    async def predict_user_preference(self, context: Dict[str, Any], preference_type: str) -> Dict[str, Any]:
        """Predict user preference for a given context."""
        try:
            if preference_type not in self.preference_models:
                return {"prediction": "medium", "confidence": 0.5}
            
            model = self.preference_models[preference_type]
            
            # Extract features from context
            features = self._extract_preference_features(context, preference_type)
            
            if len(features) < 5:  # Pad features if needed
                features.extend([0.5] * (5 - len(features)))
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba([features[:5]])[0]
                predicted_class = model.predict([features[:5]])[0]
                confidence = max(prediction_proba)
                
                # Map class to preference level
                if preference_type == "email_priority":
                    preference_map = {0: "low", 1: "medium", 2: "high"}
                    prediction = preference_map.get(predicted_class, "medium")
                else:
                    prediction = "high" if predicted_class >= 1 else "low"
            else:
                prediction_value = model.predict([features[:5]])[0]
                prediction = "high" if prediction_value > 0.6 else "medium" if prediction_value > 0.4 else "low"
                confidence = abs(prediction_value - 0.5) * 2  # Convert to confidence
            
            return {
                "prediction": prediction,
                "confidence": min(confidence, 1.0),
                "features_used": features[:5],
                "model_type": preference_type
            }
            
        except Exception as e:
            return {
                "prediction": "medium",
                "confidence": 0.5,
                "error": str(e)
            }
    
    def _extract_preference_features(self, context: Dict[str, Any], preference_type: str) -> List[float]:
        """Extract features for preference prediction."""
        features = []
        
        if preference_type == "email_priority":
            email = context.get("email", {})
            subject = email.get("subject", "").lower()
            sender = email.get("sender", "").lower()
            
            # Feature 1: Urgency keywords
            urgency_score = sum(1 for keyword in ["urgent", "asap", "critical"] if keyword in subject) / 3
            features.append(urgency_score)
            
            # Feature 2: Sender importance
            sender_score = 0.8 if any(term in sender for term in ["boss", "manager"]) else 0.5
            features.append(sender_score)
            
            # Feature 3: Time of day
            hour = datetime.now().hour
            time_score = 0.8 if 9 <= hour <= 17 else 0.4  # Business hours
            features.append(time_score)
            
            # Feature 4: Subject length
            length_score = min(len(subject) / 100, 1.0)  # Normalize to 0-1
            features.append(length_score)
            
            # Feature 5: Meeting-related
            meeting_score = 1.0 if "meeting" in subject else 0.0
            features.append(meeting_score)
        
        elif preference_type == "communication_style":
            # Extract communication features
            features = [
                context.get("formality_required", 0.5),
                context.get("urgency_level", 0.5),
                context.get("recipient_type", 0.5),
                context.get("message_complexity", 0.5),
                context.get("response_expected", 0.5)
            ]
        
        else:
            # Default features
            features = [0.5, 0.5, 0.5, 0.5, 0.5]
        
        return features
    
    async def get_user_preferences(self) -> UserPreferences:
        """Get current user preferences."""
        return self.user_preferences
    
    async def update_preferences(self, preference_updates: Dict[str, Any]) -> bool:
        """Update user preferences directly."""
        try:
            if "email_priority_factors" in preference_updates:
                self.user_preferences.email_priority_factors.update(
                    preference_updates["email_priority_factors"]
                )
            
            if "research_preferences" in preference_updates:
                self.user_preferences.research_preferences.update(
                    preference_updates["research_preferences"]
                )
            
            if "workflow_preferences" in preference_updates:
                self.user_preferences.workflow_preferences.update(
                    preference_updates["workflow_preferences"]
                )
            
            if "notification_preferences" in preference_updates:
                self.user_preferences.notification_preferences.update(
                    preference_updates["notification_preferences"]
                )
            
            return True
            
        except Exception as e:
            raise LearningError(f"Failed to update preferences: {str(e)}")


class PerformanceOptimization:
    """Optimizes system performance based on learning."""
    
    def __init__(self):
        self.performance_history = deque(maxlen=1000)
        self.optimization_models = {}
        self.optimization_strategies = [
            "resource_allocation",
            "workflow_sequencing", 
            "agent_assignment",
            "parallel_processing",
            "caching_strategy"
        ]
        self.active_optimizations = {}
        
        self._initialize_optimization_models()
    
    def _initialize_optimization_models(self):
        """Initialize optimization models."""
        self.optimization_models = {
            "resource_allocation": GradientBoostingRegressor(n_estimators=50, random_state=42),
            "workflow_efficiency": RandomForestClassifier(n_estimators=40, random_state=42),
            "agent_performance": GradientBoostingRegressor(n_estimators=30, random_state=42)
        }
        
        # Initialize with dummy data
        for model_name, model in self.optimization_models.items():
            X_dummy = np.random.random((20, 6))
            if isinstance(model, RandomForestClassifier):
                y_dummy = np.random.randint(0, 3, 20)
            else:
                y_dummy = np.random.random(20)
            model.fit(X_dummy, y_dummy)
    
    async def analyze_performance_data(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance data to identify optimization opportunities."""
        try:
            analysis_result = {
                "bottlenecks_identified": [],
                "optimization_opportunities": [],
                "performance_trends": {},
                "recommendations": []
            }
            
            if not performance_data:
                return analysis_result
            
            # Identify bottlenecks
            bottlenecks = await self._identify_bottlenecks(performance_data)
            analysis_result["bottlenecks_identified"] = bottlenecks
            
            # Find optimization opportunities
            opportunities = await self._find_optimization_opportunities(performance_data)
            analysis_result["optimization_opportunities"] = opportunities
            
            # Analyze trends
            trends = await self._analyze_performance_trends(performance_data)
            analysis_result["performance_trends"] = trends
            
            # Generate recommendations
            recommendations = await self._generate_optimization_recommendations(
                bottlenecks, opportunities, trends
            )
            analysis_result["recommendations"] = recommendations
            
            return analysis_result
            
        except Exception as e:
            raise LearningError(f"Failed to analyze performance data: {str(e)}")
    
    async def _identify_bottlenecks(self, performance_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Analyze response times
        response_times = [item.get("response_time", 0) for item in performance_data if "response_time" in item]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            if max_response_time > avg_response_time * 3:  # Outlier detection
                bottlenecks.append({
                    "type": "response_time",
                    "severity": "high" if max_response_time > 10 else "medium",
                    "description": f"Maximum response time ({max_response_time:.2f}s) significantly exceeds average ({avg_response_time:.2f}s)",
                    "affected_component": "response_processing"
                })
        
        # Analyze resource utilization
        resource_data = [item.get("resource_utilization", {}) for item in performance_data if "resource_utilization" in item]
        if resource_data:
            avg_cpu = sum(res.get("cpu", 0) for res in resource_data) / len(resource_data)
            avg_memory = sum(res.get("memory", 0) for res in resource_data) / len(resource_data)
            
            if avg_cpu > 0.8:
                bottlenecks.append({
                    "type": "cpu_utilization",
                    "severity": "high",
                    "description": f"Average CPU utilization ({avg_cpu:.2%}) is above recommended threshold",
                    "affected_component": "compute_resources"
                })
            
            if avg_memory > 0.85:
                bottlenecks.append({
                    "type": "memory_utilization", 
                    "severity": "high",
                    "description": f"Average memory utilization ({avg_memory:.2%}) is above recommended threshold",
                    "affected_component": "memory_resources"
                })
        
        # Analyze error rates
        error_data = [item.get("error_rate", 0) for item in performance_data if "error_rate" in item]
        if error_data:
            avg_error_rate = sum(error_data) / len(error_data)
            
            if avg_error_rate > 0.05:  # 5% error rate threshold
                bottlenecks.append({
                    "type": "error_rate",
                    "severity": "high" if avg_error_rate > 0.1 else "medium",
                    "description": f"Average error rate ({avg_error_rate:.2%}) exceeds acceptable threshold",
                    "affected_component": "error_handling"
                })
        
        return bottlenecks
    
    async def _find_optimization_opportunities(self, performance_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find optimization opportunities."""
        opportunities = []
        
        # Analyze task distribution
        agent_workloads = defaultdict(list)
        for item in performance_data:
            agent_id = item.get("agent_id")
            workload = item.get("workload", 0)
            if agent_id and workload:
                agent_workloads[agent_id].append(workload)
        
        if len(agent_workloads) > 1:
            # Calculate workload variance
            avg_workloads = {agent: sum(loads)/len(loads) for agent, loads in agent_workloads.items()}
            workload_values = list(avg_workloads.values())
            
            if workload_values:
                variance = np.var(workload_values)
                if variance > 0.1:  # High variance in workload distribution
                    opportunities.append({
                        "type": "load_balancing",
                        "potential_improvement": "20-30%",
                        "description": "Uneven workload distribution detected - load balancing optimization recommended",
                        "strategy": "redistribute_tasks"
                    })
        
        # Analyze task parallelization opportunities
        sequential_tasks = [item for item in performance_data if item.get("coordination_type") == "sequential"]
        if len(sequential_tasks) > 5:
            opportunities.append({
                "type": "parallelization",
                "potential_improvement": "15-25%",
                "description": "Multiple sequential tasks could benefit from parallel execution",
                "strategy": "convert_to_parallel"
            })
        
        # Analyze caching opportunities
        repeated_operations = defaultdict(int)
        for item in performance_data:
            operation_type = item.get("operation_type")
            if operation_type:
                repeated_operations[operation_type] += 1
        
        for operation, count in repeated_operations.items():
            if count > 10:  # Frequently repeated operation
                opportunities.append({
                    "type": "caching",
                    "potential_improvement": "10-20%",
                    "description": f"Operation '{operation}' repeated {count} times - caching recommended",
                    "strategy": "implement_caching"
                })
        
        return opportunities
    
    async def _analyze_performance_trends(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        trends = {}
        
        # Sort data by timestamp
        timestamped_data = [item for item in performance_data if "timestamp" in item]
        timestamped_data.sort(key=lambda x: x["timestamp"])
        
        if len(timestamped_data) < 5:
            return {"insufficient_data": True}
        
        # Analyze response time trends
        response_times = [item.get("response_time", 0) for item in timestamped_data]
        if response_times:
            x = np.arange(len(response_times))
            z = np.polyfit(x, response_times, 1)
            trend_slope = z[0]
            
            trends["response_time"] = {
                "direction": "increasing" if trend_slope > 0.01 else "decreasing" if trend_slope < -0.01 else "stable",
                "slope": trend_slope,
                "significance": "high" if abs(trend_slope) > 0.05 else "low"
            }
        
        # Analyze success rate trends
        success_rates = []
        for item in timestamped_data:
            total_ops = item.get("total_operations", 1)
            successful_ops = item.get("successful_operations", total_ops)
            success_rate = successful_ops / total_ops if total_ops > 0 else 1.0
            success_rates.append(success_rate)
        
        if success_rates:
            x = np.arange(len(success_rates))
            z = np.polyfit(x, success_rates, 1)
            trend_slope = z[0]
            
            trends["success_rate"] = {
                "direction": "improving" if trend_slope > 0.01 else "declining" if trend_slope < -0.01 else "stable",
                "slope": trend_slope,
                "current_rate": success_rates[-1] if success_rates else 0.0
            }
        
        return trends
    
    async def _generate_optimization_recommendations(self, bottlenecks: List[Dict[str, Any]], opportunities: List[Dict[str, Any]], trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Recommendations based on bottlenecks
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "response_time":
                recommendations.append({
                    "type": "performance",
                    "priority": "high",
                    "action": "Implement response time optimization",
                    "details": "Add caching and optimize slow operations",
                    "expected_improvement": "30-50% response time reduction"
                })
            
            elif bottleneck["type"] == "cpu_utilization":
                recommendations.append({
                    "type": "resource",
                    "priority": "high", 
                    "action": "Scale CPU resources or optimize CPU-intensive operations",
                    "details": "Consider horizontal scaling or algorithm optimization",
                    "expected_improvement": "20-30% CPU utilization reduction"
                })
            
            elif bottleneck["type"] == "memory_utilization":
                recommendations.append({
                    "type": "resource",
                    "priority": "high",
                    "action": "Optimize memory usage or increase memory allocation",
                    "details": "Implement memory pooling and garbage collection optimization",
                    "expected_improvement": "15-25% memory utilization reduction"
                })
        
        # Recommendations based on opportunities
        for opportunity in opportunities:
            if opportunity["type"] == "load_balancing":
                recommendations.append({
                    "type": "architecture",
                    "priority": "medium",
                    "action": "Implement intelligent load balancing",
                    "details": "Redistribute tasks based on agent capacity and performance",
                    "expected_improvement": opportunity["potential_improvement"]
                })
            
            elif opportunity["type"] == "parallelization":
                recommendations.append({
                    "type": "workflow",
                    "priority": "medium",
                    "action": "Convert sequential workflows to parallel where possible",
                    "details": "Identify independent tasks and enable parallel execution",
                    "expected_improvement": opportunity["potential_improvement"]
                })
            
            elif opportunity["type"] == "caching":
                recommendations.append({
                    "type": "optimization",
                    "priority": "low",
                    "action": "Implement intelligent caching",
                    "details": "Cache frequently accessed data and computation results",
                    "expected_improvement": opportunity["potential_improvement"]
                })
        
        # Recommendations based on trends
        if trends.get("response_time", {}).get("direction") == "increasing":
            recommendations.append({
                "type": "trend",
                "priority": "medium",
                "action": "Address performance degradation trend",
                "details": "Investigate causes of increasing response times",
                "expected_improvement": "Prevent further performance degradation"
            })
        
        if trends.get("success_rate", {}).get("direction") == "declining":
            recommendations.append({
                "type": "reliability",
                "priority": "high",
                "action": "Improve system reliability",
                "details": "Investigate and fix causes of declining success rates",
                "expected_improvement": "Restore and maintain high success rates"
            })
        
        return recommendations
    
    async def apply_optimization(self, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a specific optimization."""
        try:
            optimization_type = optimization_config.get("type")
            target_component = optimization_config.get("target_component", "system")
            expected_improvement = optimization_config.get("expected_improvement", 0.1)
            
            # Create adaptation record
            adaptation = Adaptation(
                adaptation_type=optimization_type,
                target_component=target_component,
                changes_made=optimization_config.get("changes", {}),
                expected_improvement=expected_improvement
            )
            
            # Apply optimization based on type
            if optimization_type == "resource_allocation":
                result = await self._apply_resource_optimization(optimization_config)
            elif optimization_type == "workflow_sequencing":
                result = await self._apply_workflow_optimization(optimization_config)
            elif optimization_type == "caching_strategy":
                result = await self._apply_caching_optimization(optimization_config)
            else:
                result = {"success": False, "reason": f"Unknown optimization type: {optimization_type}"}
            
            # Update adaptation with results
            adaptation.success = result.get("success", False)
            adaptation.actual_improvement = result.get("actual_improvement")
            
            # Store optimization
            self.active_optimizations[adaptation.adaptation_id] = adaptation
            
            return {
                "optimization_id": adaptation.adaptation_id,
                "success": adaptation.success,
                "expected_improvement": expected_improvement,
                "actual_improvement": adaptation.actual_improvement,
                "details": result
            }
            
        except Exception as e:
            raise LearningError(f"Failed to apply optimization: {str(e)}")
    
    async def _apply_resource_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply resource allocation optimization."""
        # Simulate resource optimization
        return {
            "success": True,
            "actual_improvement": 0.15,
            "changes_applied": [
                "Adjusted agent workload distribution",
                "Optimized resource allocation algorithm",
                "Implemented dynamic scaling"
            ]
        }
    
    async def _apply_workflow_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply workflow sequencing optimization."""
        # Simulate workflow optimization
        return {
            "success": True,
            "actual_improvement": 0.22,
            "changes_applied": [
                "Converted sequential steps to parallel",
                "Optimized task dependencies",
                "Improved coordination efficiency"
            ]
        }
    
    async def _apply_caching_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply caching optimization."""
        # Simulate caching optimization
        return {
            "success": True,
            "actual_improvement": 0.18,
            "changes_applied": [
                "Implemented intelligent caching layer",
                "Added cache invalidation strategy",
                "Optimized cache hit rates"
            ]
        }


class AdaptationEngine:
    """Manages system adaptations based on learning."""
    
    def __init__(self):
        self.adaptation_history = deque(maxlen=500)
        self.adaptation_strategies = {}
        self.adaptation_rules = []
        self.adaptation_thresholds = {
            "performance_degradation": 0.1,
            "error_rate_increase": 0.05,
            "user_satisfaction_drop": 0.15,
            "efficiency_decrease": 0.2
        }
        
        self._initialize_adaptation_strategies()
    
    def _initialize_adaptation_strategies(self):
        """Initialize adaptation strategies."""
        self.adaptation_strategies = {
            "performance_degradation": [
                "increase_parallelism",
                "optimize_algorithms", 
                "scale_resources",
                "implement_caching"
            ],
            "error_rate_increase": [
                "improve_error_handling",
                "add_retry_mechanisms",
                "enhance_validation",
                "implement_fallbacks"
            ],
            "user_satisfaction_drop": [
                "adjust_preferences",
                "improve_response_quality",
                "optimize_user_experience",
                "enhance_personalization"
            ],
            "resource_constraints": [
                "optimize_resource_usage",
                "implement_resource_pooling",
                "adjust_task_priorities",
                "enable_load_balancing"
            ]
        }
    
    async def evaluate_adaptation_need(self, system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if system adaptation is needed."""
        try:
            adaptation_needed = False
            adaptation_reasons = []
            recommended_adaptations = []
            
            # Check performance metrics
            current_performance = system_metrics.get("performance", {})
            baseline_performance = system_metrics.get("baseline_performance", {})
            
            if baseline_performance:
                # Check for performance degradation
                perf_change = self._calculate_performance_change(current_performance, baseline_performance)
                if perf_change < -self.adaptation_thresholds["performance_degradation"]:
                    adaptation_needed = True
                    adaptation_reasons.append(f"Performance degraded by {abs(perf_change):.2%}")
                    recommended_adaptations.extend(self.adaptation_strategies["performance_degradation"])
            
            # Check error rates
            current_error_rate = system_metrics.get("error_rate", 0.0)
            baseline_error_rate = system_metrics.get("baseline_error_rate", 0.02)
            
            if current_error_rate > baseline_error_rate + self.adaptation_thresholds["error_rate_increase"]:
                adaptation_needed = True
                error_increase = current_error_rate - baseline_error_rate
                adaptation_reasons.append(f"Error rate increased by {error_increase:.2%}")
                recommended_adaptations.extend(self.adaptation_strategies["error_rate_increase"])
            
            # Check user satisfaction
            current_satisfaction = system_metrics.get("user_satisfaction", 0.8)
            baseline_satisfaction = system_metrics.get("baseline_satisfaction", 0.8)
            
            satisfaction_drop = baseline_satisfaction - current_satisfaction
            if satisfaction_drop > self.adaptation_thresholds["user_satisfaction_drop"]:
                adaptation_needed = True
                adaptation_reasons.append(f"User satisfaction dropped by {satisfaction_drop:.2%}")
                recommended_adaptations.extend(self.adaptation_strategies["user_satisfaction_drop"])
            
            # Check resource utilization
            resource_metrics = system_metrics.get("resource_utilization", {})
            if any(util > 0.9 for util in resource_metrics.values()):
                adaptation_needed = True
                adaptation_reasons.append("High resource utilization detected")
                recommended_adaptations.extend(self.adaptation_strategies["resource_constraints"])
            
            # Remove duplicates from recommendations
            recommended_adaptations = list(set(recommended_adaptations))
            
            return {
                "adaptation_needed": adaptation_needed,
                "reasons": adaptation_reasons,
                "recommended_adaptations": recommended_adaptations,
                "priority": "high" if len(adaptation_reasons) > 2 else "medium" if adaptation_needed else "low",
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            raise LearningError(f"Failed to evaluate adaptation need: {str(e)}")
    
    def _calculate_performance_change(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> float:
        """Calculate performance change as a percentage."""
        current_score = self._calculate_composite_score(current)
        baseline_score = self._calculate_composite_score(baseline)
        
        if baseline_score == 0:
            return 0.0
        
        return (current_score - baseline_score) / baseline_score
    
    def _calculate_composite_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate a composite performance score."""
        weights = {
            "response_time": -0.3,  # Lower is better
            "success_rate": 0.4,    # Higher is better
            "throughput": 0.2,      # Higher is better
            "efficiency": 0.1       # Higher is better
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                if metric == "response_time":
                    # Invert response time (lower is better)
                    normalized_value = max(0, 1 - (value / 10))  # Assume 10s is terrible
                else:
                    normalized_value = min(value, 1.0)  # Cap at 1.0
                
                score += normalized_value * abs(weight)
                total_weight += abs(weight)
        
        return score / total_weight if total_weight > 0 else 0.5
    
    async def create_adaptation_plan(self, adaptation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed adaptation plan."""
        try:
            adaptation_type = adaptation_config.get("type", "performance")
            target_metrics = adaptation_config.get("target_metrics", {})
            constraints = adaptation_config.get("constraints", {})
            
            # Select adaptation strategies
            available_strategies = self.adaptation_strategies.get(adaptation_type, [])
            selected_strategies = self._select_optimal_strategies(
                available_strategies, target_metrics, constraints
            )
            
            # Create adaptation steps
            adaptation_steps = []
            for strategy in selected_strategies:
                steps = await self._create_strategy_steps(strategy, adaptation_config)
                adaptation_steps.extend(steps)
            
            # Estimate impact and timeline
            estimated_impact = self._estimate_adaptation_impact(selected_strategies)
            estimated_timeline = self._estimate_adaptation_timeline(adaptation_steps)
            
            adaptation_plan = {
                "plan_id": str(uuid.uuid4()),
                "adaptation_type": adaptation_type,
                "strategies": selected_strategies,
                "steps": adaptation_steps,
                "estimated_impact": estimated_impact,
                "estimated_timeline": estimated_timeline,
                "target_metrics": target_metrics,
                "constraints": constraints,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            return adaptation_plan
            
        except Exception as e:
            raise LearningError(f"Failed to create adaptation plan: {str(e)}")
    
    def _select_optimal_strategies(self, available_strategies: List[str], target_metrics: Dict[str, Any], constraints: Dict[str, Any]) -> List[str]:
        """Select optimal adaptation strategies."""
        # Simple strategy selection based on constraints and targets
        selected = []
        
        # Consider time constraints
        max_time = constraints.get("max_time", 3600)  # 1 hour default
        
        # Strategy priority based on impact and time
        strategy_priorities = {
            "implement_caching": {"impact": 0.8, "time": 300},
            "increase_parallelism": {"impact": 0.7, "time": 600},
            "optimize_algorithms": {"impact": 0.9, "time": 1800},
            "scale_resources": {"impact": 0.6, "time": 120},
            "improve_error_handling": {"impact": 0.5, "time": 900},
            "add_retry_mechanisms": {"impact": 0.4, "time": 300},
            "enhance_validation": {"impact": 0.3, "time": 600},
            "adjust_preferences": {"impact": 0.6, "time": 180}
        }
        
        # Sort strategies by impact/time ratio
        available_with_priorities = [
            (strategy, strategy_priorities.get(strategy, {"impact": 0.5, "time": 600}))
            for strategy in available_strategies
        ]
        
        available_with_priorities.sort(
            key=lambda x: x[1]["impact"] / (x[1]["time"] / 3600),  # Impact per hour
            reverse=True
        )
        
        # Select strategies within time constraint
        total_time = 0
        for strategy, priority in available_with_priorities:
            if total_time + priority["time"] <= max_time:
                selected.append(strategy)
                total_time += priority["time"]
                
                if len(selected) >= 3:  # Limit to 3 strategies
                    break
        
        return selected if selected else [available_strategies[0]] if available_strategies else []
    
    async def _create_strategy_steps(self, strategy: str, adaptation_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create detailed steps for a strategy."""
        steps = []
        
        if strategy == "implement_caching":
            steps = [
                {
                    "step_id": str(uuid.uuid4()),
                    "description": "Analyze cacheable operations",
                    "duration": 60,
                    "dependencies": []
                },
                {
                    "step_id": str(uuid.uuid4()),
                    "description": "Implement caching layer",
                    "duration": 180,
                    "dependencies": []
                },
                {
                    "step_id": str(uuid.uuid4()),
                    "description": "Configure cache policies",
                    "duration": 60,
                    "dependencies": []
                }
            ]
        
        elif strategy == "increase_parallelism":
            steps = [
                {
                    "step_id": str(uuid.uuid4()),
                    "description": "Identify parallelizable tasks",
                    "duration": 120,
                    "dependencies": []
                },
                {
                    "step_id": str(uuid.uuid4()),
                    "description": "Modify workflow coordination",
                    "duration": 300,
                    "dependencies": []
                },
                {
                    "step_id": str(uuid.uuid4()),
                    "description": "Test parallel execution",
                    "duration": 180,
                    "dependencies": []
                }
            ]
        
        elif strategy == "optimize_algorithms":
            steps = [
                {
                    "step_id": str(uuid.uuid4()),
                    "description": "Profile algorithm performance",
                    "duration": 300,
                    "dependencies": []
                },
                {
                    "step_id": str(uuid.uuid4()),
                    "description": "Implement optimizations",
                    "duration": 900,
                    "dependencies": []
                },
                {
                    "step_id": str(uuid.uuid4()),
                    "description": "Validate optimizations",
                    "duration": 300,
                    "dependencies": []
                }
            ]
        
        else:
            # Generic steps for unknown strategies
            steps = [
                {
                    "step_id": str(uuid.uuid4()),
                    "description": f"Execute {strategy}",
                    "duration": 300,
                    "dependencies": []
                }
            ]
        
        return steps
    
    def _estimate_adaptation_impact(self, strategies: List[str]) -> Dict[str, float]:
        """Estimate the impact of adaptation strategies."""
        impact_estimates = {
            "implement_caching": {"performance": 0.25, "resource_usage": -0.15},
            "increase_parallelism": {"throughput": 0.40, "response_time": -0.30},
            "optimize_algorithms": {"performance": 0.50, "resource_usage": -0.25},
            "scale_resources": {"capacity": 0.30, "cost": 0.20},
            "improve_error_handling": {"reliability": 0.35, "user_satisfaction": 0.20},
            "add_retry_mechanisms": {"success_rate": 0.15, "response_time": 0.10},
            "enhance_validation": {"error_rate": -0.30, "reliability": 0.25}
        }
        
        combined_impact = defaultdict(float)
        
        for strategy in strategies:
            strategy_impact = impact_estimates.get(strategy, {})
            for metric, impact in strategy_impact.items():
                combined_impact[metric] += impact
        
        return dict(combined_impact)
    
    def _estimate_adaptation_timeline(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate adaptation timeline."""
        total_duration = sum(step.get("duration", 0) for step in steps)
        
        return {
            "total_duration_seconds": total_duration,
            "total_duration_hours": total_duration / 3600,
            "number_of_steps": len(steps),
            "estimated_start": datetime.now(timezone.utc).isoformat(),
            "estimated_completion": (datetime.now(timezone.utc) + timedelta(seconds=total_duration)).isoformat()
        }


class LearningSystem:
    """Main learning system that integrates all learning components."""
    
    def __init__(self):
        self.user_preference_learning = UserPreferenceLearning()
        self.performance_optimization = PerformanceOptimization()
        self.adaptation_engine = AdaptationEngine()
        
        self.learning_history = deque(maxlen=2000)
        self.pattern_recognition = PatternRecognition()
        self.knowledge_base = {}
        self.learning_metrics = LearningMetrics()
        
        # Learning configuration
        self.learning_enabled = True
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7
        self.feedback_weight = 0.8
        
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base."""
        self.knowledge_base = {
            "patterns": {},
            "user_models": {},
            "performance_baselines": {},
            "adaptation_rules": {},
            "learning_insights": {}
        }
    
    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Main learning entry point for feedback."""
        try:
            if not self.learning_enabled:
                return {"learning_applied": False, "reason": "Learning disabled"}
            
            learning_result = {
                "learning_applied": False,
                "components_updated": [],
                "patterns_discovered": [],
                "adaptations_triggered": []
            }
            
            # Learn user preferences
            if feedback.get("feedback_type") in ["email_priority", "communication_style", "workflow_efficiency"]:
                pref_result = await self.user_preference_learning.learn_from_user_feedback(feedback)
                learning_result["preference_learning"] = pref_result
                learning_result["components_updated"].append("user_preferences")
                learning_result["learning_applied"] = True
            
            # Update performance models
            if "performance_data" in feedback:
                perf_result = await self.performance_optimization.analyze_performance_data([feedback["performance_data"]])
                learning_result["performance_analysis"] = perf_result
                learning_result["components_updated"].append("performance_optimization")
                learning_result["learning_applied"] = True
            
            # Record learning event
            learning_event = LearningEvent(
                event_type="feedback_learning",
                source="user_feedback",
                data=feedback,
                outcome=self._determine_learning_outcome(feedback),
                confidence=self._calculate_learning_confidence(feedback)
            )
            
            self.learning_history.append(learning_event)
            
            # Update learning metrics
            self.learning_metrics.total_learning_events += 1
            
            # Check for adaptation triggers
            if learning_result["learning_applied"]:
                adaptation_check = await self._check_adaptation_triggers(feedback)
                if adaptation_check["adaptation_needed"]:
                    learning_result["adaptations_triggered"] = adaptation_check["recommended_adaptations"]
            
            return learning_result
            
        except Exception as e:
            raise LearningError(f"Failed to learn from feedback: {str(e)}")
    
    async def recognize_patterns(self, data: List[Dict[str, Any]], pattern_type: str = "general") -> List[Dict[str, Any]]:
        """Recognize patterns in data using the pattern recognition component."""
        return await self.pattern_recognition.recognize_patterns(data, pattern_type)
    
    async def adapt_algorithms(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adapt algorithms based on performance data."""
        try:
            current_performance = performance_data.get("current_performance", {})
            target_performance = performance_data.get("target_performance", {})
            constraints = performance_data.get("constraints", {})
            
            # Evaluate adaptation need
            adaptation_evaluation = await self.adaptation_engine.evaluate_adaptation_need({
                "performance": current_performance,
                "baseline_performance": target_performance,
                "constraints": constraints
            })
            
            adaptations = []
            
            if adaptation_evaluation["adaptation_needed"]:
                # Create adaptation plan
                adaptation_config = {
                    "type": "performance_degradation",
                    "target_metrics": target_performance,
                    "constraints": constraints
                }
                
                adaptation_plan = await self.adaptation_engine.create_adaptation_plan(adaptation_config)
                
                # Convert plan to adaptation list
                for strategy in adaptation_plan["strategies"]:
                    adaptations.append({
                        "adaptation_type": strategy,
                        "expected_improvement": adaptation_plan["estimated_impact"].get("performance", 0.1),
                        "method": strategy,
                        "estimated_time": adaptation_plan["estimated_timeline"]["total_duration_seconds"],
                        "confidence": 0.8
                    })
            
            return adaptations
            
        except Exception as e:
            raise LearningError(f"Failed to adapt algorithms: {str(e)}")
    
    async def update_knowledge_base(self, new_knowledge: Dict[str, Any]) -> bool:
        """Update the knowledge base with new insights."""
        try:
            domain = new_knowledge.get("domain", "general")
            insights = new_knowledge.get("insights", [])
            
            if domain not in self.knowledge_base:
                self.knowledge_base[domain] = {}
            
            # Process each insight
            for insight in insights:
                insight_type = insight.get("type", "unknown")
                insight_data = insight.get("data", {})
                confidence = insight.get("confidence", 0.5)
                
                # Store insight with metadata
                insight_key = f"{insight_type}_{int(time.time())}"
                self.knowledge_base[domain][insight_key] = {
                    "type": insight_type,
                    "data": insight_data,
                    "confidence": confidence,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "usage_count": 0,
                    "validation_score": 0.0
                }
            
            # Update knowledge base metrics
            self.learning_metrics.knowledge_base_size = sum(
                len(domain_data) for domain_data in self.knowledge_base.values()
            )
            
            return True
            
        except Exception as e:
            raise LearningError(f"Failed to update knowledge base: {str(e)}")
    
    async def get_learning_metrics(self) -> LearningMetrics:
        """Get current learning metrics."""
        # Update metrics with current state
        if self.learning_history:
            recent_events = [event for event in self.learning_history 
                           if (datetime.now(timezone.utc) - event.timestamp).days <= 7]
            
            success_events = [event for event in recent_events if event.outcome == "success"]
            self.learning_metrics.adaptation_success_rate = len(success_events) / len(recent_events) if recent_events else 0.0
            
            # Calculate accuracy improvement (simplified)
            confidence_scores = [event.confidence for event in recent_events if event.confidence > 0]
            if confidence_scores:
                self.learning_metrics.pattern_recognition_accuracy = sum(confidence_scores) / len(confidence_scores)
        
        return self.learning_metrics
    
    async def predict_optimal_behavior(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal behavior for a given context."""
        try:
            prediction_result = {
                "recommendations": [],
                "confidence": 0.0,
                "reasoning": [],
                "context_factors": []
            }
            
            # Get user preference predictions
            if "preference_type" in context:
                pref_prediction = await self.user_preference_learning.predict_user_preference(
                    context, context["preference_type"]
                )
                prediction_result["user_preference"] = pref_prediction
                prediction_result["confidence"] = max(prediction_result["confidence"], pref_prediction["confidence"])
            
            # Analyze performance patterns
            if "performance_context" in context:
                perf_data = context["performance_context"]
                patterns = await self.recognize_patterns([perf_data], "performance")
                
                if patterns:
                    prediction_result["performance_patterns"] = patterns
                    pattern_confidence = max(p["confidence"] for p in patterns)
                    prediction_result["confidence"] = max(prediction_result["confidence"], pattern_confidence)
            
            # Generate recommendations based on knowledge base
            domain = context.get("domain", "general")
            if domain in self.knowledge_base:
                domain_knowledge = self.knowledge_base[domain]
                relevant_insights = [
                    insight for insight in domain_knowledge.values()
                    if insight["confidence"] > 0.6
                ]
                
                for insight in relevant_insights[:3]:  # Top 3 insights
                    prediction_result["recommendations"].append({
                        "type": insight["type"],
                        "confidence": insight["confidence"],
                        "recommendation": f"Apply {insight['type']} based on learned patterns"
                    })
            
            # Calculate overall confidence
            if prediction_result["recommendations"]:
                avg_confidence = sum(r["confidence"] for r in prediction_result["recommendations"]) / len(prediction_result["recommendations"])
                prediction_result["confidence"] = max(prediction_result["confidence"], avg_confidence)
            
            return prediction_result
            
        except Exception as e:
            raise LearningError(f"Failed to predict optimal behavior: {str(e)}")
    
    def _determine_learning_outcome(self, feedback: Dict[str, Any]) -> str:
        """Determine the outcome of a learning event."""
        user_rating = feedback.get("user_rating", 3)
        feedback_type = feedback.get("feedback_type", "neutral")
        
        if feedback_type == "positive" or user_rating >= 4:
            return "success"
        elif feedback_type == "negative" or user_rating <= 2:
            return "failure"
        else:
            return "partial"
    
    def _calculate_learning_confidence(self, feedback: Dict[str, Any]) -> float:
        """Calculate confidence in the learning from feedback."""
        factors = []
        
        # User rating confidence
        user_rating = feedback.get("user_rating", 3)
        rating_confidence = abs(user_rating - 3) / 2  # Distance from neutral
        factors.append(rating_confidence)
        
        # Context richness confidence
        context = feedback.get("context", {})
        context_confidence = min(len(context) / 10, 1.0)  # More context = higher confidence
        factors.append(context_confidence)
        
        # Feedback type confidence
        feedback_type = feedback.get("feedback_type", "general")
        type_confidence = 0.8 if feedback_type in ["email_priority", "communication_style"] else 0.5
        factors.append(type_confidence)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    async def _check_adaptation_triggers(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Check if feedback triggers any system adaptations."""
        adaptation_needed = False
        recommended_adaptations = []
        
        # Check for repeated negative feedback
        recent_negative = [
            event for event in self.learning_history[-10:]  # Last 10 events
            if event.outcome == "failure"
        ]
        
        if len(recent_negative) >= 3:  # 3 or more failures in last 10 events
            adaptation_needed = True
            recommended_adaptations.append("improve_decision_accuracy")
        
        # Check for performance degradation signals
        user_rating = feedback.get("user_rating", 3)
        if user_rating <= 2:
            performance_data = feedback.get("performance_data", {})
            if performance_data.get("response_time", 0) > 5:  # Slow response
                adaptation_needed = True
                recommended_adaptations.append("optimize_response_time")
        
        return {
            "adaptation_needed": adaptation_needed,
            "recommended_adaptations": recommended_adaptations,
            "trigger_reason": "Repeated negative feedback" if recent_negative else "Performance issues"
        }
    
    async def enable_learning(self, enabled: bool = True) -> None:
        """Enable or disable learning."""
        self.learning_enabled = enabled
    
    async def set_learning_rate(self, learning_rate: float) -> None:
        """Set the learning rate."""
        self.learning_rate = max(0.01, min(learning_rate, 1.0))  # Clamp to valid range
        self.user_preference_learning.learning_rate = self.learning_rate
    
    async def get_system_insights(self) -> Dict[str, Any]:
        """Get insights about the learning system."""
        total_events = len(self.learning_history)
        successful_events = len([e for e in self.learning_history if e.outcome == "success"])
        
        return {
            "total_learning_events": total_events,
            "success_rate": successful_events / total_events if total_events > 0 else 0.0,
            "knowledge_domains": list(self.knowledge_base.keys()),
            "knowledge_base_size": self.learning_metrics.knowledge_base_size,
            "learning_enabled": self.learning_enabled,
            "learning_rate": self.learning_rate,
            "adaptation_threshold": self.adaptation_threshold,
            "recent_patterns": await self._get_recent_patterns(),
            "system_health": self._assess_learning_system_health()
        }
    
    async def _get_recent_patterns(self) -> List[Dict[str, Any]]:
        """Get recently discovered patterns."""
        # Extract recent learning events and identify patterns
        recent_events = [event for event in self.learning_history[-50:]]  # Last 50 events
        
        if not recent_events:
            return []
        
        patterns = await self.recognize_patterns([event.to_dict() for event in recent_events])
        return patterns[:5]  # Return top 5 patterns
    
    def _assess_learning_system_health(self) -> Dict[str, Any]:
        """Assess the health of the learning system."""
        health_score = 0.0
        health_factors = []
        
        # Check learning activity
        recent_events = len([e for e in self.learning_history if (datetime.now(timezone.utc) - e.timestamp).days <= 1])
        if recent_events > 0:
            health_score += 0.3
            health_factors.append("Active learning")
        
        # Check knowledge base growth
        if self.learning_metrics.knowledge_base_size > 10:
            health_score += 0.3
            health_factors.append("Growing knowledge base")
        
        # Check adaptation success rate
        if self.learning_metrics.adaptation_success_rate > 0.7:
            health_score += 0.4
            health_factors.append("Successful adaptations")
        
        health_status = "excellent" if health_score > 0.8 else "good" if health_score > 0.6 else "fair" if health_score > 0.4 else "poor"
        
        return {
            "health_score": health_score,
            "health_status": health_status,
            "contributing_factors": health_factors,
            "recommendations": self._generate_health_recommendations(health_score)
        }
    
    def _generate_health_recommendations(self, health_score: float) -> List[str]:
        """Generate recommendations for improving learning system health."""
        recommendations = []
        
        if health_score < 0.4:
            recommendations.append("Increase user feedback collection")
            recommendations.append("Enable more learning opportunities")
        
        if health_score < 0.6:
            recommendations.append("Review adaptation strategies")
            recommendations.append("Expand knowledge base domains")
        
        if health_score < 0.8:
            recommendations.append("Fine-tune learning parameters")
            recommendations.append("Implement advanced pattern recognition")
        
        return recommendations


# Import PatternRecognition class from intelligence_engine if it exists
try:
    from .intelligence_engine import PatternRecognition
except ImportError:
    # Fallback implementation if not available
    class PatternRecognition:
        """Fallback pattern recognition implementation."""
        
        async def recognize_patterns(self, data: List[Dict[str, Any]], pattern_type: str = "general") -> List[Dict[str, Any]]:
            """Simple pattern recognition fallback."""
            return [
                {
                    "pattern_type": "learning_activity",
                    "confidence": 0.7,
                    "description": "Learning system is active",
                    "data": {"events_analyzed": len(data)}
                }
            ]