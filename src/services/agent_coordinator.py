"""
Agent Coordinator component for the Intelligence Layer.

This module provides multi-agent communication, task distribution,
conflict resolution, and workload balancing capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from .intelligence_engine import (
    CoordinationResult, AgentWorkload, CoordinationError,
    IntelligenceError
)
from ..agents.base import AgentMessage, AgentState


class CoordinationType(Enum):
    """Types of agent coordination."""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    HYBRID = "hybrid"
    COMPETITIVE = "competitive"
    COLLABORATIVE = "collaborative"


class ConflictType(Enum):
    """Types of conflicts that can occur."""
    RESOURCE_CONFLICT = "resource_conflict"
    TIMING_CONFLICT = "timing_conflict"
    PRIORITY_CONFLICT = "priority_conflict"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    CAPACITY_CONFLICT = "capacity_conflict"


@dataclass
class AgentCapability:
    """Represents an agent's capabilities."""
    agent_id: str
    capabilities: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    performance_rating: float = 0.8
    availability: float = 1.0
    current_load: float = 0.0
    max_capacity: int = 10
    response_time_avg: float = 30.0  # seconds
    success_rate: float = 0.9
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "capabilities": self.capabilities,
            "specializations": self.specializations,
            "performance_rating": self.performance_rating,
            "availability": self.availability,
            "current_load": self.current_load,
            "max_capacity": self.max_capacity,
            "response_time_avg": self.response_time_avg,
            "success_rate": self.success_rate
        }


@dataclass
class Conflict:
    """Represents a conflict between agents or resources."""
    conflict_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type: ConflictType = ConflictType.RESOURCE_CONFLICT
    involved_agents: List[str] = field(default_factory=list)
    resource_name: Optional[str] = None
    priority_levels: Dict[str, int] = field(default_factory=dict)
    severity: str = "medium"  # low, medium, high, critical
    description: str = ""
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolution_strategy: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "involved_agents": self.involved_agents,
            "resource_name": self.resource_name,
            "priority_levels": self.priority_levels,
            "severity": self.severity,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolution_strategy": self.resolution_strategy
        }


@dataclass
class CoordinationTask:
    """Represents a coordination task."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    coordination_type: CoordinationType = CoordinationType.PARALLEL
    involved_agents: List[str] = field(default_factory=list)
    agent_assignments: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    timeout_seconds: int = 300
    priority: int = 5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed, timeout
    results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "coordination_type": self.coordination_type.value,
            "involved_agents": self.involved_agents,
            "agent_assignments": self.agent_assignments,
            "dependencies": self.dependencies,
            "timeout_seconds": self.timeout_seconds,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "results": self.results
        }


class ConflictResolver:
    """Resolves conflicts between agents and resources."""
    
    def __init__(self):
        self.resolution_strategies = {
            ConflictType.RESOURCE_CONFLICT: [
                "priority_based",
                "round_robin",
                "resource_scaling",
                "queue_management"
            ],
            ConflictType.TIMING_CONFLICT: [
                "reschedule",
                "parallel_execution",
                "priority_override",
                "deadline_extension"
            ],
            ConflictType.PRIORITY_CONFLICT: [
                "priority_negotiation",
                "escalation",
                "resource_reallocation",
                "task_splitting"
            ],
            ConflictType.DEPENDENCY_CONFLICT: [
                "dependency_reordering",
                "parallel_processing",
                "dependency_breaking",
                "alternative_path"
            ],
            ConflictType.CAPACITY_CONFLICT: [
                "load_balancing",
                "capacity_scaling",
                "task_deferral",
                "agent_recruitment"
            ]
        }
        
        self.resolution_history = deque(maxlen=1000)
        self.conflict_patterns = {}
    
    async def detect_conflicts(self, coordination_tasks: List[CoordinationTask], agent_capabilities: Dict[str, AgentCapability]) -> List[Conflict]:
        """Detect conflicts in coordination tasks."""
        conflicts = []
        
        # Resource conflicts
        resource_usage = defaultdict(list)
        for task in coordination_tasks:
            for agent_id in task.involved_agents:
                resource_usage[agent_id].append(task)
        
        for agent_id, tasks in resource_usage.items():
            if len(tasks) > 1:
                # Check for resource conflicts
                capability = agent_capabilities.get(agent_id)
                if capability and len(tasks) > capability.max_capacity:
                    conflict = Conflict(
                        conflict_type=ConflictType.CAPACITY_CONFLICT,
                        involved_agents=[agent_id],
                        resource_name=agent_id,
                        priority_levels={task.task_id: task.priority for task in tasks},
                        severity="high" if len(tasks) > capability.max_capacity * 1.5 else "medium",
                        description=f"Agent {agent_id} overloaded with {len(tasks)} tasks (capacity: {capability.max_capacity})"
                    )
                    conflicts.append(conflict)
        
        # Timing conflicts
        time_conflicts = await self._detect_timing_conflicts(coordination_tasks)
        conflicts.extend(time_conflicts)
        
        # Priority conflicts
        priority_conflicts = await self._detect_priority_conflicts(coordination_tasks)
        conflicts.extend(priority_conflicts)
        
        # Dependency conflicts
        dependency_conflicts = await self._detect_dependency_conflicts(coordination_tasks)
        conflicts.extend(dependency_conflicts)
        
        return conflicts
    
    async def _detect_timing_conflicts(self, tasks: List[CoordinationTask]) -> List[Conflict]:
        """Detect timing conflicts between tasks."""
        conflicts = []
        
        # Simple timing conflict detection based on agent overlap
        agent_schedules = defaultdict(list)
        
        for task in tasks:
            for agent_id in task.involved_agents:
                agent_schedules[agent_id].append({
                    "task_id": task.task_id,
                    "start_time": task.started_at or task.created_at,
                    "estimated_duration": task.timeout_seconds,
                    "priority": task.priority
                })
        
        for agent_id, schedule in agent_schedules.items():
            if len(schedule) > 1:
                # Sort by start time
                schedule.sort(key=lambda x: x["start_time"])
                
                for i in range(len(schedule) - 1):
                    current_task = schedule[i]
                    next_task = schedule[i + 1]
                    
                    current_end = current_task["start_time"] + timedelta(seconds=current_task["estimated_duration"])
                    next_start = next_task["start_time"]
                    
                    if current_end > next_start:
                        # Timing conflict detected
                        conflict = Conflict(
                            conflict_type=ConflictType.TIMING_CONFLICT,
                            involved_agents=[agent_id],
                            priority_levels={
                                current_task["task_id"]: current_task["priority"],
                                next_task["task_id"]: next_task["priority"]
                            },
                            severity="medium",
                            description=f"Timing conflict for agent {agent_id} between tasks {current_task['task_id']} and {next_task['task_id']}"
                        )
                        conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_priority_conflicts(self, tasks: List[CoordinationTask]) -> List[Conflict]:
        """Detect priority conflicts between tasks."""
        conflicts = []
        
        # Group tasks by shared agents
        shared_agent_tasks = defaultdict(list)
        for task in tasks:
            for agent_id in task.involved_agents:
                shared_agent_tasks[agent_id].append(task)
        
        for agent_id, agent_tasks in shared_agent_tasks.items():
            if len(agent_tasks) > 1:
                # Check for priority conflicts
                priority_groups = defaultdict(list)
                for task in agent_tasks:
                    priority_groups[task.priority].append(task)
                
                # If multiple high-priority tasks compete for the same agent
                high_priority_tasks = [task for task in agent_tasks if task.priority >= 8]
                if len(high_priority_tasks) > 1:
                    conflict = Conflict(
                        conflict_type=ConflictType.PRIORITY_CONFLICT,
                        involved_agents=[agent_id],
                        priority_levels={task.task_id: task.priority for task in high_priority_tasks},
                        severity="high",
                        description=f"Multiple high-priority tasks competing for agent {agent_id}"
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_dependency_conflicts(self, tasks: List[CoordinationTask]) -> List[Conflict]:
        """Detect dependency conflicts between tasks."""
        conflicts = []
        
        # Check for circular dependencies
        task_deps = {task.task_id: list(task.dependencies.keys()) for task in tasks}
        
        def has_circular_dependency(task_id: str, visited: Set[str], path: Set[str]) -> bool:
            if task_id in path:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            path.add(task_id)
            
            for dep in task_deps.get(task_id, []):
                if has_circular_dependency(dep, visited, path):
                    return True
            
            path.remove(task_id)
            return False
        
        visited = set()
        for task_id in task_deps:
            if task_id not in visited:
                if has_circular_dependency(task_id, visited, set()):
                    # Find the tasks involved in the circular dependency
                    involved_tasks = [task_id]
                    # Simple approach: add all dependent tasks
                    involved_tasks.extend(task_deps.get(task_id, []))
                    
                    conflict = Conflict(
                        conflict_type=ConflictType.DEPENDENCY_CONFLICT,
                        involved_agents=[],  # Will be filled based on tasks
                        severity="high",
                        description=f"Circular dependency detected involving tasks: {involved_tasks}"
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    async def resolve_conflict(self, conflict: Conflict, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Resolve a specific conflict."""
        try:
            context = context or {}
            available_strategies = self.resolution_strategies.get(conflict.conflict_type, ["default"])
            
            # Select best strategy based on conflict and context
            selected_strategy = await self._select_resolution_strategy(conflict, available_strategies, context)
            
            # Apply resolution strategy
            resolution_result = await self._apply_resolution_strategy(conflict, selected_strategy, context)
            
            # Record resolution
            resolution_record = {
                "conflict_id": conflict.conflict_id,
                "strategy": selected_strategy,
                "success": resolution_result.get("success", False),
                "resolution_time": resolution_result.get("resolution_time", 0),
                "resolved_at": datetime.now(timezone.utc).isoformat()
            }
            
            self.resolution_history.append(resolution_record)
            
            # Update conflict status
            if resolution_result.get("success"):
                conflict.resolved = True
                conflict.resolution_strategy = selected_strategy
            
            return {
                "conflict_id": conflict.conflict_id,
                "resolution_type": selected_strategy,
                "success": resolution_result.get("success", False),
                "affected_agents": conflict.involved_agents,
                "resolution_time": resolution_result.get("resolution_time", 0),
                "details": resolution_result
            }
            
        except Exception as e:
            raise CoordinationError(f"Failed to resolve conflict {conflict.conflict_id}: {str(e)}")
    
    async def _select_resolution_strategy(self, conflict: Conflict, available_strategies: List[str], context: Dict[str, Any]) -> str:
        """Select the best resolution strategy for a conflict."""
        # Strategy selection based on conflict characteristics and context
        
        if conflict.severity == "critical":
            # For critical conflicts, use most direct resolution
            priority_strategies = {
                ConflictType.RESOURCE_CONFLICT: "resource_scaling",
                ConflictType.TIMING_CONFLICT: "priority_override",
                ConflictType.PRIORITY_CONFLICT: "escalation",
                ConflictType.DEPENDENCY_CONFLICT: "dependency_breaking",
                ConflictType.CAPACITY_CONFLICT: "capacity_scaling"
            }
            preferred = priority_strategies.get(conflict.conflict_type)
            if preferred in available_strategies:
                return preferred
        
        # Check historical success rates
        best_strategy = available_strategies[0]
        best_success_rate = 0.0
        
        for strategy in available_strategies:
            success_rate = self._get_strategy_success_rate(conflict.conflict_type, strategy)
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_strategy = strategy
        
        return best_strategy
    
    def _get_strategy_success_rate(self, conflict_type: ConflictType, strategy: str) -> float:
        """Get historical success rate for a strategy."""
        relevant_resolutions = [
            record for record in self.resolution_history
            if record["strategy"] == strategy
        ]
        
        if not relevant_resolutions:
            return 0.5  # Default success rate
        
        successful = sum(1 for record in relevant_resolutions if record["success"])
        return successful / len(relevant_resolutions)
    
    async def _apply_resolution_strategy(self, conflict: Conflict, strategy: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a specific resolution strategy."""
        start_time = time.time()
        
        try:
            if strategy == "priority_based":
                result = await self._resolve_priority_based(conflict, context)
            elif strategy == "round_robin":
                result = await self._resolve_round_robin(conflict, context)
            elif strategy == "resource_scaling":
                result = await self._resolve_resource_scaling(conflict, context)
            elif strategy == "reschedule":
                result = await self._resolve_reschedule(conflict, context)
            elif strategy == "parallel_execution":
                result = await self._resolve_parallel_execution(conflict, context)
            elif strategy == "load_balancing":
                result = await self._resolve_load_balancing(conflict, context)
            else:
                result = await self._resolve_default(conflict, context)
            
            resolution_time = time.time() - start_time
            result["resolution_time"] = resolution_time
            
            return result
            
        except Exception as e:
            resolution_time = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "resolution_time": resolution_time
            }
    
    async def _resolve_priority_based(self, conflict: Conflict, context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict using priority-based approach."""
        if conflict.priority_levels:
            # Sort by priority (higher number = higher priority)
            sorted_tasks = sorted(conflict.priority_levels.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "success": True,
                "resolution": "priority_ordering",
                "task_order": [task_id for task_id, _ in sorted_tasks],
                "highest_priority_task": sorted_tasks[0][0] if sorted_tasks else None
            }
        
        return {"success": False, "reason": "No priority levels available"}
    
    async def _resolve_round_robin(self, conflict: Conflict, context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict using round-robin approach."""
        if conflict.involved_agents:
            # Distribute tasks evenly among agents
            return {
                "success": True,
                "resolution": "round_robin_distribution",
                "distribution": "Even distribution among agents",
                "agents": conflict.involved_agents
            }
        
        return {"success": False, "reason": "No agents involved"}
    
    async def _resolve_resource_scaling(self, conflict: Conflict, context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict by scaling resources."""
        return {
            "success": True,
            "resolution": "resource_scaling",
            "action": "Scaled up resources to handle conflict",
            "resource": conflict.resource_name
        }
    
    async def _resolve_reschedule(self, conflict: Conflict, context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict by rescheduling tasks."""
        return {
            "success": True,
            "resolution": "rescheduling",
            "action": "Rescheduled conflicting tasks",
            "affected_tasks": list(conflict.priority_levels.keys()) if conflict.priority_levels else []
        }
    
    async def _resolve_parallel_execution(self, conflict: Conflict, context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict by enabling parallel execution."""
        return {
            "success": True,
            "resolution": "parallel_execution",
            "action": "Enabled parallel execution to resolve conflict",
            "parallelized_agents": conflict.involved_agents
        }
    
    async def _resolve_load_balancing(self, conflict: Conflict, context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict through load balancing."""
        return {
            "success": True,
            "resolution": "load_balancing",
            "action": "Redistributed load among agents",
            "balanced_agents": conflict.involved_agents
        }
    
    async def _resolve_default(self, conflict: Conflict, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default conflict resolution."""
        return {
            "success": True,
            "resolution": "default",
            "action": "Applied default conflict resolution",
            "note": "Used fallback resolution strategy"
        }


class WorkloadBalancer:
    """Balances workload across agents."""
    
    def __init__(self):
        self.agent_workloads = {}
        self.load_history = deque(maxlen=1000)
        self.balancing_strategies = [
            "least_loaded",
            "round_robin",
            "capability_based",
            "performance_weighted",
            "predictive"
        ]
        
        self.load_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.3
        }
    
    async def calculate_agent_workloads(self, agents: Dict[str, AgentCapability], active_tasks: List[CoordinationTask]) -> Dict[str, AgentWorkload]:
        """Calculate current workload for each agent."""
        workloads = {}
        
        for agent_id, capability in agents.items():
            # Count tasks assigned to this agent
            agent_tasks = [task for task in active_tasks if agent_id in task.involved_agents]
            current_tasks = len(agent_tasks)
            
            # Calculate utilization
            utilization = current_tasks / capability.max_capacity if capability.max_capacity > 0 else 1.0
            
            workload = AgentWorkload(
                agent_id=agent_id,
                current_tasks=current_tasks,
                capacity=capability.max_capacity,
                utilization=utilization,
                performance_score=capability.performance_rating
            )
            
            workloads[agent_id] = workload
            self.agent_workloads[agent_id] = workload
        
        return workloads
    
    async def balance_workload(self, workloads: Dict[str, AgentWorkload], new_task: Dict[str, Any], strategy: str = "least_loaded") -> Dict[str, Any]:
        """Balance workload by assigning a new task to the optimal agent."""
        try:
            if not workloads:
                return {"success": False, "reason": "No agents available"}
            
            if strategy == "least_loaded":
                assignment = await self._assign_least_loaded(workloads, new_task)
            elif strategy == "round_robin":
                assignment = await self._assign_round_robin(workloads, new_task)
            elif strategy == "capability_based":
                assignment = await self._assign_capability_based(workloads, new_task)
            elif strategy == "performance_weighted":
                assignment = await self._assign_performance_weighted(workloads, new_task)
            elif strategy == "predictive":
                assignment = await self._assign_predictive(workloads, new_task)
            else:
                assignment = await self._assign_least_loaded(workloads, new_task)
            
            # Update workload after assignment
            if assignment.get("success") and assignment.get("assigned_agent"):
                agent_id = assignment["assigned_agent"]
                if agent_id in workloads:
                    workloads[agent_id].current_tasks += 1
                    workloads[agent_id].utilization = workloads[agent_id].current_tasks / workloads[agent_id].capacity
            
            return assignment
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _assign_least_loaded(self, workloads: Dict[str, AgentWorkload], new_task: Dict[str, Any]) -> Dict[str, Any]:
        """Assign task to the least loaded agent."""
        available_agents = [
            (agent_id, workload) for agent_id, workload in workloads.items()
            if workload.utilization < 1.0  # Not at full capacity
        ]
        
        if not available_agents:
            return {"success": False, "reason": "All agents at full capacity"}
        
        # Find agent with lowest utilization
        best_agent_id, best_workload = min(available_agents, key=lambda x: x[1].utilization)
        
        return {
            "success": True,
            "assigned_agent": best_agent_id,
            "load_balance_score": 1.0 - best_workload.utilization,
            "expected_completion_time": self._estimate_completion_time(best_workload, new_task),
            "utilization_after_assignment": (best_workload.current_tasks + 1) / best_workload.capacity,
            "assignment_reason": "Least loaded agent"
        }
    
    async def _assign_round_robin(self, workloads: Dict[str, AgentWorkload], new_task: Dict[str, Any]) -> Dict[str, Any]:
        """Assign task using round-robin strategy."""
        available_agents = [
            agent_id for agent_id, workload in workloads.items()
            if workload.utilization < 1.0
        ]
        
        if not available_agents:
            return {"success": False, "reason": "All agents at full capacity"}
        
        # Simple round-robin selection
        selected_agent = available_agents[len(self.load_history) % len(available_agents)]
        selected_workload = workloads[selected_agent]
        
        return {
            "success": True,
            "assigned_agent": selected_agent,
            "load_balance_score": 0.8,  # Fixed score for round-robin
            "expected_completion_time": self._estimate_completion_time(selected_workload, new_task),
            "utilization_after_assignment": (selected_workload.current_tasks + 1) / selected_workload.capacity,
            "assignment_reason": "Round-robin distribution"
        }
    
    async def _assign_capability_based(self, workloads: Dict[str, AgentWorkload], new_task: Dict[str, Any]) -> Dict[str, Any]:
        """Assign task based on agent capabilities."""
        task_requirements = new_task.get("requirements", [])
        
        # Score agents based on capability match
        scored_agents = []
        for agent_id, workload in workloads.items():
            if workload.utilization >= 1.0:
                continue  # Skip overloaded agents
            
            # Calculate capability match score (simplified)
            capability_score = 0.8  # Default score
            if task_requirements:
                # In a real implementation, this would match against agent capabilities
                capability_score = 0.9 if "email" in str(task_requirements).lower() and "gmail" in agent_id else 0.7
            
            combined_score = capability_score * (1.0 - workload.utilization)
            scored_agents.append((agent_id, workload, combined_score))
        
        if not scored_agents:
            return {"success": False, "reason": "No suitable agents available"}
        
        # Select agent with highest combined score
        best_agent_id, best_workload, best_score = max(scored_agents, key=lambda x: x[2])
        
        return {
            "success": True,
            "assigned_agent": best_agent_id,
            "load_balance_score": best_score,
            "expected_completion_time": self._estimate_completion_time(best_workload, new_task),
            "utilization_after_assignment": (best_workload.current_tasks + 1) / best_workload.capacity,
            "assignment_reason": "Capability-based matching"
        }
    
    async def _assign_performance_weighted(self, workloads: Dict[str, AgentWorkload], new_task: Dict[str, Any]) -> Dict[str, Any]:
        """Assign task based on performance-weighted scoring."""
        available_agents = [
            (agent_id, workload) for agent_id, workload in workloads.items()
            if workload.utilization < 1.0
        ]
        
        if not available_agents:
            return {"success": False, "reason": "All agents at full capacity"}
        
        # Calculate performance-weighted scores
        scored_agents = []
        for agent_id, workload in available_agents:
            # Combine performance rating with inverse utilization
            performance_score = workload.performance_score * (1.0 - workload.utilization)
            scored_agents.append((agent_id, workload, performance_score))
        
        # Select agent with highest performance-weighted score
        best_agent_id, best_workload, best_score = max(scored_agents, key=lambda x: x[2])
        
        return {
            "success": True,
            "assigned_agent": best_agent_id,
            "load_balance_score": best_score,
            "expected_completion_time": self._estimate_completion_time(best_workload, new_task),
            "utilization_after_assignment": (best_workload.current_tasks + 1) / best_workload.capacity,
            "assignment_reason": "Performance-weighted assignment"
        }
    
    async def _assign_predictive(self, workloads: Dict[str, AgentWorkload], new_task: Dict[str, Any]) -> Dict[str, Any]:
        """Assign task using predictive load balancing."""
        # Predict future load based on historical patterns
        predicted_loads = {}
        
        for agent_id, workload in workloads.items():
            if workload.utilization >= 1.0:
                continue
            
            # Simple prediction: current load + trend
            historical_loads = [
                record.get("utilization", 0.5) for record in self.load_history[-10:]
                if record.get("agent_id") == agent_id
            ]
            
            if historical_loads:
                trend = (historical_loads[-1] - historical_loads[0]) / len(historical_loads)
                predicted_load = workload.utilization + trend
            else:
                predicted_load = workload.utilization
            
            predicted_loads[agent_id] = predicted_load
        
        if not predicted_loads:
            return {"success": False, "reason": "No agents available for prediction"}
        
        # Select agent with lowest predicted load
        best_agent_id = min(predicted_loads, key=predicted_loads.get)
        best_workload = workloads[best_agent_id]
        
        return {
            "success": True,
            "assigned_agent": best_agent_id,
            "load_balance_score": 1.0 - predicted_loads[best_agent_id],
            "expected_completion_time": self._estimate_completion_time(best_workload, new_task),
            "utilization_after_assignment": (best_workload.current_tasks + 1) / best_workload.capacity,
            "assignment_reason": "Predictive load balancing",
            "predicted_load": predicted_loads[best_agent_id]
        }
    
    def _estimate_completion_time(self, workload: AgentWorkload, new_task: Dict[str, Any]) -> int:
        """Estimate task completion time."""
        base_time = new_task.get("estimated_duration", 300)  # 5 minutes default
        
        # Adjust based on current load
        load_factor = 1.0 + (workload.utilization * 0.5)  # Up to 50% increase
        
        # Adjust based on performance
        performance_factor = 2.0 - workload.performance_score  # Better performance = faster
        
        estimated_time = int(base_time * load_factor * performance_factor)
        return estimated_time
    
    async def get_load_distribution_report(self) -> Dict[str, Any]:
        """Get a report on current load distribution."""
        if not self.agent_workloads:
            return {"error": "No workload data available"}
        
        total_capacity = sum(w.capacity for w in self.agent_workloads.values())
        total_current = sum(w.current_tasks for w in self.agent_workloads.values())
        
        utilizations = [w.utilization for w in self.agent_workloads.values()]
        avg_utilization = sum(utilizations) / len(utilizations)
        max_utilization = max(utilizations)
        min_utilization = min(utilizations)
        
        # Calculate load imbalance
        load_variance = np.var(utilizations) if len(utilizations) > 1 else 0
        
        overloaded_agents = [
            agent_id for agent_id, workload in self.agent_workloads.items()
            if workload.utilization > self.load_thresholds["high"]
        ]
        
        underutilized_agents = [
            agent_id for agent_id, workload in self.agent_workloads.items()
            if workload.utilization < self.load_thresholds["low"]
        ]
        
        return {
            "total_capacity": total_capacity,
            "total_current_tasks": total_current,
            "overall_utilization": total_current / total_capacity if total_capacity > 0 else 0,
            "average_utilization": avg_utilization,
            "max_utilization": max_utilization,
            "min_utilization": min_utilization,
            "load_variance": load_variance,
            "load_balance_score": 1.0 - load_variance,  # Lower variance = better balance
            "overloaded_agents": overloaded_agents,
            "underutilized_agents": underutilized_agents,
            "recommendations": self._generate_balancing_recommendations(
                overloaded_agents, underutilized_agents, load_variance
            ),
            "report_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _generate_balancing_recommendations(self, overloaded: List[str], underutilized: List[str], variance: float) -> List[str]:
        """Generate load balancing recommendations."""
        recommendations = []
        
        if overloaded:
            recommendations.append(f"Redistribute tasks from overloaded agents: {', '.join(overloaded)}")
        
        if underutilized:
            recommendations.append(f"Increase task assignment to underutilized agents: {', '.join(underutilized)}")
        
        if variance > 0.2:
            recommendations.append("High load variance detected - implement active load balancing")
        
        if not overloaded and not underutilized and variance < 0.1:
            recommendations.append("Load distribution is well balanced")
        
        return recommendations


class AgentCoordinator:
    """Main agent coordination component."""
    
    def __init__(self, message_broker):
        self.message_broker = message_broker
        self.conflict_resolver = ConflictResolver()
        self.workload_balancer = WorkloadBalancer()
        
        self.active_coordinations = {}
        self.agent_capabilities = {}
        self.coordination_history = deque(maxlen=500)
        
        # Coordination metrics
        self.total_coordinations = 0
        self.successful_coordinations = 0
        self.failed_coordinations = 0
        self.average_coordination_time = 0.0
        
        self._initialize_default_capabilities()
    
    def _initialize_default_capabilities(self):
        """Initialize default agent capabilities."""
        default_agents = [
            {
                "agent_id": "gmail_agent",
                "capabilities": ["email_processing", "communication", "scheduling"],
                "specializations": ["email_triage", "email_sending", "calendar_management"],
                "max_capacity": 15
            },
            {
                "agent_id": "research_agent", 
                "capabilities": ["research", "analysis", "data_gathering"],
                "specializations": ["web_search", "content_analysis", "report_generation"],
                "max_capacity": 10
            },
            {
                "agent_id": "code_agent",
                "capabilities": ["code_analysis", "development", "review"],
                "specializations": ["code_review", "debugging", "documentation"],
                "max_capacity": 8
            }
        ]
        
        for agent_config in default_agents:
            capability = AgentCapability(
                agent_id=agent_config["agent_id"],
                capabilities=agent_config["capabilities"],
                specializations=agent_config["specializations"],
                max_capacity=agent_config["max_capacity"]
            )
            self.agent_capabilities[agent_config["agent_id"]] = capability
    
    async def coordinate_agents(self, coordination_request: Dict[str, Any]) -> CoordinationResult:
        """Coordinate multiple agents to complete a task."""
        try:
            start_time = time.time()
            
            # Create coordination task
            coordination_task = CoordinationTask(
                task_id=coordination_request.get("task_id", str(uuid.uuid4())),
                coordination_type=CoordinationType(coordination_request.get("coordination_type", "parallel")),
                involved_agents=coordination_request.get("agents", []),
                agent_assignments=coordination_request.get("agent_assignments", {}),
                dependencies=coordination_request.get("dependencies", {}),
                timeout_seconds=coordination_request.get("timeout", 300),
                priority=coordination_request.get("priority", 5)
            )
            
            # Store coordination task
            self.active_coordinations[coordination_task.task_id] = coordination_task
            coordination_task.started_at = datetime.now(timezone.utc)
            coordination_task.status = "running"
            
            # Detect and resolve conflicts
            conflicts = await self.conflict_resolver.detect_conflicts(
                [coordination_task], self.agent_capabilities
            )
            
            conflicts_resolved = []
            if conflicts:
                for conflict in conflicts:
                    resolution = await self.conflict_resolver.resolve_conflict(conflict, coordination_request)
                    conflicts_resolved.append(resolution)
            
            # Execute coordination based on type
            if coordination_task.coordination_type == CoordinationType.PARALLEL:
                result = await self._execute_parallel_coordination(coordination_task)
            elif coordination_task.coordination_type == CoordinationType.SEQUENTIAL:
                result = await self._execute_sequential_coordination(coordination_task)
            elif coordination_task.coordination_type == CoordinationType.HYBRID:
                result = await self._execute_hybrid_coordination(coordination_task)
            else:
                result = await self._execute_parallel_coordination(coordination_task)  # Default
            
            # Calculate coordination metrics
            end_time = time.time()
            coordination_duration = end_time - start_time
            
            # Update coordination task
            coordination_task.completed_at = datetime.now(timezone.utc)
            coordination_task.status = "completed" if result["success"] else "failed"
            coordination_task.results = result
            
            # Create coordination result
            coordination_result = CoordinationResult(
                coordination_id=coordination_task.task_id,
                success=result["success"],
                coordination_type=coordination_task.coordination_type.value,
                agent_results=result.get("agent_results", {}),
                conflicts_resolved=conflicts_resolved,
                duration=coordination_duration,
                efficiency=self._calculate_coordination_efficiency(coordination_task, coordination_duration)
            )
            
            # Update metrics
            self.total_coordinations += 1
            if result["success"]:
                self.successful_coordinations += 1
            else:
                self.failed_coordinations += 1
            
            # Update average coordination time
            self.average_coordination_time = (
                (self.average_coordination_time * (self.total_coordinations - 1) + coordination_duration) /
                self.total_coordinations
            )
            
            # Store in history
            self.coordination_history.append({
                "task_id": coordination_task.task_id,
                "coordination_type": coordination_task.coordination_type.value,
                "duration": coordination_duration,
                "success": result["success"],
                "agents_involved": len(coordination_task.involved_agents),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return coordination_result
            
        except Exception as e:
            # Handle coordination failure
            self.total_coordinations += 1
            self.failed_coordinations += 1
            
            raise CoordinationError(f"Failed to coordinate agents: {str(e)}")
    
    async def _execute_parallel_coordination(self, task: CoordinationTask) -> Dict[str, Any]:
        """Execute parallel coordination."""
        try:
            # Create tasks for each agent
            agent_tasks = []
            for agent_id in task.involved_agents:
                agent_task = {
                    "agent_id": agent_id,
                    "task_data": task.agent_assignments.get(agent_id, {}),
                    "timeout": task.timeout_seconds
                }
                agent_tasks.append(self._execute_agent_task(agent_task))
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # Process results
            agent_results = {}
            success = True
            
            for i, result in enumerate(results):
                agent_id = task.involved_agents[i]
                
                if isinstance(result, Exception):
                    agent_results[agent_id] = {
                        "status": "failed",
                        "error": str(result),
                        "duration": task.timeout_seconds
                    }
                    success = False
                else:
                    agent_results[agent_id] = result
                    if not result.get("success", False):
                        success = False
            
            return {
                "success": success,
                "coordination_type": "parallel",
                "agent_results": agent_results,
                "total_agents": len(task.involved_agents),
                "execution_details": "All agents executed in parallel"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "coordination_type": "parallel"
            }
    
    async def _execute_sequential_coordination(self, task: CoordinationTask) -> Dict[str, Any]:
        """Execute sequential coordination."""
        try:
            agent_results = {}
            
            # Execute agents one by one
            for agent_id in task.involved_agents:
                agent_task = {
                    "agent_id": agent_id,
                    "task_data": task.agent_assignments.get(agent_id, {}),
                    "timeout": task.timeout_seconds,
                    "previous_results": agent_results  # Pass previous results
                }
                
                result = await self._execute_agent_task(agent_task)
                agent_results[agent_id] = result
                
                # Stop if any agent fails (unless configured otherwise)
                if not result.get("success", False):
                    break
            
            # Determine overall success
            success = all(result.get("success", False) for result in agent_results.values())
            
            return {
                "success": success,
                "coordination_type": "sequential",
                "agent_results": agent_results,
                "total_agents": len(task.involved_agents),
                "execution_details": "Agents executed sequentially"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "coordination_type": "sequential"
            }
    
    async def _execute_hybrid_coordination(self, task: CoordinationTask) -> Dict[str, Any]:
        """Execute hybrid coordination (combination of parallel and sequential)."""
        try:
            # Group agents by dependencies
            independent_agents = []
            dependent_groups = []
            
            for agent_id in task.involved_agents:
                dependencies = task.dependencies.get(agent_id, [])
                if not dependencies:
                    independent_agents.append(agent_id)
                else:
                    # Find or create dependency group
                    group_found = False
                    for group in dependent_groups:
                        if any(dep in group for dep in dependencies):
                            group.append(agent_id)
                            group_found = True
                            break
                    
                    if not group_found:
                        dependent_groups.append([agent_id])
            
            agent_results = {}
            
            # Execute independent agents in parallel
            if independent_agents:
                parallel_tasks = []
                for agent_id in independent_agents:
                    agent_task = {
                        "agent_id": agent_id,
                        "task_data": task.agent_assignments.get(agent_id, {}),
                        "timeout": task.timeout_seconds
                    }
                    parallel_tasks.append(self._execute_agent_task(agent_task))
                
                parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
                
                for i, result in enumerate(parallel_results):
                    agent_id = independent_agents[i]
                    if isinstance(result, Exception):
                        agent_results[agent_id] = {
                            "status": "failed",
                            "error": str(result),
                            "duration": task.timeout_seconds
                        }
                    else:
                        agent_results[agent_id] = result
            
            # Execute dependent groups sequentially
            for group in dependent_groups:
                for agent_id in group:
                    agent_task = {
                        "agent_id": agent_id,
                        "task_data": task.agent_assignments.get(agent_id, {}),
                        "timeout": task.timeout_seconds,
                        "previous_results": agent_results
                    }
                    
                    result = await self._execute_agent_task(agent_task)
                    agent_results[agent_id] = result
            
            # Determine overall success
            success = all(result.get("success", False) for result in agent_results.values())
            
            return {
                "success": success,
                "coordination_type": "hybrid",
                "agent_results": agent_results,
                "total_agents": len(task.involved_agents),
                "independent_agents": len(independent_agents),
                "dependent_groups": len(dependent_groups),
                "execution_details": "Hybrid execution with parallel and sequential components"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "coordination_type": "hybrid"
            }
    
    async def _execute_agent_task(self, agent_task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task for a specific agent."""
        start_time = time.time()
        
        try:
            agent_id = agent_task["agent_id"]
            task_data = agent_task.get("task_data", {})
            timeout = agent_task.get("timeout", 300)
            
            # Simulate agent task execution
            # In a real implementation, this would send a message to the agent
            # and wait for response through the message broker
            
            # Create a mock response based on agent capabilities
            capability = self.agent_capabilities.get(agent_id)
            if capability:
                # Simulate processing time based on agent performance
                processing_time = min(timeout, 30 / capability.performance_rating)
                await asyncio.sleep(min(processing_time, 2))  # Cap at 2 seconds for testing
                
                # Simulate success based on agent success rate
                import random
                success = random.random() < capability.success_rate
                
                execution_time = time.time() - start_time
                
                return {
                    "success": success,
                    "agent_id": agent_id,
                    "result": "Task completed successfully" if success else "Task failed",
                    "duration": execution_time,
                    "performance_rating": capability.performance_rating,
                    "task_data_processed": bool(task_data)
                }
            else:
                # Unknown agent
                return {
                    "success": False,
                    "agent_id": agent_id,
                    "result": "Unknown agent",
                    "duration": time.time() - start_time,
                    "error": f"Agent {agent_id} not found in capabilities"
                }
                
        except Exception as e:
            return {
                "success": False,
                "agent_id": agent_task.get("agent_id", "unknown"),
                "result": "Task execution failed",
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    def _calculate_coordination_efficiency(self, task: CoordinationTask, duration: float) -> float:
        """Calculate coordination efficiency."""
        # Base efficiency calculation
        expected_duration = len(task.involved_agents) * 30  # 30 seconds per agent baseline
        
        if task.coordination_type == CoordinationType.PARALLEL:
            expected_duration = 30  # Parallel should be fastest
        elif task.coordination_type == CoordinationType.SEQUENTIAL:
            expected_duration = len(task.involved_agents) * 30
        else:  # Hybrid
            expected_duration = len(task.involved_agents) * 20  # Between parallel and sequential
        
        # Calculate efficiency (lower duration = higher efficiency)
        if duration <= expected_duration:
            efficiency = 1.0
        else:
            efficiency = expected_duration / duration
        
        return min(efficiency, 1.0)
    
    async def resolve_conflicts(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve a list of conflicts."""
        resolutions = []
        
        for conflict_data in conflicts:
            # Convert dict to Conflict object
            conflict = Conflict(
                conflict_type=ConflictType(conflict_data.get("type", "resource_conflict")),
                involved_agents=conflict_data.get("agents", []),
                resource_name=conflict_data.get("resource"),
                severity=conflict_data.get("severity", "medium"),
                description=conflict_data.get("description", "")
            )
            
            resolution = await self.conflict_resolver.resolve_conflict(conflict)
            resolutions.append(resolution)
        
        return resolutions
    
    async def balance_workload(self, agent_workloads: Dict[str, AgentWorkload], new_task: Dict[str, Any]) -> Dict[str, Any]:
        """Balance workload across agents."""
        return await self.workload_balancer.balance_workload(agent_workloads, new_task)
    
    async def get_coordination_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a coordination task."""
        if task_id not in self.active_coordinations:
            return None
        
        task = self.active_coordinations[task_id]
        
        # Calculate progress
        if task.status == "completed":
            progress = 100.0
        elif task.status == "running":
            # Estimate progress based on time elapsed
            if task.started_at:
                elapsed = (datetime.now(timezone.utc) - task.started_at).total_seconds()
                progress = min((elapsed / task.timeout_seconds) * 100, 95)  # Cap at 95% until completion
            else:
                progress = 0.0
        else:
            progress = 0.0
        
        # Get agent statuses
        agent_statuses = {}
        for agent_id in task.involved_agents:
            if agent_id in task.results.get("agent_results", {}):
                agent_result = task.results["agent_results"][agent_id]
                agent_status = "completed" if agent_result.get("success") else "failed"
                agent_progress = 100.0 if agent_result.get("success") else 0.0
            else:
                agent_status = "running" if task.status == "running" else "pending"
                agent_progress = progress
            
            agent_statuses[agent_id] = {
                "status": agent_status,
                "progress": agent_progress
            }
        
        return {
            "coordination_id": task.task_id,
            "task_id": task_id,
            "progress": progress,
            "status": task.status,
            "coordination_type": task.coordination_type.value,
            "involved_agents": task.involved_agents,
            "agent_statuses": agent_statuses,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "estimated_completion": self._estimate_completion_time(task).isoformat() if task.status == "running" else None
        }
    
    def _estimate_completion_time(self, task: CoordinationTask) -> datetime:
        """Estimate completion time for a running task."""
        if task.started_at:
            estimated_duration = timedelta(seconds=task.timeout_seconds)
            return task.started_at + estimated_duration
        else:
            return datetime.now(timezone.utc) + timedelta(seconds=task.timeout_seconds)
    
    async def register_agent(self, agent_capability: AgentCapability) -> bool:
        """Register a new agent with the coordinator."""
        try:
            self.agent_capabilities[agent_capability.agent_id] = agent_capability
            return True
        except Exception:
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the coordinator."""
        try:
            if agent_id in self.agent_capabilities:
                del self.agent_capabilities[agent_id]
            return True
        except Exception:
            return False
    
    async def update_agent_status(self, agent_id: str, status_update: Dict[str, Any]) -> bool:
        """Update agent status information."""
        try:
            if agent_id in self.agent_capabilities:
                capability = self.agent_capabilities[agent_id]
                
                if "availability" in status_update:
                    capability.availability = status_update["availability"]
                if "current_load" in status_update:
                    capability.current_load = status_update["current_load"]
                if "performance_rating" in status_update:
                    capability.performance_rating = status_update["performance_rating"]
                if "response_time_avg" in status_update:
                    capability.response_time_avg = status_update["response_time_avg"]
                if "success_rate" in status_update:
                    capability.success_rate = status_update["success_rate"]
                
                return True
            
            return False
            
        except Exception:
            return False
    
    async def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination performance metrics."""
        success_rate = (
            self.successful_coordinations / self.total_coordinations
            if self.total_coordinations > 0 else 0.0
        )
        
        # Calculate recent performance
        recent_history = list(self.coordination_history)[-20:]  # Last 20 coordinations
        recent_success_rate = (
            sum(1 for record in recent_history if record["success"]) / len(recent_history)
            if recent_history else 0.0
        )
        
        # Agent utilization
        total_agents = len(self.agent_capabilities)
        active_agents = sum(
            1 for capability in self.agent_capabilities.values()
            if capability.current_load > 0
        )
        
        return {
            "total_coordinations": self.total_coordinations,
            "successful_coordinations": self.successful_coordinations,
            "failed_coordinations": self.failed_coordinations,
            "success_rate": success_rate,
            "recent_success_rate": recent_success_rate,
            "average_coordination_time": self.average_coordination_time,
            "total_agents": total_agents,
            "active_agents": active_agents,
            "agent_utilization_rate": active_agents / total_agents if total_agents > 0 else 0.0,
            "current_active_coordinations": len([
                task for task in self.active_coordinations.values()
                if task.status == "running"
            ]),
            "metrics_timestamp": datetime.now(timezone.utc).isoformat()
        }