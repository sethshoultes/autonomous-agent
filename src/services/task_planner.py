"""
Task Planner component for the Intelligence Layer.

This module provides sophisticated task planning, resource allocation,
dependency resolution, and workflow optimization capabilities.
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

from .intelligence_engine import (
    TaskPlan, TaskStep, TaskPlanningError, AgentWorkload,
    IntelligenceError
)


class ResourceEstimator:
    """Estimates resource requirements for tasks."""
    
    def __init__(self):
        self.resource_models = {}
        self.historical_data = deque(maxlen=1000)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize resource estimation models."""
        self.resource_models = {
            "email_processing": {
                "base_cpu_time": 30,  # seconds
                "cpu_per_email": 2,
                "memory_mb": 100,
                "network_kb": 50
            },
            "research_task": {
                "base_cpu_time": 300,
                "cpu_per_query": 60,
                "memory_mb": 500,
                "network_kb": 1000
            },
            "code_review": {
                "base_cpu_time": 600,
                "cpu_per_change": 0.5,
                "memory_mb": 200,
                "network_kb": 100
            },
            "report_generation": {
                "base_cpu_time": 900,
                "cpu_per_page": 30,
                "memory_mb": 300,
                "network_kb": 200
            }
        }
    
    def estimate_resources(self, task_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resource requirements for a task."""
        task_type = self._infer_task_type(task_definition)
        
        if task_type in self.resource_models:
            model = self.resource_models[task_type]
            
            # Base calculations
            cpu_time = model["base_cpu_time"]
            memory = model["memory_mb"]
            network = model["network_kb"]
            
            # Adjust based on task specifics
            if task_type == "email_processing":
                email_count = self._extract_email_count(task_definition)
                cpu_time += email_count * model["cpu_per_email"]
            elif task_type == "research_task":
                query_complexity = self._assess_query_complexity(task_definition)
                cpu_time += query_complexity * model["cpu_per_query"]
            elif task_type == "code_review":
                changes = self._extract_change_count(task_definition)
                cpu_time += changes * model["cpu_per_change"]
            elif task_type == "report_generation":
                pages = self._estimate_report_pages(task_definition)
                cpu_time += pages * model["cpu_per_page"]
            
            # Calculate cost estimate
            cost = self._calculate_cost(cpu_time, memory, network)
            
            return {
                "cpu_time": cpu_time,
                "memory": memory,
                "storage": max(memory // 4, 10),  # Estimate storage needs
                "network": network,
                "estimated_cost": cost,
                "confidence": 0.8
            }
        else:
            # Default estimation for unknown task types
            return {
                "cpu_time": 300,
                "memory": 256,
                "storage": 64,
                "network": 100,
                "estimated_cost": 5.0,
                "confidence": 0.5
            }
    
    def _infer_task_type(self, task_definition: Dict[str, Any]) -> str:
        """Infer task type from task definition."""
        goal = task_definition.get("goal", "").lower()
        
        if "email" in goal:
            return "email_processing"
        elif "research" in goal:
            return "research_task"
        elif "review" in goal or "code" in goal:
            return "code_review"
        elif "report" in goal:
            return "report_generation"
        else:
            return "general"
    
    def _extract_email_count(self, task_definition: Dict[str, Any]) -> int:
        """Extract email count from task definition."""
        goal = task_definition.get("goal", "")
        import re
        numbers = re.findall(r'\d+', goal)
        return int(numbers[0]) if numbers else 10  # Default 10 emails
    
    def _assess_query_complexity(self, task_definition: Dict[str, Any]) -> int:
        """Assess research query complexity."""
        query = task_definition.get("query", "")
        complexity = 1  # Base complexity
        
        # Increase complexity based on query characteristics
        if len(query) > 100:
            complexity += 1
        if "analysis" in query.lower():
            complexity += 1
        if "comparison" in query.lower():
            complexity += 1
        if task_definition.get("depth") == "detailed":
            complexity += 2
        
        return complexity
    
    def _extract_change_count(self, task_definition: Dict[str, Any]) -> int:
        """Extract change count for code review."""
        pr_data = task_definition.get("pull_request", {})
        return pr_data.get("changes", 100)  # Default 100 changes
    
    def _estimate_report_pages(self, task_definition: Dict[str, Any]) -> int:
        """Estimate report page count."""
        if "quarterly" in task_definition.get("goal", "").lower():
            return 20
        elif "detailed" in task_definition.get("goal", "").lower():
            return 15
        else:
            return 5  # Default page count
    
    def _calculate_cost(self, cpu_time: float, memory: float, network: float) -> float:
        """Calculate estimated cost."""
        # Simple cost model (in arbitrary units)
        cpu_cost = cpu_time * 0.01  # $0.01 per second
        memory_cost = memory * 0.001  # $0.001 per MB
        network_cost = network * 0.0001  # $0.0001 per KB
        
        return cpu_cost + memory_cost + network_cost
    
    def check_availability(self, resources: Dict[str, Any]) -> bool:
        """Check if resources are available."""
        # Simple availability check
        required_cpu = resources.get("cpu_time", 0)
        required_memory = resources.get("memory", 0)
        
        # Mock availability limits
        available_cpu = 3600  # 1 hour of CPU time
        available_memory = 2048  # 2GB memory
        
        return required_cpu <= available_cpu and required_memory <= available_memory


class DependencyResolver:
    """Resolves task dependencies and manages execution order."""
    
    def __init__(self):
        self.dependency_graph = {}
        self.resolved_dependencies = {}
    
    def resolve_dependencies(self, steps: List[TaskStep], custom_dependencies: List[Dict[str, Any]] = None) -> List[TaskStep]:
        """Resolve step dependencies and order steps appropriately."""
        # Build dependency graph
        dependency_map = {}
        for step in steps:
            dependency_map[step.step_id] = step.dependencies.copy()
        
        # Add custom dependencies
        if custom_dependencies:
            for dep in custom_dependencies:
                step_name = dep.get("step")
                depends_on = dep.get("depends_on", [])
                
                # Find step by name
                matching_step = next((s for s in steps if step_name.lower() in s.description.lower()), None)
                if matching_step:
                    dependency_map[matching_step.step_id].extend(depends_on)
        
        # Resolve dependencies using topological sort
        resolved_order = self._topological_sort(dependency_map, steps)
        
        # Return steps in resolved order
        step_lookup = {step.step_id: step for step in steps}
        return [step_lookup[step_id] for step_id in resolved_order if step_id in step_lookup]
    
    def _topological_sort(self, dependency_map: Dict[str, List[str]], steps: List[TaskStep]) -> List[str]:
        """Perform topological sort to resolve dependencies."""
        # Create mapping from step names to IDs
        name_to_id = {}
        for step in steps:
            # Extract key words from description for matching
            key_words = step.description.lower().split()
            for word in key_words:
                if len(word) > 3:  # Ignore short words
                    name_to_id[word] = step.step_id
        
        # Convert name-based dependencies to ID-based
        id_dependencies = {}
        for step_id, deps in dependency_map.items():
            id_dependencies[step_id] = []
            for dep_name in deps:
                matching_id = None
                for name, step_id_match in name_to_id.items():
                    if dep_name.lower() in name or name in dep_name.lower():
                        matching_id = step_id_match
                        break
                if matching_id and matching_id != step_id:
                    id_dependencies[step_id].append(matching_id)
        
        # Perform topological sort
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(step_id):
            if step_id in temp_visited:
                raise TaskPlanningError(f"Circular dependency detected involving step {step_id}")
            if step_id in visited:
                return
            
            temp_visited.add(step_id)
            
            for dep_id in id_dependencies.get(step_id, []):
                visit(dep_id)
            
            temp_visited.remove(step_id)
            visited.add(step_id)
            result.append(step_id)
        
        # Visit all steps
        for step in steps:
            if step.step_id not in visited:
                visit(step.step_id)
        
        return result
    
    def check_circular_dependencies(self, dependencies: Dict[str, List[str]]) -> bool:
        """Check for circular dependencies."""
        try:
            self._topological_sort(dependencies, [])
            return False
        except TaskPlanningError:
            return True


class WorkflowOptimizer:
    """Optimizes workflow execution for efficiency."""
    
    def __init__(self):
        self.optimization_strategies = [
            "parallel_execution",
            "resource_pooling",
            "batch_processing",
            "caching",
            "load_balancing"
        ]
        self.optimization_history = deque(maxlen=100)
    
    def optimize_sequence(self, steps: List[TaskStep]) -> List[TaskStep]:
        """Optimize step sequence for efficiency."""
        optimized_steps = steps.copy()
        
        # Apply various optimization strategies
        optimized_steps = self._optimize_parallel_execution(optimized_steps)
        optimized_steps = self._optimize_resource_usage(optimized_steps)
        optimized_steps = self._optimize_batching(optimized_steps)
        
        return optimized_steps
    
    def _optimize_parallel_execution(self, steps: List[TaskStep]) -> List[TaskStep]:
        """Optimize for parallel execution where possible."""
        # Group steps that can run in parallel
        parallel_groups = []
        current_group = []
        
        for step in steps:
            # Check if step can run in parallel with current group
            can_parallel = True
            for existing_step in current_group:
                if self._has_dependency(step, existing_step) or self._has_resource_conflict(step, existing_step):
                    can_parallel = False
                    break
            
            if can_parallel and len(current_group) < 3:  # Limit parallel group size
                current_group.append(step)
            else:
                if current_group:
                    parallel_groups.append(current_group)
                current_group = [step]
        
        if current_group:
            parallel_groups.append(current_group)
        
        # Flatten parallel groups back to list with parallel indicators
        optimized_steps = []
        for group in parallel_groups:
            if len(group) > 1:
                # Mark steps as parallel
                for step in group:
                    step.metadata = step.metadata or {}
                    step.metadata["parallel_group"] = len(optimized_steps) // len(group)
            optimized_steps.extend(group)
        
        return optimized_steps
    
    def _optimize_resource_usage(self, steps: List[TaskStep]) -> List[TaskStep]:
        """Optimize resource usage across steps."""
        # Sort steps by resource intensity
        resource_sorted = sorted(steps, key=lambda s: s.estimated_duration, reverse=True)
        
        # Distribute resource-intensive tasks
        optimized = []
        high_resource_steps = []
        low_resource_steps = []
        
        for step in resource_sorted:
            if step.estimated_duration > 300:  # 5 minutes threshold
                high_resource_steps.append(step)
            else:
                low_resource_steps.append(step)
        
        # Interleave high and low resource steps
        while high_resource_steps or low_resource_steps:
            if high_resource_steps:
                optimized.append(high_resource_steps.pop(0))
            if low_resource_steps:
                # Add 2-3 low resource steps after each high resource step
                for _ in range(min(3, len(low_resource_steps))):
                    optimized.append(low_resource_steps.pop(0))
        
        return optimized
    
    def _optimize_batching(self, steps: List[TaskStep]) -> List[TaskStep]:
        """Optimize by batching similar operations."""
        # Group similar steps together
        step_groups = defaultdict(list)
        for step in steps:
            step_type = self._classify_step_type(step.description)
            step_groups[step_type].append(step)
        
        # Create batched steps
        optimized_steps = []
        for step_type, type_steps in step_groups.items():
            if len(type_steps) > 1 and step_type in ["fetch", "process", "validate"]:
                # Create a batch step
                batch_step = TaskStep(
                    description=f"Batch {step_type} operations",
                    agent_id=type_steps[0].agent_id,
                    estimated_duration=sum(s.estimated_duration for s in type_steps) * 0.8,  # 20% efficiency gain
                    resources={"batch_size": len(type_steps), "original_steps": [s.step_id for s in type_steps]}
                )
                batch_step.metadata = {"batch_optimization": True, "original_count": len(type_steps)}
                optimized_steps.append(batch_step)
            else:
                optimized_steps.extend(type_steps)
        
        return optimized_steps
    
    def _classify_step_type(self, description: str) -> str:
        """Classify step type based on description."""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ["fetch", "get", "retrieve", "download"]):
            return "fetch"
        elif any(word in desc_lower for word in ["process", "analyze", "transform", "compute"]):
            return "process"
        elif any(word in desc_lower for word in ["validate", "check", "verify", "test"]):
            return "validate"
        elif any(word in desc_lower for word in ["send", "upload", "transmit", "deliver"]):
            return "transmit"
        else:
            return "general"
    
    def _has_dependency(self, step1: TaskStep, step2: TaskStep) -> bool:
        """Check if step1 depends on step2."""
        return step2.step_id in step1.dependencies
    
    def _has_resource_conflict(self, step1: TaskStep, step2: TaskStep) -> bool:
        """Check if steps have resource conflicts."""
        # Check agent conflict
        if step1.agent_id == step2.agent_id and step1.agent_id != "":
            return True
        
        # Check exclusive resource conflict
        resources1 = step1.resources or {}
        resources2 = step2.resources or {}
        
        exclusive1 = resources1.get("exclusive_resource")
        exclusive2 = resources2.get("exclusive_resource")
        
        return exclusive1 and exclusive2 and exclusive1 == exclusive2
    
    def calculate_efficiency(self, original_plan: TaskPlan, optimized_plan: TaskPlan) -> float:
        """Calculate efficiency improvement."""
        if original_plan.estimated_duration == 0:
            return 0.0
        
        time_improvement = (original_plan.estimated_duration - optimized_plan.estimated_duration) / original_plan.estimated_duration
        
        # Factor in resource optimization
        original_steps = len(original_plan.steps)
        optimized_steps = len(optimized_plan.steps)
        step_reduction = (original_steps - optimized_steps) / original_steps if original_steps > 0 else 0
        
        # Calculate overall efficiency
        efficiency = (time_improvement + step_reduction) / 2
        return max(0.0, min(efficiency, 1.0))


class TaskScheduler:
    """Schedules tasks based on priorities and constraints."""
    
    def __init__(self):
        self.scheduled_tasks = {}
        self.agent_schedules = defaultdict(list)
        self.resource_calendar = {}
    
    def schedule_task(self, plan: TaskPlan, constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Schedule a task plan."""
        constraints = constraints or {}
        
        # Calculate earliest start time
        earliest_start = datetime.now(timezone.utc)
        if constraints.get("not_before"):
            constraint_time = datetime.fromisoformat(constraints["not_before"].replace("Z", "+00:00"))
            earliest_start = max(earliest_start, constraint_time)
        
        # Calculate latest completion time
        latest_completion = None
        if constraints.get("deadline"):
            latest_completion = datetime.fromisoformat(constraints["deadline"].replace("Z", "+00:00"))
        
        # Schedule each step
        current_time = earliest_start
        step_schedule = {}
        
        for step in plan.steps:
            # Find available slot for step
            available_slot = self._find_available_slot(
                step, current_time, latest_completion
            )
            
            if available_slot:
                step_schedule[step.step_id] = {
                    "start_time": available_slot["start"].isoformat(),
                    "end_time": available_slot["end"].isoformat(),
                    "agent_id": step.agent_id,
                    "resources": step.resources
                }
                current_time = available_slot["end"]
            else:
                # Could not schedule step
                return {
                    "success": False,
                    "reason": f"Could not find available slot for step {step.step_id}",
                    "conflict_time": current_time.isoformat()
                }
        
        # Store schedule
        self.scheduled_tasks[plan.plan_id] = {
            "plan": plan,
            "schedule": step_schedule,
            "start_time": earliest_start.isoformat(),
            "end_time": current_time.isoformat(),
            "total_duration": (current_time - earliest_start).total_seconds()
        }
        
        return {
            "success": True,
            "schedule": step_schedule,
            "total_duration": (current_time - earliest_start).total_seconds(),
            "completion_time": current_time.isoformat()
        }
    
    def _find_available_slot(self, step: TaskStep, earliest_start: datetime, latest_end: datetime = None) -> Optional[Dict[str, datetime]]:
        """Find available time slot for a step."""
        duration = timedelta(seconds=step.estimated_duration)
        current_time = earliest_start
        
        # Check agent availability
        agent_schedule = self.agent_schedules.get(step.agent_id, [])
        
        while True:
            slot_end = current_time + duration
            
            # Check if slot exceeds deadline
            if latest_end and slot_end > latest_end:
                return None
            
            # Check for conflicts with agent schedule
            conflict = False
            for scheduled_slot in agent_schedule:
                scheduled_start = datetime.fromisoformat(scheduled_slot["start"])
                scheduled_end = datetime.fromisoformat(scheduled_slot["end"])
                
                if (current_time < scheduled_end and slot_end > scheduled_start):
                    conflict = True
                    current_time = scheduled_end
                    break
            
            if not conflict:
                # Found available slot
                agent_schedule.append({
                    "start": current_time.isoformat(),
                    "end": slot_end.isoformat(),
                    "step_id": step.step_id
                })
                
                return {
                    "start": current_time,
                    "end": slot_end
                }
            
            # Try next possible time after conflict
            continue


class ExecutionMonitor:
    """Monitors task plan execution."""
    
    def __init__(self):
        self.execution_status = {}
        self.performance_metrics = {}
        self.alerts = deque(maxlen=100)
    
    def start_monitoring(self, plan_id: str) -> None:
        """Start monitoring plan execution."""
        self.execution_status[plan_id] = {
            "status": "running",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "current_step": 0,
            "completed_steps": 0,
            "failed_steps": 0,
            "progress_percentage": 0.0
        }
    
    def update_step_status(self, plan_id: str, step_id: str, status: str, result: Dict[str, Any] = None) -> None:
        """Update status of a specific step."""
        if plan_id not in self.execution_status:
            return
        
        execution = self.execution_status[plan_id]
        
        if status == "completed":
            execution["completed_steps"] += 1
        elif status == "failed":
            execution["failed_steps"] += 1
            
            # Generate alert for failed step
            self.alerts.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "step_failure",
                "plan_id": plan_id,
                "step_id": step_id,
                "details": result or {}
            })
        
        # Update progress
        total_steps = execution.get("total_steps", 1)
        execution["progress_percentage"] = (execution["completed_steps"] / total_steps) * 100
        
        # Check if plan is complete
        if execution["completed_steps"] + execution["failed_steps"] >= total_steps:
            if execution["failed_steps"] == 0:
                execution["status"] = "completed"
            else:
                execution["status"] = "partially_completed"
            
            execution["end_time"] = datetime.now(timezone.utc).isoformat()
    
    def get_execution_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get current execution status."""
        return self.execution_status.get(plan_id)
    
    def get_performance_summary(self, plan_id: str) -> Dict[str, Any]:
        """Get performance summary for a plan."""
        status = self.execution_status.get(plan_id, {})
        
        if "start_time" in status and "end_time" in status:
            start = datetime.fromisoformat(status["start_time"])
            end = datetime.fromisoformat(status["end_time"])
            actual_duration = (end - start).total_seconds()
        else:
            actual_duration = None
        
        return {
            "plan_id": plan_id,
            "status": status.get("status", "unknown"),
            "progress": status.get("progress_percentage", 0),
            "completed_steps": status.get("completed_steps", 0),
            "failed_steps": status.get("failed_steps", 0),
            "actual_duration": actual_duration,
            "efficiency": self._calculate_efficiency(plan_id)
        }
    
    def _calculate_efficiency(self, plan_id: str) -> float:
        """Calculate execution efficiency."""
        # Placeholder efficiency calculation
        status = self.execution_status.get(plan_id, {})
        completed = status.get("completed_steps", 0)
        failed = status.get("failed_steps", 0)
        
        if completed + failed == 0:
            return 0.0
        
        return completed / (completed + failed)


class TaskPlanner:
    """Main task planner component."""
    
    def __init__(self, resource_allocator: ResourceEstimator = None, dependency_resolver: DependencyResolver = None, workflow_optimizer: WorkflowOptimizer = None):
        self.resource_allocator = resource_allocator or ResourceEstimator()
        self.dependency_resolver = dependency_resolver or DependencyResolver()
        self.workflow_optimizer = workflow_optimizer or WorkflowOptimizer()
        
        self.task_scheduler = TaskScheduler()
        self.execution_monitor = ExecutionMonitor()
        
        self.active_plans = {}
        self.plan_templates = {}
        self.planning_strategies = ["sequential", "parallel", "hybrid"]
        self.plan_archive = {}
        
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize plan templates."""
        self.plan_templates = {
            "email_processing_template": {
                "steps": [
                    {"description": "Fetch emails", "agent": "gmail_agent", "duration": 30},
                    {"description": "Classify emails", "agent": "gmail_agent", "duration": 60},
                    {"description": "Prioritize emails", "agent": "gmail_agent", "duration": 30},
                    {"description": "Process high priority", "agent": "gmail_agent", "duration": 120}
                ],
                "coordination_type": "sequential"
            },
            "research_workflow_template": {
                "steps": [
                    {"description": "Query optimization", "agent": "research_agent", "duration": 45},
                    {"description": "Source search", "agent": "research_agent", "duration": 180},
                    {"description": "Content analysis", "agent": "research_agent", "duration": 240},
                    {"description": "Synthesis and summary", "agent": "research_agent", "duration": 120}
                ],
                "coordination_type": "sequential"
            }
        }
    
    async def create_plan(self, task_definition: Dict[str, Any]) -> TaskPlan:
        """Create a comprehensive task plan."""
        try:
            # Validate task definition
            self._validate_task_definition(task_definition)
            
            # Check for template usage
            template_name = task_definition.get("template")
            if template_name and template_name in self.plan_templates:
                return await self._create_plan_from_template(task_definition, template_name)
            
            # Generate steps based on goal analysis
            steps = await self._generate_steps(task_definition)
            
            # Resolve dependencies
            custom_dependencies = task_definition.get("dependencies", [])
            resolved_steps = self.dependency_resolver.resolve_dependencies(steps, custom_dependencies)
            
            # Estimate resources
            resource_estimate = self.resource_allocator.estimate_resources(task_definition)
            
            # Determine coordination type
            coordination_type = self._determine_coordination_type(task_definition, resolved_steps)
            
            # Create agent assignments
            agent_assignments = self._create_agent_assignments(resolved_steps, task_definition)
            
            # Calculate estimated duration
            estimated_duration = self._calculate_total_duration(resolved_steps, coordination_type)
            
            # Create the plan
            plan = TaskPlan(
                goal=task_definition.get("goal", ""),
                steps=resolved_steps,
                estimated_duration=estimated_duration,
                required_resources=resource_estimate,
                agent_assignments=agent_assignments,
                coordination_type=coordination_type,
                priority=task_definition.get("priority", "medium"),
                required_agents=list(set(step.agent_id for step in resolved_steps)),
                workflow_type=self._classify_workflow_type(task_definition),
                stakeholder_involvement=bool(task_definition.get("stakeholders")),
                resource_allocation=resource_estimate
            )
            
            # Add parallel/sequential groups
            plan.parallel_groups, plan.sequential_groups = self._organize_execution_groups(resolved_steps, coordination_type)
            
            # Add approval gates for complex workflows
            if plan.workflow_type == "complex":
                plan.approval_gates = self._create_approval_gates(task_definition, resolved_steps)
            
            # Calculate efficiency score
            plan.efficiency_score = self._calculate_efficiency_score(plan)
            
            # Store plan
            self.active_plans[plan.plan_id] = plan
            
            return plan
            
        except Exception as e:
            raise TaskPlanningError(f"Failed to create task plan: {str(e)}")
    
    async def _create_plan_from_template(self, task_definition: Dict[str, Any], template_name: str) -> TaskPlan:
        """Create plan from template."""
        template = self.plan_templates[template_name]
        parameters = task_definition.get("parameters", {})
        
        # Create steps from template
        steps = []
        for step_template in template["steps"]:
            step = TaskStep(
                description=step_template["description"],
                agent_id=step_template["agent"],
                estimated_duration=step_template["duration"]
            )
            
            # Apply parameters
            if "batch_size" in parameters:
                step.resources["batch_size"] = parameters["batch_size"]
            
            steps.append(step)
        
        # Create plan
        plan = TaskPlan(
            goal=task_definition.get("goal", ""),
            steps=steps,
            estimated_duration=sum(s.estimated_duration for s in steps),
            coordination_type=template["coordination_type"],
            template_used=template_name,
            template_parameters=parameters
        )
        
        return plan
    
    async def _generate_steps(self, task_definition: Dict[str, Any]) -> List[TaskStep]:
        """Generate task steps based on goal analysis."""
        goal = task_definition.get("goal", "").lower()
        resources = task_definition.get("resources", {})
        available_agents = resources.get("agents", ["generic_agent"])
        
        steps = []
        
        # Analyze goal to determine required steps
        if "email" in goal:
            steps.extend(self._generate_email_steps(goal, available_agents))
        
        if "research" in goal:
            steps.extend(self._generate_research_steps(goal, available_agents, task_definition))
        
        if "report" in goal:
            steps.extend(self._generate_report_steps(goal, available_agents, task_definition))
        
        if "send" in goal or "email" in goal:
            steps.extend(self._generate_communication_steps(goal, available_agents))
        
        # Add workflow steps for complex tasks
        workflow_steps = task_definition.get("workflow_steps", [])
        if workflow_steps:
            for i, workflow_step in enumerate(workflow_steps):
                agent_id = available_agents[i % len(available_agents)]
                step = TaskStep(
                    description=workflow_step.replace("_", " ").title(),
                    agent_id=agent_id,
                    estimated_duration=self._estimate_step_duration(workflow_step),
                    priority=3 + i  # Increasing priority for later steps
                )
                steps.append(step)
        
        # If no specific steps generated, create generic steps
        if not steps:
            steps = [
                TaskStep(
                    description="Initialize task",
                    agent_id=available_agents[0],
                    estimated_duration=30
                ),
                TaskStep(
                    description="Execute main task",
                    agent_id=available_agents[0],
                    estimated_duration=300
                ),
                TaskStep(
                    description="Finalize and cleanup",
                    agent_id=available_agents[0],
                    estimated_duration=60
                )
            ]
        
        return steps
    
    def _generate_email_steps(self, goal: str, agents: List[str]) -> List[TaskStep]:
        """Generate email processing steps."""
        gmail_agent = next((agent for agent in agents if "gmail" in agent), agents[0])
        
        steps = [
            TaskStep(
                description="Fetch emails from inbox",
                agent_id=gmail_agent,
                estimated_duration=45,
                priority=1
            ),
            TaskStep(
                description="Classify email content",
                agent_id=gmail_agent,
                estimated_duration=90,
                priority=2,
                dependencies=[]  # Will be resolved later
            ),
            TaskStep(
                description="Prioritize emails by importance",
                agent_id=gmail_agent,
                estimated_duration=60,
                priority=3
            )
        ]
        
        # Add processing step based on goal
        if "process" in goal:
            steps.append(TaskStep(
                description="Process prioritized emails",
                agent_id=gmail_agent,
                estimated_duration=180,
                priority=4
            ))
        
        return steps
    
    def _generate_research_steps(self, goal: str, agents: List[str], task_definition: Dict[str, Any]) -> List[TaskStep]:
        """Generate research task steps."""
        research_agent = next((agent for agent in agents if "research" in agent), agents[0])
        
        steps = [
            TaskStep(
                description="Optimize search query",
                agent_id=research_agent,
                estimated_duration=60,
                priority=1
            ),
            TaskStep(
                description="Execute search across sources",
                agent_id=research_agent,
                estimated_duration=240,
                priority=2
            ),
            TaskStep(
                description="Analyze and filter results",
                agent_id=research_agent,
                estimated_duration=180,
                priority=3
            )
        ]
        
        # Add synthesis step for complex research
        depth = task_definition.get("constraints", {}).get("quality", "medium")
        if depth == "high" or "detailed" in goal:
            steps.append(TaskStep(
                description="Synthesize comprehensive findings",
                agent_id=research_agent,
                estimated_duration=300,
                priority=4
            ))
        else:
            steps.append(TaskStep(
                description="Summarize key findings",
                agent_id=research_agent,
                estimated_duration=120,
                priority=4
            ))
        
        return steps
    
    def _generate_report_steps(self, goal: str, agents: List[str], task_definition: Dict[str, Any]) -> List[TaskStep]:
        """Generate report creation steps."""
        report_agent = next((agent for agent in agents if "report" in agent), agents[0])
        
        steps = [
            TaskStep(
                description="Gather report data",
                agent_id=report_agent,
                estimated_duration=120,
                priority=1
            ),
            TaskStep(
                description="Structure report content",
                agent_id=report_agent,
                estimated_duration=180,
                priority=2
            ),
            TaskStep(
                description="Generate report document",
                agent_id=report_agent,
                estimated_duration=240,
                priority=3
            )
        ]
        
        # Add review step for important reports
        stakeholders = task_definition.get("stakeholders", [])
        if stakeholders or "quarterly" in goal or "board" in goal:
            steps.append(TaskStep(
                description="Review and approval process",
                agent_id=report_agent,
                estimated_duration=300,
                priority=4
            ))
        
        return steps
    
    def _generate_communication_steps(self, goal: str, agents: List[str]) -> List[TaskStep]:
        """Generate communication steps."""
        gmail_agent = next((agent for agent in agents if "gmail" in agent), agents[0])
        
        return [
            TaskStep(
                description="Prepare communication content",
                agent_id=gmail_agent,
                estimated_duration=90,
                priority=5
            ),
            TaskStep(
                description="Send communication",
                agent_id=gmail_agent,
                estimated_duration=30,
                priority=6
            )
        ]
    
    def _estimate_step_duration(self, step_name: str) -> int:
        """Estimate duration for a step based on its name."""
        duration_map = {
            "gather": 120,
            "analyze": 180,
            "generate": 240,
            "create": 300,
            "review": 180,
            "approve": 120,
            "distribute": 60,
            "send": 30,
            "fetch": 45,
            "process": 150,
            "validate": 90,
            "optimize": 120
        }
        
        step_lower = step_name.lower()
        for keyword, duration in duration_map.items():
            if keyword in step_lower:
                return duration
        
        return 120  # Default duration
    
    def _determine_coordination_type(self, task_definition: Dict[str, Any], steps: List[TaskStep]) -> str:
        """Determine coordination type for the plan."""
        execution_type = task_definition.get("execution_type")
        if execution_type:
            return execution_type
        
        # Auto-determine based on dependencies and resources
        has_dependencies = any(step.dependencies for step in steps)
        multiple_agents = len(set(step.agent_id for step in steps)) > 1
        
        if has_dependencies:
            return "sequential"
        elif multiple_agents and len(steps) > 3:
            return "hybrid"
        elif multiple_agents:
            return "parallel"
        else:
            return "sequential"
    
    def _create_agent_assignments(self, steps: List[TaskStep], task_definition: Dict[str, Any]) -> Dict[str, str]:
        """Create agent assignments for steps."""
        assignments = {}
        
        for step in steps:
            assignments[step.step_id] = step.agent_id
        
        return assignments
    
    def _calculate_total_duration(self, steps: List[TaskStep], coordination_type: str) -> int:
        """Calculate total estimated duration."""
        if coordination_type == "sequential":
            return sum(step.estimated_duration for step in steps)
        elif coordination_type == "parallel":
            # Assume perfect parallelization
            return max(step.estimated_duration for step in steps) if steps else 0
        else:  # hybrid
            # Estimate based on dependency groups
            return int(sum(step.estimated_duration for step in steps) * 0.7)  # 30% efficiency gain
    
    def _classify_workflow_type(self, task_definition: Dict[str, Any]) -> str:
        """Classify workflow complexity."""
        stakeholders = task_definition.get("stakeholders", [])
        workflow_steps = task_definition.get("workflow_steps", [])
        constraints = task_definition.get("constraints", {})
        
        complexity_score = 0
        
        if len(stakeholders) > 2:
            complexity_score += 2
        if len(workflow_steps) > 4:
            complexity_score += 2
        if constraints.get("quality") == "high":
            complexity_score += 1
        if "approval" in task_definition.get("goal", "").lower():
            complexity_score += 1
        
        if complexity_score >= 4:
            return "complex"
        elif complexity_score >= 2:
            return "moderate"
        else:
            return "simple"
    
    def _organize_execution_groups(self, steps: List[TaskStep], coordination_type: str) -> Tuple[List[List[str]], List[List[str]]]:
        """Organize steps into parallel and sequential groups."""
        parallel_groups = []
        sequential_groups = []
        
        if coordination_type == "parallel":
            # All steps can potentially run in parallel
            parallel_groups.append([step.step_id for step in steps])
        elif coordination_type == "sequential":
            # All steps run sequentially
            sequential_groups.append([step.step_id for step in steps])
        else:  # hybrid
            # Group steps by dependencies
            current_group = []
            for step in steps:
                if step.dependencies:
                    # Dependent step starts new sequential group
                    if current_group:
                        parallel_groups.append(current_group)
                    sequential_groups.append([step.step_id])
                    current_group = []
                else:
                    current_group.append(step.step_id)
            
            if current_group:
                parallel_groups.append(current_group)
        
        return parallel_groups, sequential_groups
    
    def _create_approval_gates(self, task_definition: Dict[str, Any], steps: List[TaskStep]) -> List[Dict[str, Any]]:
        """Create approval gates for complex workflows."""
        approval_gates = []
        stakeholders = task_definition.get("stakeholders", [])
        
        if stakeholders:
            # Add approval gate after major milestones
            milestone_steps = [i for i, step in enumerate(steps) if "review" in step.description.lower() or "generate" in step.description.lower()]
            
            for i, step_index in enumerate(milestone_steps):
                approval_gates.append({
                    "gate_id": f"approval_gate_{i+1}",
                    "after_step": steps[step_index].step_id,
                    "approvers": stakeholders[:2],  # Limit to 2 approvers
                    "required_approvals": 1,
                    "timeout_hours": 24
                })
        
        return approval_gates
    
    def _calculate_efficiency_score(self, plan: TaskPlan) -> float:
        """Calculate efficiency score for the plan."""
        base_score = 0.7  # Base efficiency
        
        # Boost for parallelization
        if plan.coordination_type == "parallel":
            base_score += 0.2
        elif plan.coordination_type == "hybrid":
            base_score += 0.1
        
        # Penalty for too many steps
        if len(plan.steps) > 10:
            base_score -= 0.1
        
        # Boost for resource optimization
        if plan.resource_allocation.get("confidence", 0) > 0.8:
            base_score += 0.1
        
        return max(0.0, min(base_score, 1.0))
    
    def _validate_task_definition(self, task_definition: Dict[str, Any]) -> None:
        """Validate task definition."""
        if not task_definition.get("goal"):
            raise TaskPlanningError("Task definition must include a goal")
        
        constraints = task_definition.get("constraints", {})
        if constraints.get("max_time", 0) < 0:
            raise TaskPlanningError("Invalid time constraint")
        
        resources = task_definition.get("resources", {})
        if not resources.get("agents"):
            # Provide default agent
            task_definition.setdefault("resources", {})["agents"] = ["generic_agent"]
    
    async def optimize_plan(self, plan: TaskPlan) -> TaskPlan:
        """Optimize an existing plan."""
        # Create optimized copy
        optimized_plan = TaskPlan(
            plan_id=plan.plan_id,
            goal=plan.goal,
            steps=plan.steps.copy(),
            required_resources=plan.required_resources.copy(),
            agent_assignments=plan.agent_assignments.copy(),
            priority=plan.priority,
            coordination_type=plan.coordination_type,
            version=plan.version + 1
        )
        
        # Apply optimizations
        optimized_steps = self.workflow_optimizer.optimize_sequence(optimized_plan.steps)
        optimized_plan.steps = optimized_steps
        
        # Recalculate duration
        optimized_plan.estimated_duration = self._calculate_total_duration(
            optimized_steps, optimized_plan.coordination_type
        )
        
        # Update efficiency score
        original_efficiency = plan.efficiency_score
        optimization_gain = self.workflow_optimizer.calculate_efficiency(plan, optimized_plan)
        optimized_plan.efficiency_score = min(original_efficiency + optimization_gain, 1.0)
        
        # Mark as optimized
        optimized_plan.optimization_applied = True
        optimized_plan.optimization_methods = ["workflow_optimization", "resource_balancing"]
        
        # Update active plans
        self.active_plans[optimized_plan.plan_id] = optimized_plan
        
        return optimized_plan
    
    async def start_execution_monitoring(self, plan_id: str) -> None:
        """Start monitoring plan execution."""
        if plan_id in self.active_plans:
            plan = self.active_plans[plan_id]
            self.execution_monitor.execution_status[plan_id] = {
                "total_steps": len(plan.steps)
            }
            self.execution_monitor.start_monitoring(plan_id)
    
    async def get_execution_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status for a plan."""
        status = self.execution_monitor.get_execution_status(plan_id)
        if status and plan_id in self.active_plans:
            plan = self.active_plans[plan_id]
            status.update({
                "plan_id": plan_id,
                "goal": plan.goal,
                "total_steps": len(plan.steps),
                "estimated_remaining_time": self._calculate_remaining_time(plan, status)
            })
        return status
    
    def _calculate_remaining_time(self, plan: TaskPlan, status: Dict[str, Any]) -> int:
        """Calculate estimated remaining time."""
        completed_steps = status.get("completed_steps", 0)
        total_steps = len(plan.steps)
        
        if completed_steps >= total_steps:
            return 0
        
        remaining_steps = total_steps - completed_steps
        avg_step_duration = plan.estimated_duration // total_steps if total_steps > 0 else 0
        
        return remaining_steps * avg_step_duration
    
    async def adapt_plan(self, execution_issue: Dict[str, Any]) -> TaskPlan:
        """Adapt plan based on execution issues."""
        plan_id = execution_issue.get("plan_id")
        issue_type = execution_issue.get("issue_type")
        
        if plan_id not in self.active_plans:
            raise TaskPlanningError(f"Plan {plan_id} not found")
        
        plan = self.active_plans[plan_id]
        adapted_plan = TaskPlan(**plan.__dict__)  # Copy plan
        adapted_plan.version += 1
        adapted_plan.adaptation_applied = True
        adapted_plan.adaptation_reason = issue_type
        
        # Apply adaptations based on issue type
        if issue_type == "agent_unavailable":
            adapted_plan = await self._adapt_for_agent_unavailability(adapted_plan, execution_issue)
        elif issue_type == "resource_shortage":
            adapted_plan = await self._adapt_for_resource_shortage(adapted_plan, execution_issue)
        elif issue_type == "time_constraint":
            adapted_plan = await self._adapt_for_time_constraint(adapted_plan, execution_issue)
        
        # Record adaptation
        adaptation_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "issue_type": issue_type,
            "adaptation_applied": True,
            "version": adapted_plan.version
        }
        adapted_plan.adaptation_history.append(adaptation_record)
        
        # Update active plans
        self.active_plans[plan_id] = adapted_plan
        
        return adapted_plan
    
    async def _adapt_for_agent_unavailability(self, plan: TaskPlan, issue: Dict[str, Any]) -> TaskPlan:
        """Adapt plan for agent unavailability."""
        unavailable_agent = issue.get("affected_agent")
        
        # Reassign steps to available agents
        available_agents = [agent for agent in plan.required_agents if agent != unavailable_agent]
        
        if not available_agents:
            raise TaskPlanningError("No available agents for reassignment")
        
        for step in plan.steps:
            if step.agent_id == unavailable_agent:
                # Assign to agent with lowest workload
                step.agent_id = available_agents[0]  # Simple assignment
                plan.agent_assignments[step.step_id] = step.agent_id
        
        return plan
    
    async def _adapt_for_resource_shortage(self, plan: TaskPlan, issue: Dict[str, Any]) -> TaskPlan:
        """Adapt plan for resource shortage."""
        # Reduce resource requirements by optimizing steps
        plan.steps = self.workflow_optimizer.optimize_sequence(plan.steps)
        
        # Recalculate resource requirements
        plan.required_resources = self.resource_allocator.estimate_resources({
            "goal": plan.goal,
            "steps": [s.__dict__ for s in plan.steps]
        })
        
        return plan
    
    async def _adapt_for_time_constraint(self, plan: TaskPlan, issue: Dict[str, Any]) -> TaskPlan:
        """Adapt plan for time constraints."""
        # Switch to more parallel execution if possible
        if plan.coordination_type == "sequential":
            plan.coordination_type = "hybrid"
            plan.estimated_duration = self._calculate_total_duration(plan.steps, "hybrid")
        
        return plan
    
    async def rollback_plan(self, plan_id: str, version: int) -> TaskPlan:
        """Rollback plan to previous version."""
        # In a real implementation, this would restore from version history
        # For now, return the current plan with version set to specified version
        if plan_id not in self.active_plans:
            raise TaskPlanningError(f"Plan {plan_id} not found")
        
        plan = self.active_plans[plan_id]
        rolled_back_plan = TaskPlan(**plan.__dict__)
        rolled_back_plan.version = version
        
        return rolled_back_plan
    
    async def validate_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """Validate a task plan."""
        issues = []
        recommendations = []
        
        # Check time constraints
        total_duration = sum(step.estimated_duration for step in plan.steps)
        if total_duration > 3600:  # More than 1 hour
            issues.append("Plan duration exceeds recommended limits")
            recommendations.append("Consider breaking into smaller tasks")
        
        # Check resource availability
        resource_check = self.resource_allocator.check_availability(plan.required_resources)
        if not resource_check:
            issues.append("Insufficient resources available")
            recommendations.append("Reduce resource requirements or schedule for later")
        
        # Check circular dependencies
        dependency_map = {step.step_id: step.dependencies for step in plan.steps}
        if self.dependency_resolver.check_circular_dependencies(dependency_map):
            issues.append("Circular dependencies detected")
            recommendations.append("Review and fix step dependencies")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
            "validation_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def calculate_cost_breakdown(self, plan: TaskPlan) -> Dict[str, Any]:
        """Calculate detailed cost breakdown."""
        agent_costs = {}
        for step in plan.steps:
            agent_id = step.agent_id
            if agent_id not in agent_costs:
                agent_costs[agent_id] = 0.0
            
            # Simple cost calculation
            step_cost = step.estimated_duration * 0.01  # $0.01 per second
            agent_costs[agent_id] += step_cost
        
        total_cost = sum(agent_costs.values())
        resource_costs = plan.required_resources.get("estimated_cost", 0.0)
        overhead_costs = total_cost * 0.1  # 10% overhead
        
        return {
            "total_cost": total_cost + resource_costs + overhead_costs,
            "agent_costs": agent_costs,
            "resource_costs": resource_costs,
            "overhead_costs": overhead_costs,
            "currency": "USD"
        }
    
    async def calculate_time_breakdown(self, plan: TaskPlan) -> Dict[str, Any]:
        """Calculate detailed time breakdown."""
        step_times = {step.step_id: step.estimated_duration for step in plan.steps}
        coordination_overhead = len(plan.steps) * 5  # 5 seconds per step for coordination
        
        total_time = plan.estimated_duration + coordination_overhead
        confidence_interval = (total_time * 0.8, total_time * 1.2)  # ±20%
        
        return {
            "estimated_total": total_time,
            "step_times": step_times,
            "coordination_overhead": coordination_overhead,
            "confidence_interval": confidence_interval,
            "time_unit": "seconds"
        }
    
    async def assess_risks(self, plan: TaskPlan) -> Dict[str, Any]:
        """Assess risks for a plan."""
        risk_factors = []
        
        # Complexity risk
        if len(plan.steps) > 8:
            risk_factors.append({
                "type": "complexity",
                "level": "medium",
                "description": "High number of steps increases failure risk"
            })
        
        # Dependency risk
        max_dependencies = max(len(step.dependencies) for step in plan.steps) if plan.steps else 0
        if max_dependencies > 3:
            risk_factors.append({
                "type": "dependency",
                "level": "high",
                "description": "Complex dependencies increase coordination risk"
            })
        
        # Resource risk
        if plan.required_resources.get("estimated_cost", 0) > 100:
            risk_factors.append({
                "type": "resource",
                "level": "medium",
                "description": "High resource requirements may cause availability issues"
            })
        
        # Calculate overall risk
        risk_levels = [rf["level"] for rf in risk_factors]
        if "high" in risk_levels:
            overall_risk = "high"
        elif "medium" in risk_levels:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        return {
            "overall_risk": overall_risk,
            "risk_factors": risk_factors,
            "mitigation_strategies": self._generate_mitigation_strategies(risk_factors),
            "contingency_plans": self._generate_contingency_plans(risk_factors)
        }
    
    def _generate_mitigation_strategies(self, risk_factors: List[Dict[str, Any]]) -> List[str]:
        """Generate risk mitigation strategies."""
        strategies = []
        
        for risk in risk_factors:
            if risk["type"] == "complexity":
                strategies.append("Break complex tasks into smaller, manageable steps")
            elif risk["type"] == "dependency":
                strategies.append("Create fallback plans for critical dependencies")
            elif risk["type"] == "resource":
                strategies.append("Reserve backup resources and implement resource monitoring")
        
        return strategies
    
    def _generate_contingency_plans(self, risk_factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate contingency plans."""
        contingencies = []
        
        for risk in risk_factors:
            if risk["type"] == "complexity":
                contingencies.append({
                    "trigger": "Step failure rate > 20%",
                    "action": "Pause execution and reassess plan",
                    "responsible": "task_manager"
                })
            elif risk["type"] == "dependency":
                contingencies.append({
                    "trigger": "Dependency unavailable > 30 minutes",
                    "action": "Activate alternative execution path",
                    "responsible": "coordination_manager"
                })
        
        return contingencies
    
    async def calculate_success_probability(self, plan: TaskPlan) -> float:
        """Calculate probability of plan success."""
        base_probability = 0.8  # Base 80% success rate
        
        # Adjust based on complexity
        complexity_penalty = len(plan.steps) * 0.02  # 2% penalty per step
        base_probability -= complexity_penalty
        
        # Adjust based on dependencies
        total_dependencies = sum(len(step.dependencies) for step in plan.steps)
        dependency_penalty = total_dependencies * 0.01  # 1% penalty per dependency
        base_probability -= dependency_penalty
        
        # Adjust based on resource availability
        resource_confidence = plan.required_resources.get("confidence", 0.8)
        base_probability *= resource_confidence
        
        # Ensure probability is within valid range
        return max(0.1, min(base_probability, 0.95))
    
    async def compare_plans(self, plan1: TaskPlan, plan2: TaskPlan) -> Dict[str, Any]:
        """Compare two plans."""
        duration_diff = plan2.estimated_duration - plan1.estimated_duration
        efficiency_diff = plan2.efficiency_score - plan1.efficiency_score
        
        # Calculate cost difference (if available)
        cost1 = plan1.required_resources.get("estimated_cost", 0)
        cost2 = plan2.required_resources.get("estimated_cost", 0)
        cost_diff = cost2 - cost1
        
        # Determine recommendation
        if efficiency_diff > 0.1 and duration_diff < 0:
            recommendation = f"Plan {plan2.plan_id} is recommended (more efficient and faster)"
        elif efficiency_diff > 0.05:
            recommendation = f"Plan {plan2.plan_id} is recommended (more efficient)"
        elif duration_diff < -300:  # 5 minutes faster
            recommendation = f"Plan {plan2.plan_id} is recommended (significantly faster)"
        else:
            recommendation = f"Plans are comparable, choose based on other criteria"
        
        return {
            "duration_difference": duration_diff,
            "cost_difference": cost_diff,
            "efficiency_difference": efficiency_diff,
            "recommendation": recommendation,
            "comparison_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def collect_plan_metrics(self, plan_id: str) -> Dict[str, Any]:
        """Collect metrics for a plan."""
        if plan_id not in self.active_plans:
            return {}
        
        plan = self.active_plans[plan_id]
        creation_time = plan.created_at
        
        # Calculate complexity score
        complexity_score = len(plan.steps) * 0.1 + len(plan.required_agents) * 0.2
        
        # Calculate resource utilization
        total_resources = sum(plan.required_resources.get(key, 0) for key in ["cpu_time", "memory", "storage"])
        resource_utilization = min(total_resources / 10000, 1.0)  # Normalize to 0-1
        
        return {
            "plan_id": plan_id,
            "creation_time": creation_time.isoformat(),
            "planning_duration": 5.0,  # Mock planning duration
            "complexity_score": complexity_score,
            "resource_utilization": resource_utilization,
            "steps_count": len(plan.steps),
            "agents_count": len(plan.required_agents),
            "estimated_duration": plan.estimated_duration,
            "efficiency_score": plan.efficiency_score
        }
    
    async def archive_plan(self, plan_id: str) -> Dict[str, Any]:
        """Archive a completed plan."""
        if plan_id not in self.active_plans:
            raise TaskPlanningError(f"Plan {plan_id} not found")
        
        plan = self.active_plans[plan_id]
        
        # Move to archive
        self.plan_archive[plan_id] = {
            "plan": plan,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "final_status": "archived"
        }
        
        # Remove from active plans
        del self.active_plans[plan_id]
        
        return {
            "archived": True,
            "plan_id": plan_id,
            "archived_at": self.plan_archive[plan_id]["archived_at"]
        }
    
    async def retrieve_plan(self, plan_id: str) -> TaskPlan:
        """Retrieve a plan from archive."""
        if plan_id in self.active_plans:
            return self.active_plans[plan_id]
        elif plan_id in self.plan_archive:
            return self.plan_archive[plan_id]["plan"]
        else:
            raise TaskPlanningError(f"Plan {plan_id} not found in active plans or archive")