"""
Tests for Task Planner component following TDD principles.

This module tests multi-step task planning, resource allocation,
dependency resolution, and workflow optimization.
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from unittest.mock import MagicMock, AsyncMock

from src.services.intelligence_engine import (
    TaskPlanner,
    TaskPlan,
    TaskStep,
    TaskPlanningError,
    MultiStepPlanning,
    ResourceAllocation,
    WorkflowOptimization,
    DependencyResolver,
    TaskScheduler,
    ExecutionMonitor,
)
from tests.mocks.intelligence_mocks import MockTaskPlanner, MockTaskPlan, MockTaskStep


class TestTaskPlanner:
    """Test suite for TaskPlanner component."""
    
    @pytest.fixture
    def mock_resource_allocator(self):
        """Create mock resource allocator."""
        allocator = MagicMock()
        allocator.estimate_resources = MagicMock(return_value={
            "cpu_time": 300,
            "memory": 512,
            "agents": ["agent1", "agent2"],
            "estimated_cost": 10.0
        })
        allocator.check_availability = MagicMock(return_value=True)
        return allocator
    
    @pytest.fixture
    def mock_dependency_resolver(self):
        """Create mock dependency resolver."""
        resolver = MagicMock()
        resolver.resolve_dependencies = MagicMock(return_value=[])
        resolver.check_circular_dependencies = MagicMock(return_value=False)
        return resolver
    
    @pytest.fixture
    def mock_workflow_optimizer(self):
        """Create mock workflow optimizer."""
        optimizer = MagicMock()
        optimizer.optimize_sequence = MagicMock()
        optimizer.calculate_efficiency = MagicMock(return_value=0.85)
        return optimizer
    
    @pytest.fixture
    def task_planner(self, mock_resource_allocator, mock_dependency_resolver, mock_workflow_optimizer):
        """Create TaskPlanner instance."""
        return TaskPlanner(
            resource_allocator=mock_resource_allocator,
            dependency_resolver=mock_dependency_resolver,
            workflow_optimizer=mock_workflow_optimizer
        )
    
    @pytest.mark.asyncio
    async def test_task_planner_initialization(self, task_planner):
        """Test TaskPlanner initialization."""
        assert task_planner.resource_allocator is not None
        assert task_planner.dependency_resolver is not None
        assert task_planner.workflow_optimizer is not None
        assert task_planner.active_plans == {}
        assert task_planner.planning_strategies is not None
        
    @pytest.mark.asyncio
    async def test_simple_task_plan_creation(self, task_planner):
        """Test creating a simple task plan."""
        task_definition = {
            "goal": "Process 20 emails",
            "constraints": {
                "max_time": 600,
                "priority": "high"
            },
            "resources": {
                "agents": ["gmail_agent"],
                "budget": 50
            }
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        assert isinstance(plan, TaskPlan)
        assert plan.goal == "Process 20 emails"
        assert len(plan.steps) > 0
        assert plan.estimated_duration > 0
        assert plan.estimated_duration <= 600  # Respects constraint
        assert "gmail_agent" in plan.required_agents
        assert plan.priority == "high"
        
    @pytest.mark.asyncio
    async def test_multi_agent_task_plan(self, task_planner):
        """Test creating a multi-agent task plan."""
        task_definition = {
            "goal": "Research AI trends, create report, and email to stakeholders",
            "constraints": {
                "deadline": "2024-12-31T23:59:59Z",
                "quality": "high"
            },
            "resources": {
                "agents": ["research_agent", "report_agent", "gmail_agent"],
                "budget": 200
            }
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        assert isinstance(plan, TaskPlan)
        assert len(plan.steps) >= 3
        assert len(plan.required_agents) >= 2
        assert "research_agent" in plan.required_agents
        assert "gmail_agent" in plan.required_agents
        assert plan.agent_assignments is not None
        assert plan.coordination_type in ["sequential", "parallel", "hybrid"]
        
    @pytest.mark.asyncio
    async def test_complex_workflow_plan(self, task_planner):
        """Test creating a complex workflow plan."""
        task_definition = {
            "goal": "Complete quarterly business review workflow",
            "constraints": {
                "deadline": "2024-12-31T23:59:59Z",
                "quality": "high",
                "stakeholders": ["CEO", "board", "department_heads"]
            },
            "resources": {
                "agents": ["research_agent", "data_agent", "report_agent", "gmail_agent"],
                "budget": 500,
                "tools": ["analytics_platform", "presentation_software"]
            },
            "workflow_steps": [
                "gather_quarterly_data",
                "analyze_performance_metrics",
                "generate_insights",
                "create_presentation",
                "review_and_approve",
                "distribute_to_stakeholders"
            ]
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        assert isinstance(plan, TaskPlan)
        assert len(plan.steps) >= 6
        assert plan.workflow_type == "complex"
        assert plan.stakeholder_involvement is True
        assert plan.approval_gates is not None
        assert len(plan.approval_gates) > 0
        
    @pytest.mark.asyncio
    async def test_plan_with_dependencies(self, task_planner):
        """Test creating a plan with step dependencies."""
        task_definition = {
            "goal": "Data processing pipeline",
            "constraints": {"max_time": 1800},
            "resources": {"agents": ["data_agent"]},
            "dependencies": [
                {"step": "validate_data", "depends_on": ["fetch_data"]},
                {"step": "process_data", "depends_on": ["validate_data"]},
                {"step": "store_results", "depends_on": ["process_data"]}
            ]
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        assert isinstance(plan, TaskPlan)
        assert len(plan.steps) >= 4
        
        # Check dependency resolution
        validate_step = next((s for s in plan.steps if "validate" in s.description.lower()), None)
        fetch_step = next((s for s in plan.steps if "fetch" in s.description.lower()), None)
        
        assert validate_step is not None
        assert fetch_step is not None
        assert fetch_step.step_id in validate_step.dependencies
        
    @pytest.mark.asyncio
    async def test_plan_optimization(self, task_planner):
        """Test plan optimization."""
        task_definition = {
            "goal": "Optimize email processing workflow",
            "constraints": {"max_time": 900},
            "resources": {"agents": ["gmail_agent", "backup_agent"]}
        }
        
        initial_plan = await task_planner.create_plan(task_definition)
        optimized_plan = await task_planner.optimize_plan(initial_plan)
        
        assert optimized_plan.estimated_duration <= initial_plan.estimated_duration
        assert optimized_plan.efficiency_score >= initial_plan.efficiency_score
        assert optimized_plan.optimization_applied is True
        assert optimized_plan.optimization_methods is not None
        
    @pytest.mark.asyncio
    async def test_parallel_execution_plan(self, task_planner):
        """Test creating a plan with parallel execution."""
        task_definition = {
            "goal": "Parallel data processing",
            "constraints": {"max_time": 600},
            "resources": {"agents": ["agent1", "agent2", "agent3"]},
            "execution_type": "parallel"
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        assert plan.coordination_type == "parallel"
        assert len(plan.parallel_groups) > 0
        assert plan.estimated_duration < sum(step.estimated_duration for step in plan.steps)
        
    @pytest.mark.asyncio
    async def test_sequential_execution_plan(self, task_planner):
        """Test creating a plan with sequential execution."""
        task_definition = {
            "goal": "Sequential document processing",
            "constraints": {"max_time": 1200},
            "resources": {"agents": ["doc_agent"]},
            "execution_type": "sequential"
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        assert plan.coordination_type == "sequential"
        assert plan.estimated_duration == sum(step.estimated_duration for step in plan.steps)
        
    @pytest.mark.asyncio
    async def test_hybrid_execution_plan(self, task_planner):
        """Test creating a plan with hybrid execution."""
        task_definition = {
            "goal": "Hybrid workflow processing",
            "constraints": {"max_time": 1800},
            "resources": {"agents": ["agent1", "agent2", "agent3", "agent4"]},
            "execution_type": "hybrid"
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        assert plan.coordination_type == "hybrid"
        assert len(plan.parallel_groups) > 0
        assert len(plan.sequential_groups) > 0
        assert plan.estimated_duration < sum(step.estimated_duration for step in plan.steps)
        
    @pytest.mark.asyncio
    async def test_resource_allocation_planning(self, task_planner):
        """Test resource allocation during planning."""
        task_definition = {
            "goal": "Resource-intensive task",
            "constraints": {"max_time": 1200, "max_cost": 100},
            "resources": {
                "agents": ["heavy_agent", "light_agent"],
                "budget": 100,
                "cpu_limit": 4,
                "memory_limit": 2048
            }
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        assert plan.resource_allocation is not None
        assert plan.resource_allocation["total_cost"] <= 100
        assert plan.resource_allocation["cpu_usage"] <= 4
        assert plan.resource_allocation["memory_usage"] <= 2048
        
    @pytest.mark.asyncio
    async def test_plan_execution_monitoring(self, task_planner):
        """Test plan execution monitoring."""
        task_definition = {
            "goal": "Monitored task execution",
            "constraints": {"max_time": 600},
            "resources": {"agents": ["monitor_agent"]}
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        # Start execution monitoring
        await task_planner.start_execution_monitoring(plan.plan_id)
        
        # Check monitoring status
        status = await task_planner.get_execution_status(plan.plan_id)
        
        assert status is not None
        assert status["plan_id"] == plan.plan_id
        assert "progress" in status
        assert "current_step" in status
        assert "estimated_remaining_time" in status
        assert "status" in status
        
    @pytest.mark.asyncio
    async def test_plan_adaptation_during_execution(self, task_planner):
        """Test plan adaptation during execution."""
        task_definition = {
            "goal": "Adaptive task execution",
            "constraints": {"max_time": 900},
            "resources": {"agents": ["adaptive_agent"]}
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        # Simulate execution issue
        execution_issue = {
            "plan_id": plan.plan_id,
            "issue_type": "agent_unavailable",
            "affected_step": plan.steps[0].step_id,
            "severity": "medium"
        }
        
        adapted_plan = await task_planner.adapt_plan(execution_issue)
        
        assert adapted_plan.plan_id == plan.plan_id
        assert adapted_plan.adaptation_applied is True
        assert adapted_plan.adaptation_reason == "agent_unavailable"
        assert len(adapted_plan.adaptation_history) > 0
        
    @pytest.mark.asyncio
    async def test_plan_rollback_capability(self, task_planner):
        """Test plan rollback capability."""
        task_definition = {
            "goal": "Rollback test task",
            "constraints": {"max_time": 600},
            "resources": {"agents": ["rollback_agent"]}
        }
        
        original_plan = await task_planner.create_plan(task_definition)
        
        # Make modifications
        modified_plan = await task_planner.optimize_plan(original_plan)
        
        # Rollback to original
        rolled_back_plan = await task_planner.rollback_plan(modified_plan.plan_id, version=1)
        
        assert rolled_back_plan.plan_id == original_plan.plan_id
        assert rolled_back_plan.version == 1
        assert rolled_back_plan.estimated_duration == original_plan.estimated_duration
        
    @pytest.mark.asyncio
    async def test_plan_validation(self, task_planner):
        """Test plan validation."""
        task_definition = {
            "goal": "Validation test task",
            "constraints": {"max_time": 300},
            "resources": {"agents": ["validation_agent"]}
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        validation_result = await task_planner.validate_plan(plan)
        
        assert validation_result["valid"] is True
        assert validation_result["issues"] == []
        assert validation_result["recommendations"] is not None
        
    @pytest.mark.asyncio
    async def test_plan_validation_with_issues(self, task_planner):
        """Test plan validation with issues."""
        # Create a plan with potential issues
        task_definition = {
            "goal": "Invalid task configuration",
            "constraints": {"max_time": 10},  # Very short time
            "resources": {"agents": ["slow_agent"]}
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        # Manually add unrealistic step
        unrealistic_step = TaskStep(
            description="Extremely long task",
            estimated_duration=3600,  # 1 hour
            agent_id="slow_agent"
        )
        plan.steps.append(unrealistic_step)
        
        validation_result = await task_planner.validate_plan(plan)
        
        assert validation_result["valid"] is False
        assert len(validation_result["issues"]) > 0
        assert any("time constraint" in issue.lower() for issue in validation_result["issues"])
        
    @pytest.mark.asyncio
    async def test_plan_cost_estimation(self, task_planner):
        """Test plan cost estimation."""
        task_definition = {
            "goal": "Cost estimation task",
            "constraints": {"max_cost": 150},
            "resources": {
                "agents": ["expensive_agent", "cheap_agent"],
                "budget": 150
            }
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        cost_breakdown = await task_planner.calculate_cost_breakdown(plan)
        
        assert cost_breakdown["total_cost"] <= 150
        assert cost_breakdown["agent_costs"] is not None
        assert cost_breakdown["resource_costs"] is not None
        assert cost_breakdown["overhead_costs"] is not None
        
    @pytest.mark.asyncio
    async def test_plan_time_estimation_accuracy(self, task_planner):
        """Test plan time estimation accuracy."""
        task_definition = {
            "goal": "Time estimation accuracy test",
            "constraints": {"max_time": 1200},
            "resources": {"agents": ["time_agent"]}
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        time_breakdown = await task_planner.calculate_time_breakdown(plan)
        
        assert time_breakdown["estimated_total"] == plan.estimated_duration
        assert time_breakdown["step_times"] is not None
        assert time_breakdown["coordination_overhead"] > 0
        assert time_breakdown["confidence_interval"] is not None
        
    @pytest.mark.asyncio
    async def test_plan_risk_assessment(self, task_planner):
        """Test plan risk assessment."""
        task_definition = {
            "goal": "Risk assessment task",
            "constraints": {"max_time": 800},
            "resources": {"agents": ["risky_agent"]}
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        risk_assessment = await task_planner.assess_risks(plan)
        
        assert risk_assessment["overall_risk"] is not None
        assert risk_assessment["risk_factors"] is not None
        assert risk_assessment["mitigation_strategies"] is not None
        assert risk_assessment["contingency_plans"] is not None
        
    @pytest.mark.asyncio
    async def test_plan_success_probability(self, task_planner):
        """Test plan success probability calculation."""
        task_definition = {
            "goal": "Success probability test",
            "constraints": {"max_time": 600},
            "resources": {"agents": ["reliable_agent"]}
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        success_probability = await task_planner.calculate_success_probability(plan)
        
        assert 0.0 <= success_probability <= 1.0
        assert success_probability > 0.5  # Should be reasonably likely to succeed
        
    @pytest.mark.asyncio
    async def test_plan_comparison(self, task_planner):
        """Test plan comparison functionality."""
        task_definition = {
            "goal": "Plan comparison test",
            "constraints": {"max_time": 800},
            "resources": {"agents": ["agent1", "agent2"]}
        }
        
        plan1 = await task_planner.create_plan(task_definition)
        plan2 = await task_planner.optimize_plan(plan1)
        
        comparison = await task_planner.compare_plans(plan1, plan2)
        
        assert comparison["duration_difference"] is not None
        assert comparison["cost_difference"] is not None
        assert comparison["efficiency_difference"] is not None
        assert comparison["recommendation"] is not None
        
    @pytest.mark.asyncio
    async def test_plan_template_usage(self, task_planner):
        """Test plan template usage."""
        template_name = "email_processing_template"
        
        task_definition = {
            "goal": "Process emails using template",
            "template": template_name,
            "parameters": {
                "batch_size": 50,
                "priority_threshold": 0.8
            }
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        assert plan.template_used == template_name
        assert plan.template_parameters is not None
        assert plan.template_parameters["batch_size"] == 50
        
    @pytest.mark.asyncio
    async def test_error_handling_invalid_definition(self, task_planner):
        """Test error handling with invalid task definition."""
        invalid_definition = {
            "goal": "",  # Empty goal
            "constraints": {"max_time": -100},  # Negative time
            "resources": {}  # Empty resources
        }
        
        with pytest.raises(TaskPlanningError) as exc_info:
            await task_planner.create_plan(invalid_definition)
        
        assert "invalid task definition" in str(exc_info.value).lower()
        
    @pytest.mark.asyncio
    async def test_error_handling_resource_conflict(self, task_planner):
        """Test error handling with resource conflicts."""
        task_definition = {
            "goal": "Resource conflict test",
            "constraints": {"max_time": 600},
            "resources": {
                "agents": ["overbooked_agent"],
                "exclusive_resource": "database_lock"
            }
        }
        
        # Simulate resource conflict
        task_planner.resource_allocator.check_availability.return_value = False
        
        with pytest.raises(TaskPlanningError) as exc_info:
            await task_planner.create_plan(task_definition)
        
        assert "resource conflict" in str(exc_info.value).lower()
        
    @pytest.mark.asyncio
    async def test_error_handling_circular_dependencies(self, task_planner):
        """Test error handling with circular dependencies."""
        task_definition = {
            "goal": "Circular dependency test",
            "constraints": {"max_time": 600},
            "resources": {"agents": ["dep_agent"]},
            "dependencies": [
                {"step": "step_a", "depends_on": ["step_b"]},
                {"step": "step_b", "depends_on": ["step_a"]}
            ]
        }
        
        # Simulate circular dependency detection
        task_planner.dependency_resolver.check_circular_dependencies.return_value = True
        
        with pytest.raises(TaskPlanningError) as exc_info:
            await task_planner.create_plan(task_definition)
        
        assert "circular dependency" in str(exc_info.value).lower()
        
    @pytest.mark.asyncio
    async def test_plan_metrics_collection(self, task_planner):
        """Test plan metrics collection."""
        task_definition = {
            "goal": "Metrics collection test",
            "constraints": {"max_time": 600},
            "resources": {"agents": ["metrics_agent"]}
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        metrics = await task_planner.collect_plan_metrics(plan.plan_id)
        
        assert metrics["creation_time"] is not None
        assert metrics["planning_duration"] > 0
        assert metrics["complexity_score"] > 0
        assert metrics["resource_utilization"] > 0
        
    @pytest.mark.asyncio
    async def test_plan_archival_and_retrieval(self, task_planner):
        """Test plan archival and retrieval."""
        task_definition = {
            "goal": "Archive test task",
            "constraints": {"max_time": 600},
            "resources": {"agents": ["archive_agent"]}
        }
        
        plan = await task_planner.create_plan(task_definition)
        
        # Archive the plan
        archive_result = await task_planner.archive_plan(plan.plan_id)
        assert archive_result["archived"] is True
        
        # Retrieve the plan
        retrieved_plan = await task_planner.retrieve_plan(plan.plan_id)
        assert retrieved_plan.plan_id == plan.plan_id
        assert retrieved_plan.goal == plan.goal