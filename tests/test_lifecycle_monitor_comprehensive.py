"""
Comprehensive tests for lifecycle monitor.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.lifecycle.monitor import HealthCheck, PerformanceMonitor, LifecycleMonitor
from src.agents.base import BaseAgent


class TestHealthCheck:
    """Test HealthCheck functionality."""
    
    @pytest.fixture
    def health_check(self):
        """Create a HealthCheck instance."""
        return HealthCheck(interval=1.0, timeout=0.5)
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = Mock(spec=BaseAgent)
        agent.health_check = AsyncMock(return_value=True)
        agent.agent_id = "test_agent"
        return agent
    
    def test_health_check_initialization(self, health_check):
        """Test HealthCheck initialization."""
        assert health_check.interval == 1.0
        assert health_check.timeout == 0.5
        assert health_check.last_check_time is None
        assert health_check.last_check_result is None
        assert health_check.consecutive_failures == 0
        assert health_check.total_checks == 0
        assert health_check.total_failures == 0
    
    def test_health_check_default_initialization(self):
        """Test HealthCheck with default parameters."""
        health_check = HealthCheck()
        assert health_check.interval == 30.0
        assert health_check.timeout == 5.0
    
    @pytest.mark.asyncio
    async def test_check_success(self, health_check, mock_agent):
        """Test successful health check."""
        result = await health_check.check(mock_agent)
        
        assert result is True
        assert health_check.last_check_result is True
        assert health_check.last_check_time is not None
        assert health_check.consecutive_failures == 0
        assert health_check.total_checks == 1
        assert health_check.total_failures == 0
    
    @pytest.mark.asyncio
    async def test_check_failure(self, health_check, mock_agent):
        """Test failed health check."""
        mock_agent.health_check = AsyncMock(return_value=False)
        
        result = await health_check.check(mock_agent)
        
        assert result is False
        assert health_check.last_check_result is False
        assert health_check.last_check_time is not None
        assert health_check.consecutive_failures == 1
        assert health_check.total_checks == 1
        assert health_check.total_failures == 1
    
    @pytest.mark.asyncio
    async def test_check_timeout(self, health_check, mock_agent):
        """Test health check timeout."""
        async def slow_health_check():
            await asyncio.sleep(1.0)  # Longer than timeout
            return True
        
        mock_agent.health_check = slow_health_check
        
        result = await health_check.check(mock_agent)
        
        assert result is False
        assert health_check.consecutive_failures == 1
        assert health_check.total_failures == 1
    
    @pytest.mark.asyncio
    async def test_check_exception(self, health_check, mock_agent):
        """Test health check when agent raises exception."""
        mock_agent.health_check = AsyncMock(side_effect=Exception("Agent error"))
        
        result = await health_check.check(mock_agent)
        
        assert result is False
        assert health_check.consecutive_failures == 1
        assert health_check.total_failures == 1
    
    @pytest.mark.asyncio
    async def test_check_agent_without_health_check_method(self, health_check):
        """Test health check with agent that doesn't have health_check method."""
        mock_agent = Mock(spec=["agent_id"])
        mock_agent.agent_id = "test_agent"
        
        result = await health_check.check(mock_agent)
        
        assert result is True
        assert health_check.consecutive_failures == 0
    
    @pytest.mark.asyncio
    async def test_consecutive_failures_tracking(self, health_check, mock_agent):
        """Test consecutive failures tracking."""
        mock_agent.health_check = AsyncMock(return_value=False)
        
        # First failure
        await health_check.check(mock_agent)
        assert health_check.consecutive_failures == 1
        
        # Second failure
        await health_check.check(mock_agent)
        assert health_check.consecutive_failures == 2
        
        # Success resets consecutive failures
        mock_agent.health_check = AsyncMock(return_value=True)
        await health_check.check(mock_agent)
        assert health_check.consecutive_failures == 0
    
    def test_get_statistics(self, health_check):
        """Test get_statistics method."""
        stats = health_check.get_statistics()
        
        expected_keys = [
            "total_checks", "total_failures", "consecutive_failures",
            "last_check_time", "last_check_result", "success_rate"
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats["success_rate"] == 0.0  # No checks performed yet
    
    @pytest.mark.asyncio
    async def test_get_statistics_with_checks(self, health_check, mock_agent):
        """Test statistics after performing checks."""
        # Perform some checks
        await health_check.check(mock_agent)  # Success
        mock_agent.health_check = AsyncMock(return_value=False)
        await health_check.check(mock_agent)  # Failure
        
        stats = health_check.get_statistics()
        
        assert stats["total_checks"] == 2
        assert stats["total_failures"] == 1
        assert stats["success_rate"] == 0.5


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""
    
    @pytest.fixture
    def monitor(self):
        """Create a PerformanceMonitor instance."""
        return PerformanceMonitor(max_history=5)
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = Mock(spec=BaseAgent)
        agent.agent_id = "test_agent"
        agent.get_state = Mock(return_value="running")
        return agent
    
    def test_performance_monitor_initialization(self, monitor):
        """Test PerformanceMonitor initialization."""
        assert monitor.max_history == 5
        assert monitor.metrics == {}
    
    def test_performance_monitor_default_initialization(self):
        """Test PerformanceMonitor with default parameters."""
        monitor = PerformanceMonitor()
        assert monitor.max_history == 100
    
    def test_record_metric_new_agent(self, monitor):
        """Test recording metric for new agent."""
        monitor.record_metric("test_agent", "cpu_usage", 50.0)
        
        assert "test_agent" in monitor.metrics
        assert "cpu_usage" in monitor.metrics["test_agent"]
        assert len(monitor.metrics["test_agent"]["cpu_usage"]) == 1
        
        entry = monitor.metrics["test_agent"]["cpu_usage"][0]
        assert entry["value"] == 50.0
        assert "timestamp" in entry
    
    def test_record_metric_existing_agent(self, monitor):
        """Test recording metric for existing agent."""
        # Record first metric
        monitor.record_metric("test_agent", "cpu_usage", 50.0)
        # Record second metric
        monitor.record_metric("test_agent", "cpu_usage", 60.0)
        
        assert len(monitor.metrics["test_agent"]["cpu_usage"]) == 2
        assert monitor.metrics["test_agent"]["cpu_usage"][1]["value"] == 60.0
    
    def test_record_metric_max_history_limit(self, monitor):
        """Test that max_history limit is enforced."""
        # Record more metrics than max_history
        for i in range(10):
            monitor.record_metric("test_agent", "cpu_usage", float(i))
        
        # Should only keep the last max_history entries
        assert len(monitor.metrics["test_agent"]["cpu_usage"]) == monitor.max_history
        # Should have the most recent values
        values = [entry["value"] for entry in monitor.metrics["test_agent"]["cpu_usage"]]
        assert values == [5.0, 6.0, 7.0, 8.0, 9.0]
    
    def test_get_metrics_existing_agent(self, monitor):
        """Test getting metrics for existing agent."""
        monitor.record_metric("test_agent", "cpu_usage", 50.0)
        monitor.record_metric("test_agent", "memory_usage", 60.0)
        
        metrics = monitor.get_metrics("test_agent")
        
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert len(metrics["cpu_usage"]) == 1
        assert len(metrics["memory_usage"]) == 1
    
    def test_get_metrics_nonexistent_agent(self, monitor):
        """Test getting metrics for non-existent agent."""
        metrics = monitor.get_metrics("nonexistent_agent")
        assert metrics == {}
    
    def test_get_metric_specific_metric(self, monitor):
        """Test getting specific metric for agent."""
        monitor.record_metric("test_agent", "cpu_usage", 50.0)
        monitor.record_metric("test_agent", "memory_usage", 60.0)
        
        cpu_metrics = monitor.get_metric("test_agent", "cpu_usage")
        
        assert len(cpu_metrics) == 1
        assert cpu_metrics[0]["value"] == 50.0
    
    def test_get_metric_nonexistent_metric(self, monitor):
        """Test getting non-existent metric."""
        monitor.record_metric("test_agent", "cpu_usage", 50.0)
        
        metrics = monitor.get_metric("test_agent", "nonexistent_metric")
        assert metrics == []
    
    def test_get_metric_nonexistent_agent(self, monitor):
        """Test getting metric for non-existent agent."""
        metrics = monitor.get_metric("nonexistent_agent", "cpu_usage")
        assert metrics == []
    
    @pytest.mark.asyncio
    async def test_collect_metrics_success(self, monitor, mock_agent):
        """Test collecting metrics from agent."""
        with patch('psutil.Process') as mock_process:
            mock_proc = Mock()
            mock_proc.cpu_percent.return_value = 25.0
            mock_proc.memory_info.return_value = Mock(rss=1024*1024*100)  # 100MB
            mock_process.return_value = mock_proc
            
            with patch('os.getpid', return_value=1234):
                await monitor.collect_metrics(mock_agent)
        
        metrics = monitor.get_metrics("test_agent")
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert metrics["cpu_usage"][0]["value"] == 25.0
        assert metrics["memory_usage"][0]["value"] == 100.0  # MB
    
    @pytest.mark.asyncio
    async def test_collect_metrics_psutil_error(self, monitor, mock_agent):
        """Test collecting metrics when psutil raises error."""
        with patch('psutil.Process', side_effect=Exception("Process not found")):
            await monitor.collect_metrics(mock_agent)
        
        # Should not raise exception, and no metrics should be recorded
        metrics = monitor.get_metrics("test_agent")
        assert metrics == {}
    
    def test_clear_metrics_specific_agent(self, monitor):
        """Test clearing metrics for specific agent."""
        monitor.record_metric("test_agent1", "cpu_usage", 50.0)
        monitor.record_metric("test_agent2", "cpu_usage", 60.0)
        
        monitor.clear_metrics("test_agent1")
        
        assert "test_agent1" not in monitor.metrics
        assert "test_agent2" in monitor.metrics
    
    def test_clear_metrics_all(self, monitor):
        """Test clearing all metrics."""
        monitor.record_metric("test_agent1", "cpu_usage", 50.0)
        monitor.record_metric("test_agent2", "cpu_usage", 60.0)
        
        monitor.clear_metrics()
        
        assert monitor.metrics == {}
    
    def test_get_summary_single_agent(self, monitor):
        """Test getting summary for single agent."""
        # Record some metrics
        for i in range(5):
            monitor.record_metric("test_agent", "cpu_usage", float(i * 10))
        
        summary = monitor.get_summary("test_agent")
        
        assert "cpu_usage" in summary
        cpu_summary = summary["cpu_usage"]
        assert cpu_summary["count"] == 5
        assert cpu_summary["avg"] == 20.0
        assert cpu_summary["min"] == 0.0
        assert cpu_summary["max"] == 40.0
    
    def test_get_summary_nonexistent_agent(self, monitor):
        """Test getting summary for non-existent agent."""
        summary = monitor.get_summary("nonexistent_agent")
        assert summary == {}
    
    def test_get_all_summaries(self, monitor):
        """Test getting summaries for all agents."""
        monitor.record_metric("test_agent1", "cpu_usage", 50.0)
        monitor.record_metric("test_agent2", "memory_usage", 60.0)
        
        summaries = monitor.get_all_summaries()
        
        assert "test_agent1" in summaries
        assert "test_agent2" in summaries
        assert "cpu_usage" in summaries["test_agent1"]
        assert "memory_usage" in summaries["test_agent2"]


class TestLifecycleMonitor:
    """Test LifecycleMonitor functionality."""
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()
    
    @pytest.fixture
    def monitor(self, mock_logger):
        """Create a LifecycleMonitor instance."""
        return LifecycleMonitor(
            health_check_interval=1.0,
            performance_check_interval=0.5,
            logger=mock_logger
        )
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = Mock(spec=BaseAgent)
        agent.health_check = AsyncMock(return_value=True)
        agent.agent_id = "test_agent"
        agent.get_state = Mock(return_value="running")
        return agent
    
    def test_lifecycle_monitor_initialization(self, monitor):
        """Test LifecycleMonitor initialization."""
        assert monitor.health_check_interval == 1.0
        assert monitor.performance_check_interval == 0.5
        assert monitor.logger is not None
        assert monitor.agents == {}
        assert monitor.monitoring_tasks == {}
        assert monitor._stop_event.is_set() is False
    
    def test_lifecycle_monitor_default_initialization(self):
        """Test LifecycleMonitor with default parameters."""
        monitor = LifecycleMonitor()
        assert monitor.health_check_interval == 30.0
        assert monitor.performance_check_interval == 60.0
    
    def test_add_agent(self, monitor, mock_agent):
        """Test adding agent to monitor."""
        monitor.add_agent(mock_agent)
        
        assert "test_agent" in monitor.agents
        assert monitor.agents["test_agent"]["agent"] == mock_agent
        assert "health_check" in monitor.agents["test_agent"]
        assert "performance_monitor" in monitor.agents["test_agent"]
    
    def test_add_agent_duplicate(self, monitor, mock_agent):
        """Test adding duplicate agent."""
        monitor.add_agent(mock_agent)
        monitor.add_agent(mock_agent)  # Add again
        
        # Should still only have one entry
        assert len(monitor.agents) == 1
    
    def test_remove_agent_existing(self, monitor, mock_agent):
        """Test removing existing agent."""
        monitor.add_agent(mock_agent)
        result = monitor.remove_agent("test_agent")
        
        assert result is True
        assert "test_agent" not in monitor.agents
    
    def test_remove_agent_nonexistent(self, monitor):
        """Test removing non-existent agent."""
        result = monitor.remove_agent("nonexistent_agent")
        assert result is False
    
    def test_get_agent_status_existing(self, monitor, mock_agent):
        """Test getting status for existing agent."""
        monitor.add_agent(mock_agent)
        
        status = monitor.get_agent_status("test_agent")
        
        assert status is not None
        assert "agent_id" in status
        assert "health_status" in status
        assert "performance_metrics" in status
        assert status["agent_id"] == "test_agent"
    
    def test_get_agent_status_nonexistent(self, monitor):
        """Test getting status for non-existent agent."""
        status = monitor.get_agent_status("nonexistent_agent")
        assert status is None
    
    def test_get_all_statuses(self, monitor, mock_agent):
        """Test getting all agent statuses."""
        monitor.add_agent(mock_agent)
        
        # Add another agent
        mock_agent2 = Mock(spec=BaseAgent)
        mock_agent2.agent_id = "test_agent2"
        monitor.add_agent(mock_agent2)
        
        statuses = monitor.get_all_statuses()
        
        assert len(statuses) == 2
        assert "test_agent" in statuses
        assert "test_agent2" in statuses
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, monitor, mock_agent):
        """Test starting monitoring."""
        monitor.add_agent(mock_agent)
        
        await monitor.start_monitoring()
        
        assert "test_agent" in monitor.monitoring_tasks
        assert not monitor.monitoring_tasks["test_agent"].done()
        
        # Clean up
        await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, monitor, mock_agent):
        """Test stopping monitoring."""
        monitor.add_agent(mock_agent)
        await monitor.start_monitoring()
        
        await monitor.stop_monitoring()
        
        assert monitor._stop_event.is_set()
        assert len(monitor.monitoring_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_monitoring_task_health_check(self, monitor, mock_agent):
        """Test that monitoring task performs health checks."""
        monitor.add_agent(mock_agent)
        
        # Start monitoring briefly
        await monitor.start_monitoring()
        await asyncio.sleep(0.1)  # Let it run briefly
        await monitor.stop_monitoring()
        
        # Check that health check was performed
        health_stats = monitor.agents["test_agent"]["health_check"].get_statistics()
        assert health_stats["total_checks"] > 0
    
    @pytest.mark.asyncio
    async def test_monitoring_task_performance_check(self, monitor, mock_agent):
        """Test that monitoring task collects performance metrics."""
        monitor.add_agent(mock_agent)
        
        with patch('psutil.Process') as mock_process:
            mock_proc = Mock()
            mock_proc.cpu_percent.return_value = 25.0
            mock_proc.memory_info.return_value = Mock(rss=1024*1024*100)
            mock_process.return_value = mock_proc
            
            with patch('os.getpid', return_value=1234):
                await monitor.start_monitoring()
                await asyncio.sleep(0.1)  # Let it run briefly
                await monitor.stop_monitoring()
        
        # Check that performance metrics were collected
        metrics = monitor.agents["test_agent"]["performance_monitor"].get_metrics("test_agent")
        assert len(metrics) > 0
    
    @pytest.mark.asyncio
    async def test_agent_failure_handling(self, monitor, mock_agent):
        """Test handling of agent failures."""
        # Set up agent to fail health checks
        mock_agent.health_check = AsyncMock(return_value=False)
        monitor.add_agent(mock_agent)
        
        await monitor.start_monitoring()
        await asyncio.sleep(0.1)  # Let it run briefly
        await monitor.stop_monitoring()
        
        health_stats = monitor.agents["test_agent"]["health_check"].get_statistics()
        assert health_stats["total_failures"] > 0
    
    def test_cleanup(self, monitor, mock_agent):
        """Test cleanup method."""
        monitor.add_agent(mock_agent)
        
        monitor.cleanup()
        
        assert len(monitor.agents) == 0
        assert len(monitor.monitoring_tasks) == 0
        assert monitor._stop_event.is_set()


class TestLifecycleMonitorIntegration:
    """Test integration scenarios for lifecycle monitoring."""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_lifecycle(self):
        """Test complete monitoring lifecycle."""
        monitor = LifecycleMonitor(
            health_check_interval=0.1,
            performance_check_interval=0.1
        )
        
        # Create mock agent
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.agent_id = "test_agent"
        mock_agent.health_check = AsyncMock(return_value=True)
        mock_agent.get_state = Mock(return_value="running")
        
        try:
            # Add agent
            monitor.add_agent(mock_agent)
            assert len(monitor.agents) == 1
            
            # Start monitoring
            await monitor.start_monitoring()
            
            # Let it run briefly
            await asyncio.sleep(0.2)
            
            # Check status
            status = monitor.get_agent_status("test_agent")
            assert status is not None
            assert status["agent_id"] == "test_agent"
            
            # Remove agent
            monitor.remove_agent("test_agent")
            assert len(monitor.agents) == 0
            
        finally:
            # Stop monitoring
            await monitor.stop_monitoring()
            monitor.cleanup()
    
    @pytest.mark.asyncio
    async def test_multiple_agents_monitoring(self):
        """Test monitoring multiple agents simultaneously."""
        monitor = LifecycleMonitor(
            health_check_interval=0.1,
            performance_check_interval=0.1
        )
        
        # Create multiple mock agents
        agents = []
        for i in range(3):
            agent = Mock(spec=BaseAgent)
            agent.agent_id = f"test_agent_{i}"
            agent.health_check = AsyncMock(return_value=True)
            agent.get_state = Mock(return_value="running")
            agents.append(agent)
        
        try:
            # Add all agents
            for agent in agents:
                monitor.add_agent(agent)
            
            assert len(monitor.agents) == 3
            
            # Start monitoring
            await monitor.start_monitoring()
            
            # Let it run briefly
            await asyncio.sleep(0.2)
            
            # Check all statuses
            statuses = monitor.get_all_statuses()
            assert len(statuses) == 3
            
            for i in range(3):
                assert f"test_agent_{i}" in statuses
            
        finally:
            await monitor.stop_monitoring()
            monitor.cleanup()
    
    @pytest.mark.asyncio
    async def test_monitoring_with_agent_failures(self):
        """Test monitoring behavior when agents fail."""
        monitor = LifecycleMonitor(
            health_check_interval=0.05,
            performance_check_interval=0.1
        )
        
        # Create agent that will fail
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.agent_id = "failing_agent"
        mock_agent.health_check = AsyncMock(return_value=False)
        mock_agent.get_state = Mock(return_value="error")
        
        try:
            monitor.add_agent(mock_agent)
            await monitor.start_monitoring()
            
            # Let it run and accumulate failures
            await asyncio.sleep(0.2)
            
            # Check that failures were recorded
            status = monitor.get_agent_status("failing_agent")
            health_stats = status["health_status"]
            assert health_stats["total_failures"] > 0
            assert health_stats["success_rate"] == 0.0
            
        finally:
            await monitor.stop_monitoring()
            monitor.cleanup()