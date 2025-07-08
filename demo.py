#!/usr/bin/env python3
"""
Demonstration script for the autonomous agent framework.

This script shows how to use the base agent classes, manager,
communication system, configuration, and lifecycle management.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to Python path for demonstration
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.base import BaseAgent, AgentMessage
from agents.manager import AgentManager, AgentConfig
from communication.broker import MessageBroker
from config.manager import ConfigManager
from lifecycle.manager import LifecycleManager
from lifecycle.hooks import PreStartHook, PostStartHook, PreStopHook, PostStopHook
from logging.manager import LoggingManager


class DemoAgent(BaseAgent):
    """Example concrete agent implementation."""
    
    async def _process_message(self, message: AgentMessage):
        """Process an incoming message."""
        self.logger.info(f"Processing message: {message.message_type} from {message.sender}")
        
        # Echo back a response
        if message.message_type == "ping":
            return AgentMessage(
                id=f"response_{message.id}",
                sender=self.agent_id,
                recipient=message.sender,
                message_type="pong",
                payload={"original_message": message.id}
            )
        
        return None
    
    async def _execute_task(self, task):
        """Execute a task."""
        self.logger.info(f"Executing task: {task}")
        
        # Simulate some work
        await asyncio.sleep(0.1)
        
        return {
            "status": "completed",
            "result": f"Task '{task.get('name', 'unknown')}' completed successfully",
            "agent_id": self.agent_id
        }
    
    async def _health_check(self):
        """Perform health check."""
        return True


async def main():
    """Main demonstration function."""
    print("🤖 Autonomous Agent Framework Demo")
    print("=" * 50)
    
    # 1. Setup logging
    print("\n1. Setting up logging...")
    logging_config = {
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    logging_manager = LoggingManager(logging_config)
    logger = logging_manager.get_logger("demo")
    logger.info("Logging system initialized")
    
    # 2. Setup configuration
    print("2. Setting up configuration...")
    config_manager = ConfigManager(logger)
    
    demo_config = {
        "agent_manager": {
            "max_agents": 10,
            "heartbeat_interval": 30,
            "communication_timeout": 10,
            "retry_attempts": 3
        },
        "logging": logging_config["logging"],
        "communication": {
            "message_broker": {
                "queue_size": 1000,
                "timeout": 5
            }
        },
        "agents": {
            "demo_agent_1": {
                "agent_type": "DemoAgent",
                "enabled": True,
                "priority": 1,
                "config": {
                    "demo_param": "value1"
                }
            },
            "demo_agent_2": {
                "agent_type": "DemoAgent", 
                "enabled": True,
                "priority": 2,
                "config": {
                    "demo_param": "value2"
                }
            }
        },
        "lifecycle": {
            "health_check_interval": 10,
            "performance_monitor_interval": 20,
            "graceful_shutdown_timeout": 5
        }
    }
    
    config_manager.load_config(demo_config)
    logger.info("Configuration loaded successfully")
    
    # 3. Setup communication
    print("3. Setting up communication...")
    message_broker = MessageBroker(logger)
    await message_broker.start()
    logger.info("Message broker started")
    
    # 4. Setup agent manager
    print("4. Setting up agent manager...")
    agent_manager = AgentManager(
        config=config_manager.get("agent_manager"),
        logger=logger,
        message_broker=message_broker
    )
    await agent_manager.start()
    logger.info("Agent manager started")
    
    # 5. Setup lifecycle management
    print("5. Setting up lifecycle management...")
    lifecycle_manager = LifecycleManager(
        config=demo_config,
        logger=logger
    )
    
    # Add lifecycle hooks
    lifecycle_manager.add_hook("pre_start", PreStartHook(logger))
    lifecycle_manager.add_hook("post_start", PostStartHook(logger))
    lifecycle_manager.add_hook("pre_stop", PreStopHook(logger))
    lifecycle_manager.add_hook("post_stop", PostStopHook(logger))
    
    logger.info("Lifecycle manager configured")
    
    # 6. Create and register agents
    print("6. Creating and registering agents...")
    
    agents = []
    for agent_id in ["demo_agent_1", "demo_agent_2"]:
        agent_config_dict = config_manager.get_agent_config(agent_id)
        
        # Create agent
        agent = DemoAgent(
            agent_id=agent_id,
            config=agent_config_dict["config"],
            logger=logging_manager.get_logger(f"agent.{agent_id}"),
            message_broker=message_broker
        )
        
        # Create agent config
        agent_config = AgentConfig.from_dict({
            "agent_id": agent_id,
            **agent_config_dict
        })
        
        # Register with manager
        await agent_manager.register_agent(agent, agent_config)
        agents.append(agent)
        
        logger.info(f"Created and registered agent: {agent_id}")
    
    # 7. Start agents with lifecycle management
    print("7. Starting agents...")
    
    for agent in agents:
        await lifecycle_manager.start_agent(agent)
        logger.info(f"Started agent: {agent.agent_id}")
    
    # 8. Demonstrate communication
    print("8. Demonstrating inter-agent communication...")
    
    # Send a ping message from agent 1 to agent 2
    ping_message = AgentMessage(
        id="demo_ping_001",
        sender="demo_agent_1",
        recipient="demo_agent_2",
        message_type="ping",
        payload={"message": "Hello from agent 1!"}
    )
    
    await agent_manager.send_message(ping_message)
    logger.info("Sent ping message from agent 1 to agent 2")
    
    # Give some time for message processing
    await asyncio.sleep(0.5)
    
    # 9. Demonstrate task execution
    print("9. Demonstrating task execution...")
    
    task = {
        "name": "demo_task",
        "action": "process_data",
        "data": [1, 2, 3, 4, 5]
    }
    
    result = await agent_manager.execute_task("demo_agent_1", task)
    logger.info(f"Task execution result: {result}")
    
    # 10. Check agent health and metrics
    print("10. Checking agent health and metrics...")
    
    health_status = await agent_manager.health_check_all_agents()
    logger.info(f"Health status: {health_status}")
    
    all_metrics = await agent_manager.get_all_agent_metrics()
    for agent_id, metrics in all_metrics.items():
        logger.info(f"Metrics for {agent_id}: {metrics}")
    
    # 11. Demonstrate broadcast messaging
    print("11. Demonstrating broadcast messaging...")
    
    broadcast_message = AgentMessage(
        id="demo_broadcast_001",
        sender="system",
        recipient="broadcast",
        message_type="announcement",
        payload={"message": "System maintenance in 5 minutes"}
    )
    
    await agent_manager.broadcast_message(broadcast_message)
    logger.info("Sent broadcast message to all agents")
    
    # Give some time for message processing
    await asyncio.sleep(0.5)
    
    # 12. Show system status
    print("12. System status summary...")
    
    manager_status = agent_manager.get_manager_status()
    logger.info(f"Agent Manager Status: {manager_status}")
    
    lifecycle_status = lifecycle_manager.get_manager_status()
    logger.info(f"Lifecycle Manager Status: {lifecycle_status}")
    
    # 13. Graceful shutdown
    print("13. Performing graceful shutdown...")
    
    # Stop agents with lifecycle management
    for agent in agents:
        await lifecycle_manager.stop_agent(agent.agent_id)
        logger.info(f"Stopped agent: {agent.agent_id}")
    
    # Stop manager and broker
    await agent_manager.stop()
    await message_broker.stop()
    
    logger.info("Demo completed successfully!")
    print("\n✅ Demo completed! Check the logs above for detailed execution flow.")


if __name__ == "__main__":
    # Run the demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise