#!/usr/bin/env python3
"""
Intelligence Layer Integration Example

This example demonstrates how to integrate the Intelligence Layer
with existing agents for enhanced decision-making and coordination.
"""

import asyncio
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from services.intelligence_engine import (
    IntelligenceEngine, DecisionContext, Decision
)
from agents.base import AgentMessage, AgentState
from agents.gmail_agent import GmailAgent


class IntelligentEmailProcessor:
    """Example of Gmail Agent enhanced with Intelligence Layer."""
    
    def __init__(self):
        self.intelligence = IntelligenceEngine()
        self.gmail_agent = GmailAgent()
    
    async def initialize(self):
        """Initialize the intelligence engine and Gmail agent."""
        await self.intelligence.initialize()
        print("✅ Intelligence Layer initialized")
        print("✅ Gmail Agent ready")
    
    async def process_email_with_intelligence(self, email_data):
        """Process an email using intelligent decision-making."""
        print(f"\n📧 Processing email: {email_data.get('subject', 'No Subject')}")
        
        # Create decision context for email triage
        context = DecisionContext(
            agent_id="gmail_agent",
            task_type="email_triage",
            input_data={
                "email": email_data,
                "sender": email_data.get("sender", ""),
                "subject": email_data.get("subject", ""),
                "content": email_data.get("body", ""),
                "attachments": email_data.get("attachments", [])
            },
            constraints={
                "max_processing_time": 30,  # seconds
                "priority_levels": ["urgent", "high", "normal", "low"]
            }
        )
        
        # Make intelligent decision
        decision = await self.intelligence.make_decision(context)
        
        print(f"🎯 Decision: {decision.action}")
        print(f"🔍 Confidence: {decision.confidence:.2f}")
        print(f"💭 Reasoning: {decision.reasoning}")
        
        # Act on the decision
        await self._execute_email_action(email_data, decision)
        
        # Learn from the outcome
        feedback = {
            "decision_id": decision.decision_id,
            "outcome": "success",  # In practice, this would be determined by user feedback
            "processing_time": 25,  # Example processing time
            "user_satisfaction": 0.9
        }
        
        learning_result = await self.intelligence.learn_from_feedback(feedback)
        print(f"🧠 Learning applied: {learning_result.get('improvement', 'No changes')}")
        
        return decision
    
    async def _execute_email_action(self, email_data, decision: Decision):
        """Execute the action determined by the intelligence layer."""
        action = decision.action
        
        if action == "urgent_response":
            print("⚡ Marking as urgent and preparing immediate response")
        elif action == "schedule_response":
            scheduled_time = decision.metadata.get("scheduled_time", "later today")
            print(f"📅 Scheduling response for {scheduled_time}")
        elif action == "archive":
            print("📁 Archiving email (low priority)")
        elif action == "flag_for_review":
            print("🏷️ Flagging for manual review")
        else:
            print(f"❓ Unknown action: {action}")


class IntelligentMultiAgentCoordinator:
    """Example of multi-agent coordination using Intelligence Layer."""
    
    def __init__(self):
        self.intelligence = IntelligenceEngine()
    
    async def initialize(self):
        """Initialize the intelligence engine."""
        await self.intelligence.initialize()
        print("✅ Multi-Agent Coordinator initialized")
    
    async def coordinate_complex_task(self):
        """Demonstrate intelligent coordination of multiple agents."""
        print("\n🤝 Coordinating complex task across multiple agents")
        
        # Define a complex task requiring multiple agents
        task_definition = {
            "task_id": "research_and_report",
            "description": "Research recent AI developments and create summary report",
            "requirements": [
                "web_research",
                "content_analysis", 
                "report_generation",
                "email_notification"
            ],
            "priority": "high",
            "deadline": (datetime.now(timezone.utc).timestamp() + 3600)  # 1 hour
        }
        
        # Create task plan using intelligence
        plan = await self.intelligence.create_task_plan(task_definition)
        
        print(f"📋 Task Plan created with {len(plan.steps)} steps:")
        for i, step in enumerate(plan.steps, 1):
            print(f"  {i}. {step.description} (Agent: {step.agent_id})")
        
        # Coordinate agents for execution
        coordination_request = {
            "task_plan": plan.to_dict(),
            "available_agents": ["research_agent", "code_agent", "gmail_agent"],
            "resource_constraints": {
                "max_parallel_tasks": 3,
                "time_limit": 3600
            }
        }
        
        coordination_result = await self.intelligence.coordinate_agents(coordination_request)
        
        print(f"🎯 Coordination result: {coordination_result.strategy}")
        print(f"📊 Estimated completion: {coordination_result.estimated_completion_time}s")
        
        if coordination_result.conflicts:
            print(f"⚠️ Conflicts detected: {len(coordination_result.conflicts)}")
            for conflict in coordination_result.conflicts:
                print(f"  - {conflict.description}")
        
        return coordination_result


async def demonstrate_learning_adaptation():
    """Demonstrate the learning and adaptation capabilities."""
    print("\n🧠 Demonstrating Learning and Adaptation")
    
    intelligence = IntelligenceEngine()
    await intelligence.initialize()
    
    # Simulate multiple decision-making scenarios with feedback
    scenarios = [
        {
            "context": DecisionContext(
                agent_id="gmail_agent",
                task_type="email_triage",
                input_data={"subject": "URGENT: Server Down", "sender": "ops@company.com"}
            ),
            "feedback": {"outcome": "success", "user_satisfaction": 0.95}
        },
        {
            "context": DecisionContext(
                agent_id="research_agent", 
                task_type="query_optimization",
                input_data={"query": "machine learning trends 2024"}
            ),
            "feedback": {"outcome": "partial", "user_satisfaction": 0.7}
        }
    ]
    
    print("📈 Processing scenarios and learning from feedback:")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n  Scenario {i}:")
        decision = await intelligence.make_decision(scenario["context"])
        print(f"    Decision: {decision.action} (confidence: {decision.confidence:.2f})")
        
        # Provide feedback
        feedback = {
            "decision_id": decision.decision_id,
            **scenario["feedback"]
        }
        
        learning_result = await intelligence.learn_from_feedback(feedback)
        print(f"    Learning: {learning_result.get('patterns_discovered', 0)} new patterns")
        print(f"    Adaptation: {learning_result.get('adaptations_applied', 0)} improvements")


async def main():
    """Main demonstration function."""
    print("🚀 Intelligence Layer Integration Demonstration")
    print("=" * 50)
    
    try:
        # Demonstrate intelligent email processing
        email_processor = IntelligentEmailProcessor()
        await email_processor.initialize()
        
        # Example email data
        sample_email = {
            "subject": "Critical System Alert: Database Performance Issue",
            "sender": "monitoring@company.com",
            "body": "Database response times have increased by 300% in the last hour...",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attachments": ["performance_report.pdf"]
        }
        
        await email_processor.process_email_with_intelligence(sample_email)
        
        # Demonstrate multi-agent coordination
        coordinator = IntelligentMultiAgentCoordinator()
        await coordinator.initialize()
        await coordinator.coordinate_complex_task()
        
        # Demonstrate learning and adaptation
        await demonstrate_learning_adaptation()
        
        print("\n✅ Intelligence Layer demonstration completed successfully!")
        print("\n📝 Key Features Demonstrated:")
        print("  - Intelligent decision-making with confidence scoring")
        print("  - Multi-agent task coordination and conflict resolution")
        print("  - Learning from feedback and continuous adaptation")
        print("  - Integration with existing agent architecture")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        print("💡 Ensure all dependencies are installed: pip install -r requirements.txt")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)