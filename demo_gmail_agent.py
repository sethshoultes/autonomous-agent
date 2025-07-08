#!/usr/bin/env python3
"""
Gmail Agent Demonstration Script

This script demonstrates the capabilities of the Gmail Agent within the
autonomous agent system. It showcases email fetching, classification,
automated responses, archiving, and integration with the framework.

Usage:
    python demo_gmail_agent.py [--config CONFIG_FILE] [--mock] [--verbose]

Arguments:
    --config: Path to configuration file (default: config/demo_config.json)
    --mock: Use mock Gmail service instead of real API
    --verbose: Enable verbose logging
"""

import argparse
import asyncio
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.gmail_agent import GmailAgent, EmailSummary
from agents.manager import AgentManager
from communication.broker import CommunicationBroker
from config.manager import ConfigManager
from logging.manager import LoggingManager

# Import test mocks for demonstration
from tests.mocks.gmail_mocks import (
    MockGmailService,
    MockGmailAPIContext,
    generate_sample_emails,
    generate_work_emails,
    generate_important_emails,
    generate_spam_emails,
    generate_newsletter_emails,
    generate_threaded_conversation,
)


class GmailAgentDemo:
    """Gmail Agent demonstration class."""
    
    def __init__(self, config_file: str, use_mock: bool = True, verbose: bool = False):
        """
        Initialize the demonstration.
        
        Args:
            config_file: Path to configuration file
            use_mock: Whether to use mock Gmail service
            verbose: Enable verbose logging
        """
        self.config_file = config_file
        self.use_mock = use_mock
        self.verbose = verbose
        
        # Components
        self.config_manager = None
        self.logging_manager = None
        self.communication_broker = None
        self.agent_manager = None
        self.gmail_agent = None
        self.mock_service = None
        self.logger = None
        
        # Demo data
        self.demo_emails = []
        self.demo_results = {}
    
    async def setup(self) -> None:
        """Set up the demonstration environment."""
        print("🚀 Setting up Gmail Agent Demonstration...")
        
        # Load configuration
        await self._setup_configuration()
        
        # Setup logging
        await self._setup_logging()
        
        # Setup communication
        await self._setup_communication()
        
        # Setup agent manager
        await self._setup_agent_manager()
        
        # Setup Gmail agent
        await self._setup_gmail_agent()
        
        # Setup demo data
        await self._setup_demo_data()
        
        print("✅ Setup complete!\n")
    
    async def _setup_configuration(self) -> None:
        """Setup configuration management."""
        self.config_manager = ConfigManager()
        
        if Path(self.config_file).exists():
            self.config_manager.load_config(self.config_file)
            print(f"📁 Loaded configuration from {self.config_file}")
        else:
            # Create demo configuration
            demo_config = self._create_demo_config()
            self.config_manager.load_config(demo_config)
            print("📁 Using default demo configuration")
    
    async def _setup_logging(self) -> None:
        """Setup logging management."""
        self.logging_manager = LoggingManager()
        logging_config = self.config_manager.get("logging", {})
        
        if self.verbose:
            logging_config["level"] = "DEBUG"
        
        self.logging_manager.configure(logging_config)
        self.logger = self.logging_manager.get_logger("gmail_demo")
        print("📋 Logging configured")
    
    async def _setup_communication(self) -> None:
        """Setup communication broker."""
        broker_config = self.config_manager.get("communication.message_broker", {})
        
        self.communication_broker = CommunicationBroker(
            queue_size=broker_config.get("queue_size", 1000),
            timeout=broker_config.get("timeout", 30.0),
            logger=self.logger
        )
        print("📡 Communication broker initialized")
    
    async def _setup_agent_manager(self) -> None:
        """Setup agent manager."""
        self.agent_manager = AgentManager(self.config_manager, self.logger)
        print("👥 Agent manager initialized")
    
    async def _setup_gmail_agent(self) -> None:
        """Setup Gmail agent."""
        config = self.config_manager.config
        
        self.gmail_agent = GmailAgent(
            agent_id="gmail_demo_agent",
            config=config,
            logger=self.logger,
            message_broker=self.communication_broker
        )
        
        # Register with agent manager
        await self.agent_manager.register_agent(self.gmail_agent)
        print("📧 Gmail agent created and registered")
    
    async def _setup_demo_data(self) -> None:
        """Setup demonstration data."""
        if self.use_mock:
            self.mock_service = MockGmailService()
            
            # Generate diverse email data
            sample_emails = generate_sample_emails(5)
            work_emails = generate_work_emails(5)
            important_emails = generate_important_emails(3)
            spam_emails = generate_spam_emails(3)
            newsletter_emails = generate_newsletter_emails(2)
            
            # Add all emails to mock service
            all_emails = sample_emails + work_emails + important_emails + spam_emails + newsletter_emails
            for email in all_emails:
                self.mock_service.add_message(email)
            
            self.demo_emails = all_emails
            print(f"📋 Generated {len(all_emails)} demo emails")
        else:
            print("📧 Using real Gmail API (ensure credentials are configured)")
    
    def _create_demo_config(self) -> Dict[str, Any]:
        """Create demonstration configuration."""
        return {
            "agent_manager": {
                "max_agents": 10,
                "heartbeat_interval": 30.0,
                "communication_timeout": 60.0,
                "retry_attempts": 3
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "level": "INFO"
                    }
                }
            },
            "communication": {
                "message_broker": {
                    "queue_size": 1000,
                    "timeout": 30.0
                }
            },
            "gmail": {
                "credentials_path": "/tmp/demo_credentials.json",
                "scopes": [
                    "https://www.googleapis.com/auth/gmail.readonly",
                    "https://www.googleapis.com/auth/gmail.send",
                    "https://www.googleapis.com/auth/gmail.modify"
                ],
                "user_email": "demo@example.com",
                "batch_size": 100,
                "rate_limit_per_minute": 250,
                "max_retries": 3,
                "retry_delay": 1.0,
                "classification": {
                    "enabled": True,
                    "spam_threshold": 0.8,
                    "importance_threshold": 0.7,
                    "categories": ["important", "spam", "personal", "work", "archive"],
                    "keywords": {
                        "important": ["urgent", "asap", "deadline", "priority", "critical"],
                        "spam": ["prize", "winner", "lottery", "click here", "free", "guarantee"],
                        "work": ["meeting", "project", "deadline", "team", "call", "conference"],
                        "personal": ["family", "friend", "personal", "dinner", "vacation"]
                    }
                },
                "auto_response": {
                    "enabled": True,
                    "response_delay": 300,
                    "max_responses_per_day": 50,
                    "templates": {
                        "out_of_office": "Thank you for your email. I'm currently out of office and will respond when I return.",
                        "meeting_request": "Thank you for the meeting request. I'll review my calendar and get back to you soon.",
                        "general_inquiry": "Thank you for your email. I'll respond as soon as possible."
                    },
                    "trigger_patterns": {
                        "out_of_office": ["vacation", "out of office", "unavailable"],
                        "meeting_request": ["meeting", "call", "appointment", "schedule"],
                        "general_inquiry": ["question", "inquiry", "help", "support"]
                    }
                },
                "archiving": {
                    "enabled": True,
                    "archive_after_days": 30,
                    "auto_label": True,
                    "label_rules": [
                        {"pattern": "newsletter", "label": "Newsletters"},
                        {"pattern": "noreply", "label": "Automated"},
                        {"pattern": "github", "label": "GitHub"},
                        {"pattern": "linkedin", "label": "LinkedIn"}
                    ],
                    "smart_folders": {
                        "receipts": ["receipt", "invoice", "purchase", "order"],
                        "travel": ["flight", "hotel", "booking", "reservation"],
                        "social": ["facebook", "twitter", "instagram", "social"]
                    }
                }
            },
            "agents": {
                "gmail_demo_agent": {
                    "agent_type": "gmail",
                    "enabled": True,
                    "priority": 1,
                    "config": {}
                }
            }
        }
    
    async def run_demonstration(self) -> None:
        """Run the complete Gmail Agent demonstration."""
        print("🎬 Starting Gmail Agent Demonstration\n")
        
        try:
            # Setup environment
            await self.setup()
            
            # Start the agent system
            await self._start_system()
            
            # Run demonstration scenarios
            await self._demo_email_fetching()
            await self._demo_email_classification()
            await self._demo_auto_responses()
            await self._demo_email_archiving()
            await self._demo_email_summarization()
            await self._demo_inter_agent_communication()
            await self._demo_performance_metrics()
            
            # Display results
            await self._display_results()
            
        except Exception as e:
            print(f"❌ Demonstration failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
        
        finally:
            # Cleanup
            await self._cleanup()
    
    async def _start_system(self) -> None:
        """Start the agent system."""
        print("🟢 Starting agent system...")
        
        if self.use_mock:
            with MockGmailAPIContext(self.mock_service):
                await self.agent_manager.start_agents()
        else:
            await self.agent_manager.start_agents()
        
        # Verify agent is active
        status = self.agent_manager.get_agent_status("gmail_demo_agent")
        if status["state"] == "active":
            print("✅ Gmail agent is active and ready")
        else:
            raise Exception(f"Gmail agent failed to start: {status}")
    
    async def _demo_email_fetching(self) -> None:
        """Demonstrate email fetching capabilities."""
        print("\n📥 Demonstrating Email Fetching")
        print("=" * 50)
        
        try:
            if self.use_mock:
                with MockGmailAPIContext(self.mock_service):
                    # Fetch all emails
                    all_emails = await self.gmail_agent._fetch_emails(max_results=20)
                    print(f"📧 Fetched {len(all_emails)} total emails")
                    
                    # Fetch unread emails
                    unread_emails = await self.gmail_agent._fetch_emails(
                        query="is:unread",
                        max_results=10
                    )
                    print(f"📬 Found {len(unread_emails)} unread emails")
                    
                    # Fetch emails with specific labels
                    inbox_emails = await self.gmail_agent._fetch_emails(
                        label_ids=["INBOX"],
                        max_results=10
                    )
                    print(f"📮 Found {len(inbox_emails)} emails in inbox")
                    
                    self.demo_results["email_fetching"] = {
                        "total_emails": len(all_emails),
                        "unread_emails": len(unread_emails),
                        "inbox_emails": len(inbox_emails)
                    }
            else:
                print("📧 Real Gmail API fetching would occur here")
                self.demo_results["email_fetching"] = {"status": "skipped_real_api"}
            
            print("✅ Email fetching demonstration complete")
            
        except Exception as e:
            print(f"❌ Email fetching failed: {e}")
            self.demo_results["email_fetching"] = {"error": str(e)}
    
    async def _demo_email_classification(self) -> None:
        """Demonstrate email classification capabilities."""
        print("\n🏷️  Demonstrating Email Classification")
        print("=" * 50)
        
        try:
            if self.use_mock:
                with MockGmailAPIContext(self.mock_service):
                    emails = await self.gmail_agent._fetch_emails(max_results=10)
                    
                    classifications = {}
                    for email in emails[:8]:  # Classify first 8 emails
                        classification = await self.gmail_agent._classify_email(email)
                        category = classification.category
                        
                        if category not in classifications:
                            classifications[category] = []
                        
                        classifications[category].append({
                            "subject": email.get("subject", ""),
                            "from": email.get("from", ""),
                            "confidence": classification.confidence,
                            "keywords": classification.keywords
                        })
                    
                    # Display classification results
                    for category, emails_in_category in classifications.items():
                        print(f"\n📂 {category.upper()} ({len(emails_in_category)} emails):")
                        for email_info in emails_in_category[:3]:  # Show first 3
                            print(f"  • {email_info['subject'][:50]}... (confidence: {email_info['confidence']:.2f})")
                    
                    self.demo_results["email_classification"] = {
                        "categories": {cat: len(emails) for cat, emails in classifications.items()},
                        "total_classified": sum(len(emails) for emails in classifications.values())
                    }
            else:
                print("🏷️  Real Gmail API classification would occur here")
                self.demo_results["email_classification"] = {"status": "skipped_real_api"}
            
            print("✅ Email classification demonstration complete")
            
        except Exception as e:
            print(f"❌ Email classification failed: {e}")
            self.demo_results["email_classification"] = {"error": str(e)}
    
    async def _demo_auto_responses(self) -> None:
        """Demonstrate automated response capabilities."""
        print("\n🤖 Demonstrating Automated Responses")
        print("=" * 50)
        
        try:
            if self.use_mock:
                with MockGmailAPIContext(self.mock_service):
                    # Get some emails to test auto-responses
                    emails = await self.gmail_agent._fetch_emails(max_results=5)
                    
                    responses_generated = 0
                    response_types = {}
                    
                    for email in emails:
                        # Generate auto response
                        response = await self.gmail_agent._generate_auto_response(email)
                        
                        if response:
                            responses_generated += 1
                            
                            # Determine response type based on content
                            response_body = response.get("body", "").lower()
                            if "meeting" in response_body:
                                response_type = "meeting_request"
                            elif "out of office" in response_body:
                                response_type = "out_of_office"
                            else:
                                response_type = "general_inquiry"
                            
                            response_types[response_type] = response_types.get(response_type, 0) + 1
                            
                            print(f"📤 Generated {response_type} response for: {email.get('subject', '')[:40]}...")
                    
                    print(f"\n📊 Auto-response summary:")
                    print(f"  • Total responses generated: {responses_generated}")
                    for resp_type, count in response_types.items():
                        print(f"  • {resp_type}: {count}")
                    
                    self.demo_results["auto_responses"] = {
                        "total_generated": responses_generated,
                        "response_types": response_types
                    }
            else:
                print("🤖 Real Gmail API auto-responses would occur here")
                self.demo_results["auto_responses"] = {"status": "skipped_real_api"}
            
            print("✅ Automated responses demonstration complete")
            
        except Exception as e:
            print(f"❌ Automated responses failed: {e}")
            self.demo_results["auto_responses"] = {"error": str(e)}
    
    async def _demo_email_archiving(self) -> None:
        """Demonstrate email archiving and organization capabilities."""
        print("\n📁 Demonstrating Email Archiving & Organization")
        print("=" * 50)
        
        try:
            if self.use_mock:
                with MockGmailAPIContext(self.mock_service):
                    emails = await self.gmail_agent._fetch_emails(max_results=8)
                    
                    label_assignments = {}
                    smart_folder_assignments = {}
                    
                    for email in emails:
                        # Determine labels
                        labels = await self.gmail_agent._determine_labels(email)
                        if labels:
                            for label in labels:
                                label_assignments[label] = label_assignments.get(label, 0) + 1
                        
                        # Assign smart folder
                        smart_folder = await self.gmail_agent._assign_smart_folder(email)
                        if smart_folder:
                            smart_folder_assignments[smart_folder] = smart_folder_assignments.get(smart_folder, 0) + 1
                        
                        print(f"📝 {email.get('subject', '')[:40]}...")
                        if labels:
                            print(f"    Labels: {', '.join(labels)}")
                        if smart_folder:
                            print(f"    Smart folder: {smart_folder}")
                    
                    print(f"\n📊 Organization summary:")
                    print("  Label assignments:")
                    for label, count in label_assignments.items():
                        print(f"    • {label}: {count} emails")
                    
                    print("  Smart folder assignments:")
                    for folder, count in smart_folder_assignments.items():
                        print(f"    • {folder}: {count} emails")
                    
                    self.demo_results["email_archiving"] = {
                        "label_assignments": label_assignments,
                        "smart_folder_assignments": smart_folder_assignments
                    }
            else:
                print("📁 Real Gmail API archiving would occur here")
                self.demo_results["email_archiving"] = {"status": "skipped_real_api"}
            
            print("✅ Email archiving demonstration complete")
            
        except Exception as e:
            print(f"❌ Email archiving failed: {e}")
            self.demo_results["email_archiving"] = {"error": str(e)}
    
    async def _demo_email_summarization(self) -> None:
        """Demonstrate email summarization capabilities."""
        print("\n📊 Demonstrating Email Summarization")
        print("=" * 50)
        
        try:
            if self.use_mock:
                with MockGmailAPIContext(self.mock_service):
                    # Generate email summary
                    summary = await self.gmail_agent._generate_email_summary("last_24_hours")
                    
                    print(f"📈 Email Summary for {summary.time_range}:")
                    print(f"  • Total emails: {summary.total_emails}")
                    print(f"  • Unread emails: {summary.unread_emails}")
                    print(f"  • Important emails: {summary.important_emails}")
                    print(f"  • Spam emails: {summary.spam_emails}")
                    
                    print("\n📂 Categories breakdown:")
                    for category, count in summary.categories.items():
                        if count > 0:
                            print(f"  • {category}: {count}")
                    
                    print("\n👥 Top senders:")
                    for sender, count in list(summary.senders.items())[:5]:
                        print(f"  • {sender}: {count} emails")
                    
                    self.demo_results["email_summarization"] = summary.to_dict()
            else:
                print("📊 Real Gmail API summarization would occur here")
                self.demo_results["email_summarization"] = {"status": "skipped_real_api"}
            
            print("✅ Email summarization demonstration complete")
            
        except Exception as e:
            print(f"❌ Email summarization failed: {e}")
            self.demo_results["email_summarization"] = {"error": str(e)}
    
    async def _demo_inter_agent_communication(self) -> None:
        """Demonstrate inter-agent communication capabilities."""
        print("\n🔄 Demonstrating Inter-Agent Communication")
        print("=" * 50)
        
        try:
            from agents.base import AgentMessage
            
            # Simulate message from scheduler agent
            fetch_request = AgentMessage(
                id="demo_msg_001",
                sender="scheduler_agent",
                recipient="gmail_demo_agent",
                message_type="fetch_emails",
                payload={
                    "query": "is:unread",
                    "max_results": 5,
                    "label_ids": ["INBOX"]
                }
            )
            
            print("📤 Sending fetch_emails request to Gmail agent...")
            
            if self.use_mock:
                with MockGmailAPIContext(self.mock_service):
                    response = await self.gmail_agent._process_message(fetch_request)
                    
                    if response:
                        print(f"📥 Received response: {response.message_type}")
                        print(f"    Email count: {response.payload.get('count', 0)}")
                        
                        # Test classification request
                        if response.payload.get("emails"):
                            first_email = response.payload["emails"][0]
                            classify_request = AgentMessage(
                                id="demo_msg_002",
                                sender="scheduler_agent",
                                recipient="gmail_demo_agent",
                                message_type="classify_email",
                                payload={"email_data": first_email}
                            )
                            
                            print("📤 Sending classify_email request...")
                            classify_response = await self.gmail_agent._process_message(classify_request)
                            
                            if classify_response:
                                classification = classify_response.payload.get("classification", {})
                                print(f"📥 Classification result: {classification.get('category', 'unknown')} "
                                     f"(confidence: {classification.get('confidence', 0):.2f})")
                        
                        self.demo_results["inter_agent_communication"] = {
                            "messages_processed": 2,
                            "fetch_response": response.payload.get('count', 0),
                            "classification_successful": True
                        }
                    else:
                        print("❌ No response received")
                        self.demo_results["inter_agent_communication"] = {"error": "no_response"}
            else:
                print("🔄 Real agent communication would occur here")
                self.demo_results["inter_agent_communication"] = {"status": "skipped_real_api"}
            
            print("✅ Inter-agent communication demonstration complete")
            
        except Exception as e:
            print(f"❌ Inter-agent communication failed: {e}")
            self.demo_results["inter_agent_communication"] = {"error": str(e)}
    
    async def _demo_performance_metrics(self) -> None:
        """Demonstrate performance metrics collection."""
        print("\n📊 Demonstrating Performance Metrics")
        print("=" * 50)
        
        try:
            # Get current metrics
            metrics = self.gmail_agent.get_metrics()
            
            print("📈 Gmail Agent Metrics:")
            print(f"  • Emails processed: {metrics.get('emails_processed', 0)}")
            print(f"  • Classifications made: {metrics.get('classifications_made', 0)}")
            print(f"  • Auto responses sent: {metrics.get('auto_responses_sent', 0)}")
            print(f"  • Emails archived: {metrics.get('emails_archived', 0)}")
            print(f"  • API calls made: {metrics.get('api_calls_made', 0)}")
            print(f"  • Rate limit hits: {metrics.get('rate_limit_hits', 0)}")
            
            print("\n🔧 Framework Metrics:")
            print(f"  • Messages processed: {metrics.get('messages_processed', 0)}")
            print(f"  • Tasks completed: {metrics.get('tasks_completed', 0)}")
            print(f"  • Errors: {metrics.get('errors', 0)}")
            print(f"  • Uptime: {metrics.get('uptime', 0):.2f} seconds")
            print(f"  • State: {metrics.get('state', 'unknown')}")
            
            # Calculate performance ratios
            total_operations = (metrics.get('emails_processed', 0) + 
                              metrics.get('classifications_made', 0) + 
                              metrics.get('auto_responses_sent', 0))
            
            if total_operations > 0:
                error_rate = metrics.get('errors', 0) / total_operations * 100
                print(f"\n⚡ Performance Summary:")
                print(f"  • Total operations: {total_operations}")
                print(f"  • Error rate: {error_rate:.2f}%")
                print(f"  • Operations per second: {total_operations / max(metrics.get('uptime', 1), 1):.2f}")
            
            self.demo_results["performance_metrics"] = metrics
            
            print("✅ Performance metrics demonstration complete")
            
        except Exception as e:
            print(f"❌ Performance metrics failed: {e}")
            self.demo_results["performance_metrics"] = {"error": str(e)}
    
    async def _display_results(self) -> None:
        """Display comprehensive demonstration results."""
        print("\n" + "=" * 70)
        print("🎯 GMAIL AGENT DEMONSTRATION RESULTS")
        print("=" * 70)
        
        for section, results in self.demo_results.items():
            print(f"\n📋 {section.replace('_', ' ').title()}:")
            
            if isinstance(results, dict):
                if "error" in results:
                    print(f"  ❌ Error: {results['error']}")
                elif "status" in results and results["status"] == "skipped_real_api":
                    print(f"  ⏭️  Skipped (real API not configured)")
                else:
                    for key, value in results.items():
                        if isinstance(value, dict):
                            print(f"  • {key}:")
                            for subkey, subvalue in value.items():
                                print(f"    - {subkey}: {subvalue}")
                        else:
                            print(f"  • {key}: {value}")
            else:
                print(f"  • Result: {results}")
        
        print("\n" + "=" * 70)
        print("✅ Demonstration completed successfully!")
        print("=" * 70)
    
    async def _cleanup(self) -> None:
        """Cleanup demonstration resources."""
        print("\n🧹 Cleaning up...")
        
        try:
            if self.agent_manager:
                await self.agent_manager.stop_agents()
                print("✅ Agents stopped")
            
            if self.communication_broker:
                # Close broker connections if applicable
                pass
            
            print("✅ Cleanup complete")
            
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


async def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Gmail Agent Demonstration")
    parser.add_argument(
        "--config",
        default="config/demo_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock Gmail service (default: True)"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real Gmail API (requires credentials)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Determine whether to use mock or real API
    use_mock = not args.real
    
    print("📧 Gmail Agent Demonstration")
    print("=" * 50)
    print(f"Configuration: {args.config}")
    print(f"Mode: {'Mock' if use_mock else 'Real'} Gmail API")
    print(f"Verbose: {args.verbose}")
    
    if not use_mock:
        print("\n⚠️  Using real Gmail API requires:")
        print("  • Valid Gmail API credentials")
        print("  • Proper OAuth2 setup")
        print("  • Internet connection")
        
        confirm = input("\nContinue with real API? (y/N): ")
        if confirm.lower() != 'y':
            print("Switching to mock mode...")
            use_mock = True
    
    # Run demonstration
    demo = GmailAgentDemo(
        config_file=args.config,
        use_mock=use_mock,
        verbose=args.verbose
    )
    
    await demo.run_demonstration()


if __name__ == "__main__":
    asyncio.run(main())