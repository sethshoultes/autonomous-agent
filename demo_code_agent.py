#!/usr/bin/env python3
"""
Code Agent Demo Script

This script demonstrates the comprehensive GitHub integration and AI-powered
code review capabilities of the Code Agent, including repository monitoring,
automated pull request analysis, security vulnerability detection, and
development workflow automation.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.code_agent import CodeAgent
from agents.manager import AgentManager, AgentConfig
from communication.broker import MessageBroker
from config.manager import ConfigManager
from logging.manager import LoggingManager


class CodeAgentDemo:
    """Demo class for Code Agent functionality."""
    
    def __init__(self):
        """Initialize the demo."""
        self.logger = None
        self.agent_manager = None
        self.code_agent = None
        self.message_broker = None
    
    async def setup(self) -> None:
        """Setup demo environment."""
        print("🚀 Setting up Code Agent Demo...")
        
        # Initialize logging
        logging_config = {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "handlers": {
                "console": {"enabled": True},
                "file": {"enabled": False}
            }
        }
        
        logging_manager = LoggingManager(logging_config)
        await logging_manager.setup()
        self.logger = logging_manager.get_logger("code_agent_demo")
        
        # Load configuration
        config_path = Path(__file__).parent / "config" / "code_agent_config.yaml"
        if not config_path.exists():
            # Use sample configuration if config file doesn't exist
            config_path = Path(__file__).parent / "config" / "sample_ollama_config.yaml"
        
        config_manager = ConfigManager()
        await config_manager.load_config(str(config_path))
        config = config_manager.get_config()
        
        # Check for required environment variables
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            print("⚠️  Warning: GITHUB_TOKEN environment variable not set.")
            print("   Code Agent will use mock GitHub service for demo.")
            config["agents"]["code_agent"]["config"]["github"]["token"] = "demo_token"
        
        # Initialize message broker
        self.message_broker = MessageBroker(config.get("communication", {}), self.logger)
        await self.message_broker.start()
        
        # Initialize agent manager
        self.agent_manager = AgentManager(
            config.get("agent_manager", {}),
            self.logger,
            self.message_broker
        )
        await self.agent_manager.start()
        
        # Create and register Code Agent
        code_agent_config = AgentConfig(
            agent_id="demo_code_agent",
            agent_type="CodeAgent",
            config=config["agents"]["code_agent"]["config"],
            enabled=True,
            priority=1
        )
        
        self.code_agent = CodeAgent(
            agent_id="demo_code_agent",
            config=config["agents"]["code_agent"]["config"],
            logger=self.logger,
            message_broker=self.message_broker
        )
        
        await self.agent_manager.register_agent(self.code_agent, code_agent_config)
        await self.agent_manager.start_agent("demo_code_agent")
        
        print("✅ Code Agent Demo setup complete!")
    
    async def demo_code_analysis(self) -> None:
        """Demonstrate code analysis capabilities."""
        print("\n📊 Code Analysis Demo")
        print("=" * 50)
        
        # Sample vulnerable Python code
        vulnerable_code = '''
def authenticate_user(username, password):
    # Vulnerable: SQL injection
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    result = execute_query(query)
    
    # Vulnerable: Command injection
    import os
    os.system("echo " + username)
    
    # Performance issue: inefficient loop
    users = []
    for i in range(10000):
        users.append({"id": i, "name": f"user_{i}"})
    
    return result

def insecure_file_upload(filename, content):
    # Vulnerable: Path traversal
    file_path = "/uploads/" + filename
    with open(file_path, "w") as f:
        f.write(content)
    
    return file_path
        '''
        
        try:
            # Create code analysis task
            task = {
                "type": "code_analysis",
                "code": vulnerable_code,
                "language": "python",
                "analysis_type": "comprehensive"
            }
            
            print("🔍 Analyzing vulnerable Python code...")
            result = await self.code_agent.execute_task(task)
            
            if result.get("success"):
                print("✅ Code analysis completed!")
                print(f"📈 Overall Score: {result.get('overall_score', 'N/A')}")
                print(f"🔍 Files Analyzed: {result.get('files_analyzed', 1)}")
                print(f"⚠️  Issues Found: {result.get('issues_found', 0)}")
                
                # Display security issues
                security_issues = result.get("security_issues", [])
                if security_issues:
                    print(f"\n🔒 Security Issues ({len(security_issues)}):")
                    for i, issue in enumerate(security_issues[:3], 1):
                        print(f"  {i}. {issue.get('type', 'Unknown')} (Line {issue.get('line', 'N/A')})")
                        print(f"     Severity: {issue.get('severity', 'Unknown')}")
                        print(f"     Description: {issue.get('description', 'No description')}")
                
                # Display performance issues
                performance_issues = result.get("performance_issues", [])
                if performance_issues:
                    print(f"\n⚡ Performance Issues ({len(performance_issues)}):")
                    for i, issue in enumerate(performance_issues[:2], 1):
                        print(f"  {i}. {issue.get('type', 'Unknown')} (Line {issue.get('line', 'N/A')})")
                        print(f"     Description: {issue.get('description', 'No description')}")
                
                # Display recommendations
                recommendations = result.get("recommendations", [])
                if recommendations:
                    print(f"\n💡 Recommendations ({len(recommendations)}):")
                    for i, rec in enumerate(recommendations[:3], 1):
                        print(f"  {i}. {rec}")
            else:
                print(f"❌ Code analysis failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during code analysis: {e}")
    
    async def demo_vulnerability_scan(self) -> None:
        """Demonstrate vulnerability scanning capabilities."""
        print("\n🔐 Vulnerability Scanning Demo")
        print("=" * 50)
        
        try:
            # Create vulnerability scan task
            task = {
                "type": "vulnerability_scan",
                "repository": "demo/vulnerable-app",
                "branch": "main",
                "scan_type": "full",
                "include_dependencies": True
            }
            
            print("🔍 Performing security vulnerability scan...")
            result = await self.code_agent.execute_task(task)
            
            if result.get("success"):
                print("✅ Vulnerability scan completed!")
                print(f"🆔 Scan ID: {result.get('scan_id', 'N/A')}")
                print(f"🎯 Repository: {result.get('repository', 'N/A')}")
                print(f"🌿 Branch: {result.get('branch', 'N/A')}")
                
                # Display vulnerability summary
                total_vulns = result.get("total_vulnerabilities", 0)
                critical_vulns = result.get("critical_vulnerabilities", 0)
                high_vulns = result.get("high_vulnerabilities", 0)
                medium_vulns = result.get("medium_vulnerabilities", 0)
                low_vulns = result.get("low_vulnerabilities", 0)
                
                print(f"\n📊 Vulnerability Summary:")
                print(f"  Total: {total_vulns}")
                print(f"  Critical: {critical_vulns}")
                print(f"  High: {high_vulns}")
                print(f"  Medium: {medium_vulns}")
                print(f"  Low: {low_vulns}")
                print(f"  Risk Score: {result.get('risk_score', 0):.1f}/10")
                
                # Display top vulnerabilities
                vulnerabilities = result.get("vulnerabilities", [])
                if vulnerabilities:
                    print(f"\n🚨 Top Vulnerabilities:")
                    for i, vuln in enumerate(vulnerabilities[:3], 1):
                        print(f"  {i}. {vuln.get('type', 'Unknown')} - {vuln.get('severity', 'Unknown')}")
                        print(f"     Line: {vuln.get('line', 'N/A')}")
                        print(f"     Impact: {vuln.get('impact', 'Unknown')}")
                        print(f"     Fix: {vuln.get('remediation', 'No fix available')}")
                
                # Display dependency issues
                dependencies = result.get("dependencies", {})
                dep_vulns = dependencies.get("vulnerable_packages", [])
                if dep_vulns:
                    print(f"\n📦 Dependency Vulnerabilities ({len(dep_vulns)}):")
                    for i, dep in enumerate(dep_vulns[:2], 1):
                        print(f"  {i}. {dep.get('package', 'Unknown')} v{dep.get('version', 'Unknown')}")
                        print(f"     CVE: {dep.get('vulnerability', 'Unknown')}")
                        print(f"     Severity: {dep.get('severity', 'Unknown')}")
            else:
                print(f"❌ Vulnerability scan failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during vulnerability scan: {e}")
    
    async def demo_pull_request_review(self) -> None:
        """Demonstrate pull request review capabilities."""
        print("\n🔍 Pull Request Review Demo")
        print("=" * 50)
        
        try:
            # Create PR review task with mock data
            task = {
                "type": "code_review",
                "repository": "demo/example-repo",
                "pull_request": 42,
                "review_type": "full",
                "focus_areas": ["security", "performance", "style"]
            }
            
            print("📝 Reviewing pull request...")
            result = await self.code_agent.execute_task(task)
            
            if result.get("success"):
                print("✅ Pull request review completed!")
                print(f"🆔 Review ID: {result.get('review_id', 'N/A')}")
                print(f"📊 Overall Score: {result.get('score', 'N/A')}/10")
                print(f"📁 Files Analyzed: {result.get('files_analyzed', 0)}")
                print(f"💬 Comments Generated: {len(result.get('comments', []))}")
                
                # Display review assessment
                assessment = result.get("overall_assessment", "comment")
                print(f"🎯 Assessment: {assessment.upper()}")
                
                # Display review comments
                comments = result.get("comments", [])
                if comments:
                    print(f"\n💬 Review Comments:")
                    for i, comment in enumerate(comments[:3], 1):
                        print(f"  {i}. File: {comment.get('file', 'Unknown')}")
                        print(f"     Line: {comment.get('line', 'N/A')}")
                        print(f"     Type: {comment.get('type', 'Unknown')}")
                        print(f"     Message: {comment.get('message', 'No message')}")
                        print(f"     Confidence: {comment.get('confidence', 0):.2f}")
                
                # Display security improvements
                security_improvements = result.get("security_improvements", [])
                if security_improvements:
                    print(f"\n🔒 Security Improvements:")
                    for i, improvement in enumerate(security_improvements[:2], 1):
                        print(f"  {i}. {improvement}")
                
                # Display testing suggestions
                testing_suggestions = result.get("testing_suggestions", [])
                if testing_suggestions:
                    print(f"\n🧪 Testing Suggestions:")
                    for i, suggestion in enumerate(testing_suggestions[:2], 1):
                        print(f"  {i}. {suggestion}")
            else:
                print(f"❌ Pull request review failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during pull request review: {e}")
    
    async def demo_documentation_generation(self) -> None:
        """Demonstrate documentation generation capabilities."""
        print("\n📚 Documentation Generation Demo")
        print("=" * 50)
        
        # Sample code for documentation
        sample_code = '''
class UserManager:
    """Manages user accounts and authentication."""
    
    def __init__(self, database_url: str):
        """Initialize the UserManager.
        
        Args:
            database_url: URL of the database connection
        """
        self.db_url = database_url
        self.users = {}
    
    def create_user(self, username: str, email: str, password: str) -> dict:
        """Create a new user account.
        
        Args:
            username: Unique username for the account
            email: User's email address
            password: Password for the account
            
        Returns:
            Dictionary containing user information
            
        Raises:
            ValueError: If username already exists
        """
        if username in self.users:
            raise ValueError(f"Username {username} already exists")
        
        user_data = {
            "username": username,
            "email": email,
            "created_at": datetime.now(),
            "active": True
        }
        
        self.users[username] = user_data
        return user_data
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate a user login attempt.
        
        Args:
            username: Username to authenticate
            password: Password to verify
            
        Returns:
            True if authentication successful, False otherwise
        """
        # Implementation would verify password hash
        return username in self.users
        '''
        
        try:
            # Create documentation generation task
            task = {
                "type": "generate_documentation",
                "repository": "demo/example-repo",
                "files": ["user_manager.py"],
                "doc_type": "api",
                "format": "markdown",
                "create_files": False
            }
            
            print("📝 Generating API documentation...")
            result = await self.code_agent.execute_task(task)
            
            if result.get("success"):
                print("✅ Documentation generation completed!")
                print(f"🆔 Doc ID: {result.get('doc_id', 'N/A')}")
                print(f"📁 Files Processed: {result.get('files_processed', 0)}")
                print(f"📄 Format: {result.get('format', 'N/A')}")
                
                # Display documentation structure
                documentation = result.get("documentation", {})
                functions = documentation.get("functions", [])
                classes = documentation.get("classes", [])
                
                print(f"\n📊 Documentation Structure:")
                print(f"  Functions: {len(functions)}")
                print(f"  Classes: {len(classes)}")
                
                # Display sample function documentation
                if functions:
                    print(f"\n🔧 Sample Function Documentation:")
                    func = functions[0]
                    print(f"  Name: {func.get('name', 'Unknown')}")
                    print(f"  Description: {func.get('description', 'No description')}")
                    
                    params = func.get("parameters", [])
                    if params:
                        print(f"  Parameters ({len(params)}):")
                        for param in params[:2]:
                            print(f"    - {param.get('name', 'Unknown')} ({param.get('type', 'Unknown')})")
                
                # Display sample class documentation
                if classes:
                    print(f"\n📦 Sample Class Documentation:")
                    cls = classes[0]
                    print(f"  Name: {cls.get('name', 'Unknown')}")
                    print(f"  Description: {cls.get('description', 'No description')}")
                    
                    methods = cls.get("methods", [])
                    if methods:
                        print(f"  Methods ({len(methods)}):")
                        for method in methods[:2]:
                            print(f"    - {method.get('name', 'Unknown')}")
                
                # Display markdown preview
                markdown = result.get("markdown_output", "")
                if markdown:
                    print(f"\n📄 Markdown Preview (first 300 chars):")
                    print(f"  {markdown[:300]}...")
            else:
                print(f"❌ Documentation generation failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during documentation generation: {e}")
    
    async def demo_repository_monitoring(self) -> None:
        """Demonstrate repository monitoring capabilities."""
        print("\n👁️  Repository Monitoring Demo")
        print("=" * 50)
        
        try:
            # Add repository to monitoring
            monitor_task = {
                "type": "monitor_repository",
                "repository": "demo/monitored-repo",
                "action": "add"
            }
            
            print("➕ Adding repository to monitoring...")
            result = await self.code_agent.execute_task(monitor_task)
            
            if result.get("success"):
                print("✅ Repository added to monitoring!")
                print(f"📦 Repository: {result.get('repository', 'N/A')}")
                print(f"📊 Total Monitored: {result.get('total_monitored', 0)}")
                
                # Check monitoring status
                status_task = {
                    "type": "monitor_repository",
                    "repository": "demo/monitored-repo",
                    "action": "status"
                }
                
                print("\n📊 Checking monitoring status...")
                status_result = await self.code_agent.execute_task(status_task)
                
                if status_result.get("success"):
                    status = status_result.get("status", {})
                    print(f"  Status: {status.get('enabled', 'Unknown')}")
                    print(f"  Events: {', '.join(status.get('events', []))}")
                    print(f"  Auto Review: {status.get('auto_review', 'Unknown')}")
                    print(f"  Security Scan: {status.get('auto_scan_security', 'Unknown')}")
                
                # Display monitoring features
                print(f"\n🔧 Monitoring Features:")
                print(f"  • Automatic PR reviews")
                print(f"  • Security vulnerability scanning")
                print(f"  • Issue analysis and categorization")
                print(f"  • Workflow automation triggers")
                print(f"  • Real-time event processing")
                print(f"  • Intelligent notifications")
            else:
                print(f"❌ Failed to add repository to monitoring: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during repository monitoring demo: {e}")
    
    async def demo_agent_metrics(self) -> None:
        """Display Code Agent metrics and statistics."""
        print("\n📈 Code Agent Metrics")
        print("=" * 50)
        
        try:
            # Get agent metrics
            metrics = self.code_agent.get_metrics()
            
            print("📊 Performance Metrics:")
            print(f"  Uptime: {metrics.get('uptime', 0):.1f} seconds")
            print(f"  Messages Processed: {metrics.get('messages_processed', 0)}")
            print(f"  Tasks Completed: {metrics.get('tasks_completed', 0)}")
            print(f"  Errors: {metrics.get('errors', 0)}")
            
            print(f"\n🔍 Code Analysis Metrics:")
            print(f"  Pull Requests Reviewed: {metrics.get('pull_requests_reviewed', 0)}")
            print(f"  Vulnerabilities Detected: {metrics.get('vulnerabilities_detected', 0)}")
            print(f"  Documentation Generated: {metrics.get('documentation_generated', 0)}")
            print(f"  Repositories Monitored: {metrics.get('repositories_monitored', 0)}")
            print(f"  Webhooks Processed: {metrics.get('webhooks_processed', 0)}")
            
            print(f"\n🤖 Agent Status:")
            print(f"  State: {metrics.get('state', 'Unknown')}")
            print(f"  Agent ID: {metrics.get('agent_id', 'Unknown')}")
            print(f"  Active Reviews: {metrics.get('active_reviews', 0)}")
            print(f"  Webhook Queue Size: {metrics.get('webhook_queue_size', 0)}")
            
            # Display health check
            health = await self.code_agent.health_check()
            print(f"\n💚 Health Status: {'Healthy' if health else 'Unhealthy'}")
            
        except Exception as e:
            print(f"❌ Error getting agent metrics: {e}")
    
    async def run_demo(self) -> None:
        """Run the complete Code Agent demo."""
        try:
            await self.setup()
            
            print("\n🎯 Code Agent Demo - Comprehensive GitHub Integration & AI-Powered Development")
            print("=" * 80)
            print("This demo showcases the Code Agent's capabilities for:")
            print("• AI-powered code analysis and review")
            print("• Security vulnerability detection") 
            print("• Automated pull request reviews")
            print("• Documentation generation")
            print("• Repository monitoring and automation")
            print("• Development workflow integration")
            
            # Run all demo scenarios
            await self.demo_code_analysis()
            await self.demo_vulnerability_scan()
            await self.demo_pull_request_review()
            await self.demo_documentation_generation()
            await self.demo_repository_monitoring()
            await self.demo_agent_metrics()
            
            print("\n✅ Code Agent Demo Completed Successfully!")
            print("\n🚀 Next Steps:")
            print("1. Set GITHUB_TOKEN environment variable for real GitHub integration")
            print("2. Configure repository monitoring for your projects")
            print("3. Set up webhooks for real-time event processing")
            print("4. Customize AI models and analysis criteria")
            print("5. Integrate with CI/CD pipelines for automated quality gates")
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def cleanup(self) -> None:
        """Cleanup demo resources."""
        try:
            if self.agent_manager:
                await self.agent_manager.stop()
            if self.message_broker:
                await self.message_broker.stop()
            print("\n🧹 Demo cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")


async def main():
    """Main demo function."""
    print("🤖 Code Agent Demo - AI-Powered Development Assistant")
    print("=" * 60)
    
    demo = CodeAgentDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())