#!/usr/bin/env python3
"""
Demonstration script for the ResearchAgent functionality.
Shows how to integrate the ResearchAgent with the AgentManager and perform research tasks.
"""

import asyncio
import json
import logging
from unittest.mock import MagicMock

from src.agents import AgentManager, AgentConfig, ResearchAgent, ResearchTask
from src.communication.broker import MessageBroker  # Placeholder


async def demo_research_agent():
    """Demonstrate ResearchAgent functionality."""
    print("=== Research Agent Demonstration ===\n")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("demo")
    
    # Mock message broker for demonstration
    message_broker = MagicMock()
    message_broker.publish = lambda msg: print(f"📤 Message sent: {msg.message_type}")
    message_broker.disconnect = lambda: None
    
    # Configuration for the ResearchAgent
    research_config = {
        "max_concurrent_requests": 3,
        "request_timeout": 10,
        "rate_limit_delay": 0.5,
        "cache_ttl": 1800,  # 30 minutes
        "user_agent": "AutonomousAgent/1.0 Research Bot Demo",
        "respect_robots_txt": True,
        "max_content_length": 512 * 1024,  # 512KB for demo
        "relevance_threshold": 0.5,
        "deduplication_threshold": 0.7,
        "max_feed_items": 50,
        "research_topics": ["artificial intelligence", "machine learning", "automation"],
        "rss_feeds": [
            "https://feeds.feedburner.com/oreilly/radar",
            "https://techcrunch.com/feed/",
        ],
    }
    
    # Create AgentManager
    manager_config = {
        "max_agents": 10,
        "heartbeat_interval": 60,
        "communication_timeout": 30,
    }
    
    agent_manager = AgentManager(
        config=manager_config,
        logger=logger,
        message_broker=message_broker
    )
    
    try:
        # Start the agent manager
        await agent_manager.start()
        print("✅ AgentManager started successfully")
        
        # Create and register ResearchAgent
        research_agent = ResearchAgent(
            agent_id="research_agent_demo",
            config=research_config,
            logger=logger,
            message_broker=message_broker
        )
        
        agent_config = AgentConfig(
            agent_id="research_agent_demo",
            agent_type="research",
            config=research_config,
            enabled=True,
            priority=1
        )
        
        await agent_manager.register_agent(research_agent, agent_config)
        print("✅ ResearchAgent registered successfully")
        
        # Start the research agent
        await agent_manager.start_agent("research_agent_demo")
        print("✅ ResearchAgent started successfully")
        
        # Demonstrate research task creation
        print("\n=== Creating Research Tasks ===")
        
        research_tasks = [
            ResearchTask(
                id="task_001",
                query="artificial intelligence trends 2024",
                sources=[
                    "https://example.com/ai-news",  # Would be real sources in production
                    "https://feeds.example.com/tech.rss"
                ],
                max_results=20,
                priority=1,
            ),
            ResearchTask(
                id="task_002", 
                query="machine learning automation",
                sources=[
                    "https://example.com/ml-research",
                    "https://feeds.example.com/research.rss"
                ],
                max_results=15,
                priority=2,
            ),
        ]
        
        # Display task information
        for task in research_tasks:
            print(f"\n📋 Task: {task.id}")
            print(f"   Query: {task.query}")
            print(f"   Sources: {len(task.sources)} configured")
            print(f"   Max Results: {task.max_results}")
            print(f"   Priority: {task.priority}")
        
        # Note: In a real scenario, these tasks would be executed, but for demo purposes
        # we'll show the configuration and integration only
        print(f"\n⚠️  Note: This is a demonstration script. In production, tasks would:")
        print("   - Fetch real content from web sources")
        print("   - Process RSS feeds") 
        print("   - Extract and score content")
        print("   - Generate research reports")
        print("   - Cache results for efficiency")
        
        # Show agent metrics
        print("\n=== Agent Metrics ===")
        metrics = research_agent.get_metrics()
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        # Show manager status
        print("\n=== Manager Status ===")
        status = agent_manager.get_manager_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # Health check
        print("\n=== Health Check ===")
        health_results = await agent_manager.health_check_all_agents()
        for agent_id, is_healthy in health_results.items():
            status_emoji = "✅" if is_healthy else "❌"
            print(f"   {agent_id}: {status_emoji} {'Healthy' if is_healthy else 'Unhealthy'}")
        
        print("\n✅ ResearchAgent demonstration completed successfully!")
        print("\nKey Features Implemented:")
        print("  ✅ Multi-source web scraping")
        print("  ✅ RSS feed monitoring")
        print("  ✅ Content extraction and scoring")
        print("  ✅ Deduplication algorithms")
        print("  ✅ Research report generation")
        print("  ✅ Intelligent caching")
        print("  ✅ Robots.txt compliance")
        print("  ✅ Rate limiting")
        print("  ✅ Agent Manager integration")
        print("  ✅ Comprehensive test coverage")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            await agent_manager.stop()
            print("🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")


async def demo_research_components():
    """Demonstrate individual research components."""
    print("\n=== Individual Component Demonstration ===\n")
    
    # Demo ResearchQuery optimization
    from src.agents.research import ResearchQuery
    
    print("📝 Query Optimization:")
    query = ResearchQuery("artificial intelligence and machine learning trends")
    optimized = query.optimize()
    expanded = query.expand()
    topics = query.extract_topics()
    
    print(f"   Original: {query.original_query}")
    print(f"   Optimized: {optimized}")
    print(f"   Expanded queries: {len(expanded)}")
    for i, exp_query in enumerate(expanded[:3], 1):
        print(f"     {i}. {exp_query}")
    print(f"   Extracted topics: {topics[:5]}")
    
    # Demo ContentItem
    from src.agents.research import ContentItem
    from datetime import datetime, timezone
    
    print("\n📄 Content Item:")
    content_item = ContentItem(
        id="demo_001",
        url="https://example.com/ai-article",
        title="The Future of AI: Trends and Predictions",
        content="Artificial intelligence is rapidly evolving with new breakthroughs in machine learning, natural language processing, and computer vision. This comprehensive analysis explores the latest developments and future predictions.",
        author="Dr. Sarah Johnson",
        published_date=datetime.now(timezone.utc),
        source_type="web",
        tags=["AI", "machine learning", "technology"],
        relevance_score=0.89,
        summary="Analysis of AI trends and future predictions"
    )
    
    print(f"   Title: {content_item.title}")
    print(f"   Author: {content_item.author}")
    print(f"   Relevance Score: {content_item.relevance_score}")
    print(f"   Tags: {', '.join(content_item.tags)}")
    print(f"   Content Preview: {content_item.content[:100]}...")
    
    # Demo similarity calculation
    content_item2 = ContentItem(
        id="demo_002",
        url="https://example.com/ml-trends",
        title="Machine Learning Trends for 2024",
        content="Machine learning continues to advance with new algorithms and applications. This article discusses the latest trends in ML and artificial intelligence technologies.",
        author="Prof. Michael Chen",
        published_date=datetime.now(timezone.utc),
        source_type="web",
        tags=["ML", "artificial intelligence", "trends"],
        relevance_score=0.82,
    )
    
    similarity = content_item.calculate_similarity(content_item2)
    is_duplicate = content_item.is_duplicate(content_item2, threshold=0.8)
    
    print(f"\n🔍 Content Similarity:")
    print(f"   Similarity Score: {similarity:.3f}")
    print(f"   Is Duplicate (threshold 0.8): {is_duplicate}")
    
    print("\n✅ Component demonstration completed!")


if __name__ == "__main__":
    asyncio.run(demo_research_agent())
    asyncio.run(demo_research_components())