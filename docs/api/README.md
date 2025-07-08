# Autonomous Agent API Documentation

## Overview

The Autonomous Agent API provides comprehensive endpoints for interacting with the intelligent agent system. This RESTful API enables seamless integration with Gmail, Research, and Code agents, along with the advanced Intelligence Engine for multi-agent coordination and learning.

## Base URL

**Production:** `https://your-domain.com/api/v1`
**Development:** `http://localhost:8000/api/v1`

## Authentication

All API endpoints require authentication using JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## API Endpoints

### Health & Status

#### GET /health
Returns the overall health status of the system.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "ollama": "healthy",
    "agents": {
      "gmail": "healthy",
      "research": "healthy",
      "code": "healthy"
    },
    "intelligence_engine": "healthy"
  }
}
```

#### GET /status
Returns detailed system status and metrics.

**Response:**
```json
{
  "uptime": 86400,
  "request_count": 1000,
  "error_rate": 0.01,
  "response_time": {
    "avg": 0.5,
    "p95": 1.2,
    "p99": 2.1
  },
  "resource_usage": {
    "cpu": 45.2,
    "memory": 62.8,
    "disk": 78.5
  }
}
```

### Gmail Agent

#### POST /agents/gmail/send
Send an email through the Gmail agent.

**Request:**
```json
{
  "to": ["recipient@example.com"],
  "cc": ["cc@example.com"],
  "bcc": ["bcc@example.com"],
  "subject": "Email Subject",
  "body": "Email body content",
  "attachments": [
    {
      "filename": "document.pdf",
      "content": "base64-encoded-content",
      "mime_type": "application/pdf"
    }
  ]
}
```

**Response:**
```json
{
  "message_id": "msg_123456789",
  "status": "sent",
  "timestamp": "2024-01-01T12:00:00Z",
  "thread_id": "thread_987654321"
}
```

#### GET /agents/gmail/messages
Retrieve emails from Gmail.

**Query Parameters:**
- `limit` (optional): Number of messages to retrieve (default: 10, max: 100)
- `query` (optional): Gmail search query
- `label` (optional): Label to filter by

**Response:**
```json
{
  "messages": [
    {
      "id": "msg_123456789",
      "thread_id": "thread_987654321",
      "from": "sender@example.com",
      "to": ["recipient@example.com"],
      "subject": "Email Subject",
      "body": "Email body content",
      "date": "2024-01-01T12:00:00Z",
      "labels": ["INBOX", "IMPORTANT"],
      "attachments": [
        {
          "filename": "document.pdf",
          "mime_type": "application/pdf",
          "size": 1024
        }
      ]
    }
  ],
  "total": 1,
  "next_page_token": "token_for_next_page"
}
```

#### GET /agents/gmail/labels
Get Gmail labels.

**Response:**
```json
{
  "labels": [
    {
      "id": "Label_1",
      "name": "INBOX",
      "type": "system"
    },
    {
      "id": "Label_2",
      "name": "Important",
      "type": "user"
    }
  ]
}
```

#### POST /agents/gmail/labels/{messageId}
Add or remove labels from a message.

**Request:**
```json
{
  "add_labels": ["Label_1", "Label_2"],
  "remove_labels": ["Label_3"]
}
```

**Response:**
```json
{
  "message_id": "msg_123456789",
  "labels": ["Label_1", "Label_2"],
  "status": "updated"
}
```

### Research Agent

#### POST /agents/research/search
Perform a research search.

**Request:**
```json
{
  "query": "artificial intelligence trends 2024",
  "sources": ["web", "academic", "news"],
  "limit": 10,
  "filters": {
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    },
    "language": "en",
    "domains": ["edu", "org"]
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "title": "AI Trends 2024: What to Expect",
      "url": "https://example.com/ai-trends-2024",
      "summary": "This article discusses the latest trends in artificial intelligence...",
      "source": "web",
      "relevance_score": 0.95,
      "date": "2024-01-15T10:00:00Z",
      "authors": ["John Smith", "Jane Doe"]
    }
  ],
  "query": "artificial intelligence trends 2024",
  "total_results": 1,
  "search_time": 1.2
}
```

#### GET /agents/research/history
Get research history.

**Query Parameters:**
- `limit` (optional): Number of results to retrieve (default: 10, max: 100)
- `offset` (optional): Offset for pagination (default: 0)

**Response:**
```json
{
  "searches": [
    {
      "id": "search_123456789",
      "query": "artificial intelligence trends 2024",
      "timestamp": "2024-01-01T12:00:00Z",
      "results_count": 10,
      "status": "completed"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

#### GET /agents/research/search/{searchId}
Get details of a specific search.

**Response:**
```json
{
  "id": "search_123456789",
  "query": "artificial intelligence trends 2024",
  "timestamp": "2024-01-01T12:00:00Z",
  "results": [
    {
      "title": "AI Trends 2024: What to Expect",
      "url": "https://example.com/ai-trends-2024",
      "summary": "This article discusses the latest trends in artificial intelligence...",
      "source": "web",
      "relevance_score": 0.95,
      "date": "2024-01-15T10:00:00Z"
    }
  ],
  "status": "completed",
  "search_time": 1.2
}
```

### Code Agent

#### POST /agents/code/analyze
Analyze code for issues and improvements.

**Request:**
```json
{
  "code": "def hello_world():\n    print('Hello, World!')",
  "language": "python",
  "analysis_type": ["syntax", "style", "security", "performance"],
  "config": {
    "max_line_length": 80,
    "enforce_docstrings": true
  }
}
```

**Response:**
```json
{
  "analysis_id": "analysis_123456789",
  "issues": [
    {
      "type": "style",
      "severity": "warning",
      "line": 1,
      "column": 1,
      "message": "Function missing docstring",
      "rule": "missing-docstring",
      "suggestion": "Add a docstring to describe the function"
    }
  ],
  "metrics": {
    "lines_of_code": 2,
    "complexity": 1,
    "maintainability_index": 85
  },
  "suggestions": [
    {
      "type": "improvement",
      "description": "Consider adding type hints",
      "example": "def hello_world() -> None:"
    }
  ]
}
```

#### POST /agents/code/generate
Generate code based on requirements.

**Request:**
```json
{
  "requirements": "Create a Python function that calculates the factorial of a number",
  "language": "python",
  "style": "functional",
  "include_tests": true,
  "include_docs": true
}
```

**Response:**
```json
{
  "generation_id": "gen_123456789",
  "code": "def factorial(n: int) -> int:\n    \"\"\"Calculate the factorial of a number.\"\"\"\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)",
  "tests": "def test_factorial():\n    assert factorial(0) == 1\n    assert factorial(1) == 1\n    assert factorial(5) == 120",
  "documentation": "# Factorial Function\n\nThis function calculates the factorial of a given number...",
  "explanation": "The factorial function uses recursion to calculate the result..."
}
```

#### POST /agents/code/review
Review code for quality and best practices.

**Request:**
```json
{
  "code": "def calculate_average(numbers):\n    return sum(numbers) / len(numbers)",
  "language": "python",
  "review_type": "comprehensive",
  "focus_areas": ["performance", "error_handling", "documentation"]
}
```

**Response:**
```json
{
  "review_id": "review_123456789",
  "overall_score": 6.5,
  "feedback": [
    {
      "category": "error_handling",
      "severity": "high",
      "message": "Function doesn't handle empty list case",
      "suggestion": "Add check for empty list to prevent ZeroDivisionError"
    },
    {
      "category": "documentation",
      "severity": "medium",
      "message": "Missing docstring and type hints",
      "suggestion": "Add function documentation and type annotations"
    }
  ],
  "improved_code": "def calculate_average(numbers: List[float]) -> float:\n    \"\"\"Calculate the average of a list of numbers.\"\"\"\n    if not numbers:\n        raise ValueError(\"Cannot calculate average of empty list\")\n    return sum(numbers) / len(numbers)"
}
```

### Intelligence Engine

#### POST /intelligence/decisions
Make intelligent decisions based on context.

**Request:**
```json
{
  "context": {
    "user_id": "user_123",
    "current_task": "email_processing",
    "available_agents": ["gmail", "research"],
    "user_preferences": {
      "priority": "efficiency",
      "notification_level": "medium"
    }
  },
  "decision_type": "agent_selection",
  "options": [
    {
      "agent": "gmail",
      "action": "send_email",
      "estimated_time": 5
    },
    {
      "agent": "research",
      "action": "fact_check",
      "estimated_time": 30
    }
  ]
}
```

**Response:**
```json
{
  "decision_id": "decision_123456789",
  "recommended_action": {
    "agent": "gmail",
    "action": "send_email",
    "confidence": 0.85,
    "reasoning": "Based on user preferences for efficiency and current context..."
  },
  "alternatives": [
    {
      "agent": "research",
      "action": "fact_check",
      "confidence": 0.65,
      "reasoning": "Could provide more accurate information but takes longer..."
    }
  ],
  "learning_applied": true,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### POST /intelligence/coordinate
Coordinate multiple agents for complex tasks.

**Request:**
```json
{
  "task": {
    "description": "Research and email summary about AI trends",
    "priority": "high",
    "deadline": "2024-01-02T12:00:00Z"
  },
  "agents": ["research", "gmail"],
  "user_preferences": {
    "communication_style": "concise",
    "detail_level": "medium"
  }
}
```

**Response:**
```json
{
  "coordination_id": "coord_123456789",
  "execution_plan": [
    {
      "step": 1,
      "agent": "research",
      "action": "search",
      "parameters": {
        "query": "AI trends 2024",
        "limit": 5
      },
      "estimated_time": 60
    },
    {
      "step": 2,
      "agent": "intelligence",
      "action": "analyze_results",
      "parameters": {
        "format": "summary"
      },
      "estimated_time": 30
    },
    {
      "step": 3,
      "agent": "gmail",
      "action": "compose_email",
      "parameters": {
        "style": "concise",
        "include_sources": true
      },
      "estimated_time": 15
    }
  ],
  "total_estimated_time": 105,
  "status": "planning_complete"
}
```

#### GET /intelligence/learn
Get learning insights and patterns.

**Response:**
```json
{
  "learning_summary": {
    "total_interactions": 1000,
    "patterns_detected": 15,
    "user_preferences_learned": {
      "communication_style": "concise",
      "preferred_agents": ["gmail", "research"],
      "common_tasks": ["email_processing", "research_tasks"]
    },
    "performance_improvements": {
      "response_time_reduction": 15,
      "accuracy_increase": 8,
      "user_satisfaction": 92
    }
  },
  "recent_patterns": [
    {
      "pattern": "morning_email_check",
      "frequency": 0.95,
      "context": "User typically checks emails between 8-9 AM"
    }
  ]
}
```

### Task Management

#### POST /tasks
Create a new task.

**Request:**
```json
{
  "title": "Research AI market trends",
  "description": "Find latest information about AI market trends for Q1 2024",
  "priority": "high",
  "deadline": "2024-01-15T17:00:00Z",
  "assigned_agent": "research",
  "tags": ["research", "ai", "market"],
  "metadata": {
    "source": "user_request",
    "estimated_time": 1800
  }
}
```

**Response:**
```json
{
  "task_id": "task_123456789",
  "status": "created",
  "created_at": "2024-01-01T12:00:00Z",
  "estimated_completion": "2024-01-01T12:30:00Z"
}
```

#### GET /tasks
Get list of tasks.

**Query Parameters:**
- `status` (optional): Filter by task status
- `agent` (optional): Filter by assigned agent
- `priority` (optional): Filter by priority level
- `limit` (optional): Number of tasks to retrieve (default: 10, max: 100)
- `offset` (optional): Offset for pagination

**Response:**
```json
{
  "tasks": [
    {
      "task_id": "task_123456789",
      "title": "Research AI market trends",
      "status": "in_progress",
      "priority": "high",
      "assigned_agent": "research",
      "created_at": "2024-01-01T12:00:00Z",
      "updated_at": "2024-01-01T12:15:00Z",
      "progress": 45
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

#### GET /tasks/{taskId}
Get details of a specific task.

**Response:**
```json
{
  "task_id": "task_123456789",
  "title": "Research AI market trends",
  "description": "Find latest information about AI market trends for Q1 2024",
  "status": "completed",
  "priority": "high",
  "assigned_agent": "research",
  "created_at": "2024-01-01T12:00:00Z",
  "completed_at": "2024-01-01T12:30:00Z",
  "result": {
    "summary": "AI market shows strong growth in Q1 2024...",
    "sources": ["https://example.com/ai-report"],
    "key_findings": [
      "Market growth of 25% year-over-year",
      "Increased investment in generative AI"
    ]
  },
  "execution_log": [
    {
      "timestamp": "2024-01-01T12:05:00Z",
      "action": "started_research",
      "agent": "research"
    },
    {
      "timestamp": "2024-01-01T12:25:00Z",
      "action": "analysis_complete",
      "agent": "intelligence"
    }
  ]
}
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Missing required parameter: query",
    "details": {
      "parameter": "query",
      "expected_type": "string"
    },
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_123456789"
  }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default limit**: 1000 requests per hour per user
- **Burst limit**: 50 requests per minute
- **Rate limit headers** are included in all responses:
  - `X-RateLimit-Limit`: Total requests allowed
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset time (Unix timestamp)

## Webhooks

The system supports webhooks for real-time notifications:

### Webhook Events

- `task.created`: New task created
- `task.completed`: Task completed
- `task.failed`: Task failed
- `agent.status_changed`: Agent status changed
- `system.alert`: System alert triggered

### Webhook Payload Example

```json
{
  "event": "task.completed",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "task_id": "task_123456789",
    "title": "Research AI market trends",
    "status": "completed",
    "result": {
      "summary": "AI market shows strong growth..."
    }
  }
}
```

## SDK Examples

### Python SDK

```python
from autonomous_agent_sdk import Client

client = Client(
    base_url="https://api.autonomous-agent.com/v1",
    token="your-jwt-token"
)

# Send an email
response = client.gmail.send_email(
    to=["recipient@example.com"],
    subject="Test Email",
    body="This is a test email"
)

# Perform research
results = client.research.search(
    query="AI trends 2024",
    sources=["web", "academic"],
    limit=10
)

# Analyze code
analysis = client.code.analyze(
    code="def hello(): print('Hello')",
    language="python"
)
```

### JavaScript SDK

```javascript
import { AutonomousAgentClient } from 'autonomous-agent-sdk';

const client = new AutonomousAgentClient({
  baseURL: 'https://api.autonomous-agent.com/v1',
  token: 'your-jwt-token'
});

// Send an email
const emailResponse = await client.gmail.sendEmail({
  to: ['recipient@example.com'],
  subject: 'Test Email',
  body: 'This is a test email'
});

// Perform research
const searchResults = await client.research.search({
  query: 'AI trends 2024',
  sources: ['web', 'academic'],
  limit: 10
});

// Analyze code
const analysis = await client.code.analyze({
  code: "def hello(): print('Hello')",
  language: 'python'
});
```

## Best Practices

1. **Always use HTTPS** in production
2. **Implement proper error handling** for all API calls
3. **Use pagination** for large result sets
4. **Cache responses** when appropriate
5. **Monitor rate limits** and implement backoff strategies
6. **Use webhooks** for real-time updates instead of polling
7. **Keep JWT tokens secure** and rotate them regularly
8. **Validate input** before sending requests
9. **Handle timeouts** gracefully
10. **Log API interactions** for debugging and monitoring

## Support

For API support and questions:
- **Documentation**: https://docs.autonomous-agent.com
- **Support Email**: support@autonomous-agent.com
- **GitHub Issues**: https://github.com/autonomous-agent/issues
- **Status Page**: https://status.autonomous-agent.com