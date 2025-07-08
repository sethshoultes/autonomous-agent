# Gmail Agent Documentation

## Overview

The Gmail Agent is a comprehensive email processing and management component of the autonomous agent system. It provides intelligent email handling, classification, automated responses, archiving, and seamless integration with the existing framework architecture.

## Features

### Core Capabilities

- **Email Fetching & Parsing**: Robust email retrieval with comprehensive parsing of headers, body, and attachments
- **Intelligent Classification**: ML-based email categorization (important, spam, personal, work, archive)
- **Automated Responses**: Context-aware auto-reply generation with rate limiting and templates
- **Email Archiving**: Smart organization with auto-labeling and folder assignment
- **Attachment Processing**: File extraction and organization capabilities
- **Thread Management**: Email conversation threading and relationship tracking
- **Email Summarization**: Daily/weekly email briefings and analytics

### Integration Features

- **Framework Integration**: Seamless integration with AgentManager, ConfigManager, and CommunicationBroker
- **Inter-Agent Communication**: Message-based communication with other agents
- **Rate Limiting**: Gmail API quota management and request throttling
- **Error Handling**: Comprehensive error recovery and retry mechanisms
- **Performance Monitoring**: Detailed metrics collection and reporting

## Architecture

### Class Hierarchy

```
BaseAgent (src/agents/base.py)
└── GmailAgent (src/agents/gmail_agent.py)
    ├── EmailClassification
    ├── EmailSummary
    ├── GmailRateLimiter
    └── Various helper classes
```

### Key Components

1. **GmailAgent**: Main agent class extending BaseAgent
2. **EmailClassification**: Classification result data structure
3. **EmailSummary**: Email summary and analytics data structure
4. **GmailRateLimiter**: API rate limiting management
5. **Error Classes**: Specialized exceptions for Gmail operations

## Configuration

### Configuration Schema

The Gmail Agent uses a comprehensive configuration schema defined in `src/config/manager.py`:

```yaml
gmail:
  credentials_path: "/path/to/credentials.json"
  scopes:
    - "https://www.googleapis.com/auth/gmail.readonly"
    - "https://www.googleapis.com/auth/gmail.send"
    - "https://www.googleapis.com/auth/gmail.modify"
  user_email: "user@example.com"
  batch_size: 100
  rate_limit_per_minute: 250
  max_retries: 3
  retry_delay: 1.0
  
  classification:
    enabled: true
    spam_threshold: 0.8
    importance_threshold: 0.7
    categories:
      - "important"
      - "spam"
      - "personal"
      - "work"
      - "archive"
    keywords:
      important:
        - "urgent"
        - "asap"
        - "deadline"
        - "priority"
      spam:
        - "prize"
        - "winner"
        - "lottery"
        - "click here"
      work:
        - "meeting"
        - "project"
        - "deadline"
        - "team"
      personal:
        - "family"
        - "friend"
        - "personal"
  
  auto_response:
    enabled: true
    response_delay: 300
    max_responses_per_day: 50
    templates:
      out_of_office: "I'm currently out of office..."
      meeting_request: "Thank you for the meeting request..."
      general_inquiry: "Thank you for your email..."
    trigger_patterns:
      out_of_office:
        - "vacation"
        - "out of office"
      meeting_request:
        - "meeting"
        - "call"
        - "appointment"
      general_inquiry:
        - "question"
        - "inquiry"
        - "help"
  
  archiving:
    enabled: true
    archive_after_days: 30
    auto_label: true
    label_rules:
      - pattern: "newsletter"
        label: "Newsletters"
      - pattern: "noreply"
        label: "Automated"
    smart_folders:
      receipts:
        - "receipt"
        - "invoice"
        - "purchase"
      travel:
        - "flight"
        - "hotel"
        - "booking"
```

### Agent Configuration

```yaml
agents:
  gmail_agent_001:
    agent_type: "gmail"
    enabled: true
    priority: 1
    config: {}
```

## API Reference

### Core Methods

#### Email Processing

```python
async def _fetch_emails(
    self,
    max_results: int = 100,
    query: Optional[str] = None,
    label_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]
```
Fetch emails from Gmail with optional filtering.

```python
async def _classify_email(
    self, 
    email_data: Dict[str, Any]
) -> EmailClassification
```
Classify an email into categories with confidence scores.

```python
async def _generate_auto_response(
    self, 
    email_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]
```
Generate automatic response for an email if applicable.

#### Archive and Organization

```python
async def _determine_labels(
    self, 
    email_data: Dict[str, Any]
) -> List[str]
```
Determine appropriate labels for an email based on configuration rules.

```python
async def _assign_smart_folder(
    self, 
    email_data: Dict[str, Any]
) -> Optional[str]
```
Assign email to a smart folder based on content analysis.

```python
async def _archive_old_emails(self) -> int
```
Archive emails older than the configured threshold.

#### Summarization and Analytics

```python
async def _generate_email_summary(
    self, 
    time_range: str
) -> EmailSummary
```
Generate comprehensive email summary for specified time range.

### Message Processing

The Gmail Agent supports inter-agent communication through message processing:

#### Supported Message Types

1. **fetch_emails**: Request email fetching
2. **classify_email**: Request email classification
3. **send_response**: Request auto-response sending
4. **get_summary**: Request email summary generation

#### Example Message Handling

```python
# Fetch emails request
fetch_message = AgentMessage(
    id="msg_001",
    sender="scheduler_agent",
    recipient="gmail_agent",
    message_type="fetch_emails",
    payload={
        "query": "is:unread",
        "max_results": 10,
        "label_ids": ["INBOX"]
    }
)

# Response
response = AgentMessage(
    id="resp_001",
    sender="gmail_agent",
    recipient="scheduler_agent",
    message_type="fetch_emails_response",
    payload={
        "emails": [...],
        "count": 5
    }
)
```

### Task Execution

The Gmail Agent supports various task types:

#### Email Summary Task

```python
task = {
    "task_type": "email_summary",
    "parameters": {
        "time_range": "last_24_hours",
        "categories": ["important", "work"],
        "include_attachments": False
    }
}
```

#### Bulk Archive Task

```python
task = {
    "task_type": "bulk_archive",
    "parameters": {
        "query": "older_than:30d",
        "batch_size": 100,
        "dry_run": False
    }
}
```

#### Label Cleanup Task

```python
task = {
    "task_type": "label_cleanup",
    "parameters": {
        "remove_unused_labels": True,
        "merge_similar_labels": True,
        "dry_run": False
    }
}
```

## Usage Examples

### Basic Agent Setup

```python
from src.agents.gmail_agent import GmailAgent
from src.config.manager import ConfigManager
import logging

# Load configuration
config_manager = ConfigManager()
config_manager.load_config("config/gmail_config.yaml")

# Create logger
logger = logging.getLogger("gmail_agent")

# Create message broker (mock for example)
message_broker = MockMessageBroker()

# Create Gmail agent
gmail_agent = GmailAgent(
    agent_id="gmail_agent_001",
    config=config_manager.config,
    logger=logger,
    message_broker=message_broker
)

# Start the agent
await gmail_agent.start()
```

### Email Classification

```python
# Fetch emails
emails = await gmail_agent._fetch_emails(max_results=10)

# Classify each email
for email in emails:
    classification = await gmail_agent._classify_email(email)
    print(f"Email: {email['subject']}")
    print(f"Category: {classification.category}")
    print(f"Confidence: {classification.confidence}")
    print(f"Keywords: {classification.keywords}")
```

### Automated Response Generation

```python
# Enable out-of-office mode
gmail_agent.out_of_office_mode = True

# Process incoming email
incoming_email = {...}  # Email data
response = await gmail_agent._generate_auto_response(incoming_email)

if response:
    # Send the auto response
    await gmail_agent._send_auto_response(response)
```

### Email Summarization

```python
# Generate daily summary
summary = await gmail_agent._generate_email_summary("last_24_hours")

print(f"Total emails: {summary.total_emails}")
print(f"Unread emails: {summary.unread_emails}")
print(f"Important emails: {summary.important_emails}")

# Category breakdown
for category, count in summary.categories.items():
    print(f"{category}: {count}")

# Top senders
for sender, count in summary.senders.items():
    print(f"{sender}: {count}")
```

## Testing

### Test Structure

The Gmail Agent includes comprehensive testing following TDD principles:

- **Unit Tests**: `tests/test_gmail_agent.py`
- **Integration Tests**: `tests/integration/test_gmail_agent_integration.py`
- **Mock Framework**: `tests/mocks/gmail_mocks.py`

### Mock Framework

The testing framework includes sophisticated mocks:

```python
from tests.mocks.gmail_mocks import (
    MockGmailService,
    MockGmailAPIContext,
    generate_sample_emails,
    generate_spam_emails
)

# Create mock service
mock_service = MockGmailService()

# Add sample emails
sample_emails = generate_sample_emails(5)
for email in sample_emails:
    mock_service.add_message(email)

# Use in tests
with MockGmailAPIContext(mock_service):
    await gmail_agent.start()
    emails = await gmail_agent._fetch_emails()
```

### Running Tests

```bash
# Run Gmail Agent tests
pytest tests/test_gmail_agent.py -v

# Run integration tests
pytest tests/integration/test_gmail_agent_integration.py -v

# Run with coverage
pytest tests/test_gmail_agent.py --cov=src.agents.gmail_agent

# Run all Gmail-related tests
pytest tests/ -k "gmail" -v
```

## Demonstration

### Demo Script

The Gmail Agent includes a comprehensive demonstration script:

```bash
# Run with mock data (default)
python demo_gmail_agent.py --mock --verbose

# Run with real Gmail API (requires credentials)
python demo_gmail_agent.py --real --config config/production.yaml

# Use custom configuration
python demo_gmail_agent.py --config config/custom_gmail.json
```

### Demo Features

The demonstration showcases:

1. **Email Fetching**: Various filtering and querying scenarios
2. **Classification**: Email categorization with different types
3. **Auto Responses**: Context-aware response generation
4. **Archiving**: Smart labeling and folder assignment
5. **Summarization**: Comprehensive email analytics
6. **Inter-Agent Communication**: Message-based interactions
7. **Performance Metrics**: System monitoring and reporting

## Security Considerations

### Authentication

- Uses OAuth2 service account credentials
- Supports credential refresh and rotation
- Implements secure credential storage practices

### Rate Limiting

- Comprehensive Gmail API quota management
- Request throttling to prevent API limits
- Graceful handling of rate limit errors

### Data Privacy

- Processes emails locally without external transmission
- Configurable data retention policies
- Secure handling of sensitive email content

## Performance Optimization

### Batch Processing

- Configurable batch sizes for email processing
- Efficient bulk operations for large email volumes
- Optimized API call patterns

### Caching

- Classification result caching
- Intelligent cache invalidation
- Memory-efficient data structures

### Resource Management

- Async/await pattern for non-blocking operations
- Proper resource cleanup and connection management
- Memory-efficient email processing

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify credentials path and format
   - Check OAuth2 scopes configuration
   - Ensure service account permissions

2. **Rate Limiting**
   - Adjust `rate_limit_per_minute` configuration
   - Implement exponential backoff
   - Monitor API usage quotas

3. **Classification Accuracy**
   - Tune threshold values
   - Update keyword lists
   - Review classification training data

4. **Performance Issues**
   - Adjust batch sizes
   - Optimize query patterns
   - Review memory usage

### Debug Mode

Enable verbose logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or configure through LoggingManager
logging_config = {
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
```

## Contributing

### Development Guidelines

1. Follow TDD principles - write tests first
2. Maintain existing code style and patterns
3. Update documentation for new features
4. Ensure comprehensive error handling
5. Add appropriate logging statements

### Adding New Features

1. **Email Classification**: Extend classification keywords and categories
2. **Auto Responses**: Add new template types and triggers
3. **Archiving Rules**: Implement additional smart folder logic
4. **API Integration**: Add support for additional Gmail API features

### Code Quality

- Maintain 90%+ test coverage
- Follow SOLID principles
- Use type hints consistently
- Document all public methods
- Implement proper error handling

## License

This Gmail Agent implementation is part of the autonomous agent system and follows the same licensing terms as the parent project.

## Support

For support and questions regarding the Gmail Agent:

1. Check the troubleshooting section
2. Review test cases for usage examples
3. Run the demonstration script for feature validation
4. Consult the integration tests for framework interaction patterns