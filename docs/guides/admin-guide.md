# Autonomous Agent Administrator Guide

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Configuration Management](#configuration-management)
4. [User Management](#user-management)
5. [Agent Management](#agent-management)
6. [Security Configuration](#security-configuration)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Performance Optimization](#performance-optimization)
9. [Backup and Recovery](#backup-and-recovery)
10. [Troubleshooting](#troubleshooting)
11. [Maintenance Procedures](#maintenance-procedures)
12. [API Management](#api-management)

## System Overview

### Architecture Components

The Autonomous Agent system consists of several key components:

- **Application Services**: Main business logic and API endpoints
- **Agent Services**: Gmail, Research, and Code agents
- **Intelligence Engine**: Decision-making and coordination
- **Database**: PostgreSQL for persistent data storage
- **Cache**: Redis for session and application caching
- **AI Service**: Ollama for local AI processing
- **Reverse Proxy**: Nginx for load balancing and SSL termination
- **Monitoring**: Prometheus, Grafana, and Alertmanager

### System Requirements

#### Minimum Requirements
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- Network: 100Mbps

#### Recommended Requirements
- CPU: 8 cores
- RAM: 16GB
- Storage: 200GB SSD
- Network: 1Gbps

#### Production Requirements
- CPU: 16 cores
- RAM: 32GB
- Storage: 500GB SSD
- Network: 10Gbps

## Installation and Setup

### Production Deployment

#### Prerequisites

1. **System Preparation**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install required packages
   sudo apt install -y curl wget git docker.io docker-compose
   
   # Configure Docker
   sudo usermod -aG docker $USER
   sudo systemctl enable docker
   sudo systemctl start docker
   ```

2. **Create Directory Structure**
   ```bash
   sudo mkdir -p /opt/autonomous-agent
   sudo chown $USER:$USER /opt/autonomous-agent
   cd /opt/autonomous-agent
   ```

3. **Clone Repository**
   ```bash
   git clone https://github.com/your-org/autonomous-agent.git .
   git checkout production
   ```

#### Environment Setup

1. **Generate Secrets**
   ```bash
   ./scripts/generate-secrets.sh
   ```

2. **Configure Environment Variables**
   ```bash
   cp .env.example .env.production
   # Edit .env.production with your values
   ```

3. **Set Up SSL Certificates**
   ```bash
   # Place SSL certificates in docker/nginx/ssl/
   sudo mkdir -p docker/nginx/ssl
   sudo cp your-cert.pem docker/nginx/ssl/
   sudo cp your-key.pem docker/nginx/ssl/
   ```

#### Deployment

1. **Deploy System**
   ```bash
   ./scripts/production/deploy.sh
   ```

2. **Verify Deployment**
   ```bash
   ./scripts/production/health-check.sh
   ```

### Development Setup

#### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Development Environment**
   ```bash
   docker-compose up -d
   python -m pytest tests/
   python src/main.py
   ```

## Configuration Management

### Environment Configuration

#### Production Configuration Files

**Location**: `/opt/autonomous-agent/config/production/`

**Key Files**:
- `app.yml`: Application configuration
- `agents.yml`: Agent-specific settings
- `database.yml`: Database configuration
- `security.yml`: Security settings
- `monitoring.yml`: Monitoring configuration

#### Application Configuration

**`config/production/app.yml`**
```yaml
app:
  name: "Autonomous Agent"
  version: "1.0.0"
  environment: "production"
  debug: false
  log_level: "INFO"
  
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30
  
database:
  host: "postgres"
  port: 5432
  name: "autonomous_agent"
  user: "agent"
  pool_size: 20
  max_overflow: 30
  
redis:
  host: "redis"
  port: 6379
  db: 0
  password: "${REDIS_PASSWORD}"
  
ollama:
  host: "ollama"
  port: 11434
  model: "llama2"
  timeout: 120
```

#### Agent Configuration

**`config/production/agents.yml`**
```yaml
gmail_agent:
  enabled: true
  rate_limit: 100
  batch_size: 50
  timeout: 30
  retry_attempts: 3
  
research_agent:
  enabled: true
  sources:
    - web
    - academic
    - news
  max_results: 100
  cache_ttl: 3600
  
code_agent:
  enabled: true
  supported_languages:
    - python
    - javascript
    - java
    - cpp
    - go
    - rust
  max_file_size: 1048576
  
intelligence_engine:
  enabled: true
  decision_timeout: 30
  learning_rate: 0.1
  confidence_threshold: 0.7
```

### Security Configuration

#### Authentication Settings

**`config/production/security.yml`**
```yaml
authentication:
  jwt:
    secret: "${JWT_SECRET}"
    expiration: 3600
    refresh_expiration: 86400
    algorithm: "HS256"
  
  oauth:
    google:
      client_id: "${GOOGLE_CLIENT_ID}"
      client_secret: "${GOOGLE_CLIENT_SECRET}"
      redirect_uri: "https://your-domain.com/auth/google/callback"
  
authorization:
  rbac:
    enabled: true
    default_role: "user"
    admin_role: "admin"
  
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    burst_size: 100
  
security:
  encryption:
    algorithm: "AES-256-GCM"
    key: "${ENCRYPTION_KEY}"
  
  cors:
    enabled: true
    origins:
      - "https://your-domain.com"
    methods:
      - "GET"
      - "POST"
      - "PUT"
      - "DELETE"
    headers:
      - "Content-Type"
      - "Authorization"
  
  ssl:
    enabled: true
    certificate: "/etc/ssl/certs/your-cert.pem"
    private_key: "/etc/ssl/private/your-key.pem"
    protocols:
      - "TLSv1.2"
      - "TLSv1.3"
```

## User Management

### User Operations

#### Creating Users

**API Method**:
```bash
curl -X POST https://your-domain.com/api/v1/admin/users \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "newuser",
    "email": "newuser@example.com",
    "password": "SecurePassword123",
    "role": "user"
  }'
```

**Database Method**:
```sql
INSERT INTO users (username, email, password_hash, role, created_at)
VALUES ('newuser', 'newuser@example.com', '$2b$12$...', 'user', NOW());
```

#### User Roles

**Role Hierarchy**:
- `admin`: Full system access
- `manager`: User management and monitoring
- `user`: Standard user access
- `readonly`: Read-only access

**Role Permissions**:
```yaml
admin:
  - system:*
  - user:*
  - agent:*
  - monitoring:*
  
manager:
  - user:read
  - user:create
  - user:update
  - monitoring:read
  - agent:read
  
user:
  - agent:use
  - task:create
  - task:read
  - task:update
  - profile:update
  
readonly:
  - task:read
  - profile:read
```

#### User Monitoring

**Active Sessions**:
```sql
SELECT u.username, s.created_at, s.last_activity, s.ip_address
FROM users u
JOIN sessions s ON u.id = s.user_id
WHERE s.active = true
ORDER BY s.last_activity DESC;
```

**User Activity**:
```sql
SELECT u.username, COUNT(t.id) as task_count, 
       MAX(t.created_at) as last_task
FROM users u
LEFT JOIN tasks t ON u.id = t.user_id
WHERE t.created_at > NOW() - INTERVAL '7 days'
GROUP BY u.id, u.username
ORDER BY task_count DESC;
```

## Agent Management

### Agent Configuration

#### Agent Status Monitoring

**Check Agent Health**:
```bash
curl -s https://your-domain.com/api/v1/agents/gmail/health | jq .
curl -s https://your-domain.com/api/v1/agents/research/health | jq .
curl -s https://your-domain.com/api/v1/agents/code/health | jq .
```

**Agent Metrics**:
```bash
# Get agent performance metrics
curl -s "https://your-domain.com/api/v1/admin/agents/metrics" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"
```

#### Agent Configuration Updates

**Update Agent Settings**:
```bash
curl -X PUT https://your-domain.com/api/v1/admin/agents/gmail/config \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "rate_limit": 200,
    "batch_size": 100,
    "timeout": 60
  }'
```

### Agent Troubleshooting

#### Common Issues

**Agent Not Responding**:
```bash
# Check agent logs
docker-compose logs -f app | grep "gmail_agent"

# Restart specific agent
curl -X POST https://your-domain.com/api/v1/admin/agents/gmail/restart \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"
```

**Performance Issues**:
```bash
# Check agent metrics
curl -s https://your-domain.com/api/v1/admin/agents/gmail/metrics | jq .

# Adjust agent configuration
curl -X PUT https://your-domain.com/api/v1/admin/agents/gmail/config \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -d '{"workers": 8, "timeout": 120}'
```

## Security Configuration

### SSL/TLS Setup

#### Certificate Management

**Install SSL Certificates**:
```bash
# Copy certificates to nginx directory
sudo cp your-cert.pem /opt/autonomous-agent/docker/nginx/ssl/
sudo cp your-key.pem /opt/autonomous-agent/docker/nginx/ssl/

# Update nginx configuration
sudo nano /opt/autonomous-agent/docker/nginx/conf.d/default.conf

# Reload nginx
docker-compose exec nginx nginx -s reload
```

**Certificate Renewal**:
```bash
# Automated renewal with Let's Encrypt
sudo certbot renew --nginx
```

#### Security Headers

**Nginx Security Configuration**:
```nginx
# Security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
add_header Content-Security-Policy "default-src 'self'" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
```

### Authentication & Authorization

#### JWT Configuration

**Token Management**:
```bash
# Generate new JWT secret
openssl rand -hex 32

# Update JWT configuration
export JWT_SECRET="your-new-secret"
```

#### OAuth Integration

**Google OAuth Setup**:
1. Create Google Cloud Project
2. Enable Gmail API
3. Create OAuth 2.0 credentials
4. Configure redirect URIs
5. Update application configuration

### Security Monitoring

#### Security Logs

**Access Logs**:
```bash
# View access logs
docker-compose logs nginx | grep "GET\|POST\|PUT\|DELETE"

# Monitor failed login attempts
docker-compose logs app | grep "authentication_failed"
```

**Security Alerts**:
```bash
# Set up security monitoring
curl -X POST https://your-domain.com/api/v1/admin/security/alerts \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -d '{
    "type": "failed_login",
    "threshold": 5,
    "time_window": 300,
    "action": "block_ip"
  }'
```

## Monitoring and Logging

### Prometheus Configuration

#### Metrics Collection

**System Metrics**:
- CPU usage
- Memory usage
- Disk usage
- Network I/O
- Container health

**Application Metrics**:
- Request rate
- Response time
- Error rate
- Agent performance
- Task completion rate

#### Custom Metrics

**Add Custom Metrics**:
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
request_count = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
response_time = Histogram('response_time_seconds', 'Response time')
active_users = Gauge('active_users', 'Number of active users')

# Use in application
request_count.labels(method='GET', endpoint='/api/v1/health').inc()
response_time.observe(0.5)
active_users.set(100)
```

### Grafana Dashboards

#### Dashboard Management

**Import Dashboard**:
```bash
curl -X POST http://admin:password@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @config/monitoring/grafana-dashboards.json
```

**Create Custom Dashboard**:
1. Access Grafana at `http://localhost:3000`
2. Login with admin credentials
3. Create new dashboard
4. Add panels with Prometheus queries
5. Configure alerts and notifications

### Log Management

#### Centralized Logging

**ELK Stack Configuration**:
```yaml
# docker-compose.yml
elasticsearch:
  image: elasticsearch:8.10.0
  environment:
    - discovery.type=single-node
    - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
  volumes:
    - elasticsearch_data:/usr/share/elasticsearch/data

logstash:
  image: logstash:8.10.0
  volumes:
    - ./config/logstash/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
  depends_on:
    - elasticsearch

kibana:
  image: kibana:8.10.0
  depends_on:
    - elasticsearch
  ports:
    - "5601:5601"
```

#### Log Analysis

**Query Examples**:
```bash
# Search for errors
curl -X GET "elasticsearch:9200/logs-*/_search" \
  -H "Content-Type: application/json" \
  -d '{"query": {"match": {"level": "ERROR"}}}'

# Analyze performance
curl -X GET "elasticsearch:9200/logs-*/_search" \
  -H "Content-Type: application/json" \
  -d '{"aggs": {"avg_response_time": {"avg": {"field": "response_time"}}}}'
```

## Performance Optimization

### Database Optimization

#### Query Optimization

**Identify Slow Queries**:
```sql
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

**Index Optimization**:
```sql
-- Create indexes for frequently queried columns
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_tasks_user_id ON tasks(user_id);
CREATE INDEX idx_tasks_created_at ON tasks(created_at);

-- Analyze index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

#### Connection Pooling

**PgBouncer Configuration**:
```ini
[databases]
autonomous_agent = host=postgres port=5432 dbname=autonomous_agent

[pgbouncer]
listen_port = 6432
listen_addr = 0.0.0.0
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 100
default_pool_size = 20
```

### Application Optimization

#### Caching Strategy

**Redis Caching**:
```python
import redis
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=0)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached_result = redis_client.get(cache_key)
            
            if cached_result:
                return json.loads(cached_result)
            
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

#### Async Processing

**Task Queue Configuration**:
```python
from celery import Celery

app = Celery('autonomous_agent')
app.config_from_object('celeryconfig')

@app.task
def process_email_async(email_data):
    # Process email in background
    return gmail_agent.process_email(email_data)

# Usage
process_email_async.delay(email_data)
```

### Infrastructure Optimization

#### Load Balancing

**Nginx Load Balancer**:
```nginx
upstream app_servers {
    server app1:8000 weight=1;
    server app2:8000 weight=1;
    server app3:8000 weight=1;
}

server {
    location / {
        proxy_pass http://app_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Auto-scaling

**Docker Swarm Configuration**:
```yaml
version: '3.8'
services:
  app:
    image: autonomous-agent:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
      resources:
        limits:
          cpus: '2.0'
          memory: 1G
        reservations:
          cpus: '1.0'
          memory: 512M
```

## Backup and Recovery

### Database Backup

#### Automated Backups

**Backup Script**:
```bash
#!/bin/bash
# /opt/autonomous-agent/scripts/backup-db.sh

BACKUP_DIR="/opt/autonomous-agent/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="database_backup_${TIMESTAMP}.sql"

# Create backup
docker-compose exec -T postgres pg_dump -U agent autonomous_agent > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Clean old backups (keep last 7 days)
find "$BACKUP_DIR" -name "database_backup_*.sql.gz" -mtime +7 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
```

**Cron Job Setup**:
```bash
# Add to crontab
0 2 * * * /opt/autonomous-agent/scripts/backup-db.sh
```

#### Point-in-Time Recovery

**Enable WAL Archiving**:
```bash
# PostgreSQL configuration
echo "wal_level = replica" >> postgresql.conf
echo "archive_mode = on" >> postgresql.conf
echo "archive_command = 'cp %p /opt/autonomous-agent/backups/wal/%f'" >> postgresql.conf
```

### Application Backup

#### Configuration Backup

**Backup Configuration**:
```bash
#!/bin/bash
# Backup configuration files
tar -czf "/opt/autonomous-agent/backups/config_$(date +%Y%m%d_%H%M%S).tar.gz" \
  /opt/autonomous-agent/config \
  /opt/autonomous-agent/docker \
  /opt/autonomous-agent/.env*
```

#### Volume Backup

**Data Volume Backup**:
```bash
#!/bin/bash
# Backup Docker volumes
docker run --rm \
  -v autonomous-agent_postgres_data:/source:ro \
  -v /opt/autonomous-agent/backups:/backup \
  alpine tar czf /backup/postgres_data_$(date +%Y%m%d_%H%M%S).tar.gz -C /source .
```

### Disaster Recovery

#### Recovery Procedures

**Database Recovery**:
```bash
#!/bin/bash
# Stop services
docker-compose down

# Restore database
gunzip -c database_backup_YYYYMMDD_HHMMSS.sql.gz | \
  docker-compose exec -T postgres psql -U agent -d autonomous_agent

# Restore volumes
docker run --rm \
  -v autonomous-agent_postgres_data:/target \
  -v /opt/autonomous-agent/backups:/backup:ro \
  alpine tar xzf /backup/postgres_data_YYYYMMDD_HHMMSS.tar.gz -C /target

# Start services
docker-compose up -d
```

## Troubleshooting

### Common Issues

#### Service Startup Issues

**Check Service Status**:
```bash
# Check all services
docker-compose ps

# Check specific service logs
docker-compose logs app
docker-compose logs postgres
docker-compose logs redis
```

**Database Connection Issues**:
```bash
# Test database connectivity
docker-compose exec postgres psql -U agent -d autonomous_agent -c "SELECT version();"

# Check database logs
docker-compose logs postgres | grep ERROR
```

#### Performance Issues

**Resource Monitoring**:
```bash
# Check system resources
docker stats

# Check disk usage
df -h
du -sh /opt/autonomous-agent/

# Check memory usage
free -h
```

**Database Performance**:
```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity;

-- Check long-running queries
SELECT query, state, query_start, now() - query_start AS duration
FROM pg_stat_activity
WHERE state = 'active'
ORDER BY duration DESC;
```

### Emergency Procedures

#### Service Recovery

**Emergency Restart**:
```bash
#!/bin/bash
# Emergency restart procedure
cd /opt/autonomous-agent

# Stop all services
docker-compose down

# Clean up resources
docker system prune -f

# Restart services
docker-compose up -d

# Wait for health checks
sleep 60

# Verify system health
./scripts/production/health-check.sh
```

#### Data Recovery

**Emergency Backup**:
```bash
#!/bin/bash
# Emergency backup before maintenance
EMERGENCY_BACKUP="/opt/autonomous-agent/backups/emergency_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EMERGENCY_BACKUP"

# Backup database
docker-compose exec -T postgres pg_dump -U agent autonomous_agent > "$EMERGENCY_BACKUP/database.sql"

# Backup volumes
docker run --rm \
  -v autonomous-agent_postgres_data:/source:ro \
  -v "$EMERGENCY_BACKUP":/backup \
  alpine tar czf /backup/postgres_data.tar.gz -C /source .

echo "Emergency backup created: $EMERGENCY_BACKUP"
```

## Maintenance Procedures

### Regular Maintenance

#### Daily Tasks

**System Health Check**:
```bash
#!/bin/bash
# Daily health check script
./scripts/production/health-check.sh --format json > /tmp/health.json

# Check for issues
if grep -q "critical" /tmp/health.json; then
    echo "Critical issues found" | mail -s "System Alert" admin@example.com
fi
```

#### Weekly Tasks

**Database Maintenance**:
```sql
-- Analyze table statistics
ANALYZE;

-- Vacuum database
VACUUM;

-- Reindex if needed
REINDEX DATABASE autonomous_agent;
```

**Log Rotation**:
```bash
#!/bin/bash
# Rotate application logs
find /opt/autonomous-agent/logs -name "*.log" -size +100M -exec gzip {} \;
find /opt/autonomous-agent/logs -name "*.log.gz" -mtime +30 -delete
```

#### Monthly Tasks

**Security Updates**:
```bash
#!/bin/bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Docker images
docker-compose pull
docker-compose up -d

# Update application dependencies
pip install --upgrade -r requirements.txt
```

**Performance Review**:
```bash
#!/bin/bash
# Generate performance report
curl -s "http://localhost:9090/api/v1/query?query=rate(http_requests_total[30d])" | jq .

# Review slow queries
psql -U agent -d autonomous_agent -c "
SELECT query, mean_time, calls
FROM pg_stat_statements
WHERE mean_time > 1000
ORDER BY mean_time DESC;
"
```

### Update Procedures

#### Application Updates

**Rolling Update**:
```bash
#!/bin/bash
# Rolling update procedure
cd /opt/autonomous-agent

# Pull latest code
git pull origin production

# Build new image
docker build -t autonomous-agent:latest .

# Rolling update
docker-compose up -d --scale app=2
sleep 30
docker-compose up -d --scale app=1 --no-recreate
```

#### Configuration Updates

**Update Configuration**:
```bash
#!/bin/bash
# Update configuration without downtime
cp config/production/app.yml config/production/app.yml.bak

# Edit configuration
nano config/production/app.yml

# Reload configuration
curl -X POST http://localhost:8000/admin/reload-config \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"
```

## API Management

### API Monitoring

#### Usage Analytics

**API Metrics**:
```bash
# Get API usage statistics
curl -s "http://localhost:9090/api/v1/query?query=sum(rate(http_requests_total[5m])) by (endpoint)" | jq .

# Get error rates
curl -s "http://localhost:9090/api/v1/query?query=sum(rate(http_requests_total{status=~\"5..\"}[5m])) by (endpoint)" | jq .
```

#### Rate Limiting

**Configure Rate Limits**:
```bash
curl -X PUT http://localhost:8000/admin/rate-limits \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -d '{
    "endpoint": "/api/v1/agents/gmail/send",
    "limit": 100,
    "window": 3600
  }'
```

### API Documentation

#### OpenAPI Specification

**Generate API Documentation**:
```bash
# Generate OpenAPI spec
python -m src.generate_openapi > docs/api/openapi.json

# Generate HTML documentation
npx redoc-cli build docs/api/openapi.json --output docs/api/index.html
```

#### API Versioning

**Version Management**:
```python
from fastapi import FastAPI

app = FastAPI(
    title="Autonomous Agent API",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json"
)

# Version-specific routes
@app.get("/api/v1/health")
async def health_v1():
    return {"status": "healthy", "version": "1.0"}

@app.get("/api/v2/health")
async def health_v2():
    return {"status": "healthy", "version": "2.0", "detailed": True}
```

---

*This administrator guide is regularly updated with new features and procedures. Always refer to the latest version for current best practices.*