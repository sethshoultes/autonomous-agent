# Autonomous Agent Operations Runbook

## Table of Contents

1. [Overview](#overview)
2. [Daily Operations](#daily-operations)
3. [Weekly Operations](#weekly-operations)
4. [Monthly Operations](#monthly-operations)
5. [Incident Response](#incident-response)
6. [System Maintenance](#system-maintenance)
7. [Performance Monitoring](#performance-monitoring)
8. [Backup and Recovery](#backup-and-recovery)
9. [Security Operations](#security-operations)
10. [Troubleshooting Guide](#troubleshooting-guide)

## Overview

This runbook provides step-by-step procedures for operating and maintaining the Autonomous Agent system in production. It covers daily operations, incident response, maintenance procedures, and troubleshooting guides.

### Key Contacts

- **Primary On-Call**: ops-primary@autonomous-agent.com
- **Secondary On-Call**: ops-secondary@autonomous-agent.com  
- **Security Team**: security@autonomous-agent.com
- **Development Team**: dev-team@autonomous-agent.com
- **Management**: management@autonomous-agent.com

### Emergency Procedures

**Severity 1 (Critical)**
- System completely down
- Security breach
- Data loss
- Response time: 15 minutes

**Severity 2 (High)**
- Partial system outage
- Performance degradation
- Single component failure
- Response time: 1 hour

**Severity 3 (Medium)**
- Minor issues
- Non-critical warnings
- Response time: 4 hours

## Daily Operations

### Morning Health Check (08:00)

#### System Status Verification

1. **Run Health Check Script**
   ```bash
   cd /opt/autonomous-agent
   ./scripts/production/health-check.sh --format text
   ```

2. **Check System Metrics**
   ```bash
   # CPU Usage
   top -b -n1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}'
   
   # Memory Usage
   free -h | grep Mem | awk '{print $3 "/" $2}'
   
   # Disk Usage
   df -h | grep -vE '^Filesystem|tmpfs|cdrom'
   ```

3. **Verify Service Status**
   ```bash
   docker-compose -f docker-compose.prod.yml ps
   ```

4. **Check Application Logs**
   ```bash
   # Recent errors
   docker-compose logs --tail=100 app | grep -i error
   
   # Performance warnings
   docker-compose logs --tail=100 app | grep -i "slow\|timeout\|performance"
   ```

#### Database Health Check

1. **Connection Test**
   ```bash
   docker-compose exec postgres pg_isready -U agent
   ```

2. **Active Connections**
   ```sql
   SELECT count(*) FROM pg_stat_activity WHERE state = 'active';
   ```

3. **Long Running Queries**
   ```sql
   SELECT pid, now() - pg_stat_activity.query_start AS duration, query
   FROM pg_stat_activity
   WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
   AND state = 'active';
   ```

4. **Database Size**
   ```sql
   SELECT pg_database.datname, pg_size_pretty(pg_database_size(pg_database.datname))
   FROM pg_database
   WHERE datname = 'autonomous_agent';
   ```

#### Agent Status Check

1. **Agent Health Endpoints**
   ```bash
   curl -s http://localhost:8000/api/v1/agents/gmail/health | jq .
   curl -s http://localhost:8000/api/v1/agents/research/health | jq .
   curl -s http://localhost:8000/api/v1/agents/code/health | jq .
   curl -s http://localhost:8000/api/v1/intelligence/health | jq .
   ```

2. **Agent Performance Metrics**
   ```bash
   curl -s http://localhost:8000/api/v1/agents/gmail/metrics | jq .response_time
   curl -s http://localhost:8000/api/v1/agents/research/metrics | jq .response_time
   curl -s http://localhost:8000/api/v1/agents/code/metrics | jq .response_time
   ```

#### Monitoring Dashboard Review

1. **Access Grafana Dashboard**
   - Open http://localhost:3000
   - Review "Autonomous Agent - Production Overview" dashboard
   - Check for any red alerts or warnings

2. **Key Metrics to Review**
   - Request rate and response times
   - Error rates
   - System resource utilization
   - Agent performance metrics

### End-of-Day Review (18:00)

#### Performance Summary

1. **Generate Daily Report**
   ```bash
   ./scripts/production/daily-report.sh
   ```

2. **Check Error Logs**
   ```bash
   # Count errors by type
   docker-compose logs app | grep -i error | awk '{print $NF}' | sort | uniq -c
   
   # Review critical errors
   docker-compose logs app | grep -i "critical\|fatal"
   ```

3. **Review Task Completion**
   ```bash
   # Check task completion rate
   curl -s http://localhost:8000/api/v1/admin/metrics/tasks | jq .completion_rate
   
   # Check failed tasks
   curl -s http://localhost:8000/api/v1/admin/tasks?status=failed | jq .total
   ```

#### Security Review

1. **Check Access Logs**
   ```bash
   # Review nginx access logs
   docker-compose logs nginx | grep -E "(40[0-9]|50[0-9])" | tail -20
   
   # Check for suspicious activity
   docker-compose logs nginx | grep -E "(admin|login|auth)" | tail -20
   ```

2. **Failed Login Attempts**
   ```bash
   docker-compose logs app | grep "authentication_failed" | wc -l
   ```

## Weekly Operations

### Monday: System Maintenance

#### Database Maintenance

1. **Database Statistics Update**
   ```sql
   ANALYZE;
   ```

2. **Vacuum Operations**
   ```sql
   VACUUM (ANALYZE, VERBOSE);
   ```

3. **Index Maintenance**
   ```sql
   -- Check index usage
   SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
   FROM pg_stat_user_indexes
   ORDER BY idx_scan DESC;
   
   -- Rebuild if necessary
   REINDEX INDEX CONCURRENTLY idx_name;
   ```

#### Log Rotation

1. **Application Logs**
   ```bash
   # Rotate application logs
   find /opt/autonomous-agent/logs -name "*.log" -size +100M -exec gzip {} \;
   
   # Clean old logs (older than 30 days)
   find /opt/autonomous-agent/logs -name "*.log.gz" -mtime +30 -delete
   ```

2. **System Logs**
   ```bash
   # Rotate system logs
   sudo logrotate -f /etc/logrotate.conf
   ```

### Wednesday: Performance Review

#### Performance Analysis

1. **Response Time Analysis**
   ```bash
   # Get average response times
   curl -s "http://localhost:9090/api/v1/query?query=avg_over_time(http_request_duration_seconds[7d])" | jq .
   ```

2. **Resource Usage Trends**
   ```bash
   # CPU usage trend
   curl -s "http://localhost:9090/api/v1/query?query=avg_over_time(node_cpu_seconds_total[7d])" | jq .
   
   # Memory usage trend
   curl -s "http://localhost:9090/api/v1/query?query=avg_over_time(node_memory_MemAvailable_bytes[7d])" | jq .
   ```

3. **Database Performance**
   ```sql
   -- Slowest queries
   SELECT query, mean_time, calls, total_time
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;
   
   -- Most frequent queries
   SELECT query, calls, mean_time
   FROM pg_stat_statements
   ORDER BY calls DESC
   LIMIT 10;
   ```

### Friday: Security Review

#### Security Audit

1. **Access Log Analysis**
   ```bash
   # Analyze access patterns
   docker-compose logs nginx | awk '{print $1}' | sort | uniq -c | sort -nr | head -20
   
   # Check for security events
   docker-compose logs nginx | grep -E "(40[0-9]|50[0-9])" | tail -50
   ```

2. **SSL Certificate Check**
   ```bash
   # Check certificate expiration
   openssl s_client -connect your-domain.com:443 -servername your-domain.com < /dev/null 2>/dev/null | \
   openssl x509 -noout -dates
   ```

3. **Security Scan**
   ```bash
   # Run security scan
   ./scripts/security-scan.sh --report weekly
   ```

## Monthly Operations

### First Monday: System Updates

#### Security Updates

1. **System Package Updates**
   ```bash
   # Update system packages
   sudo apt update && sudo apt upgrade -y
   
   # Check for security updates
   sudo apt list --upgradable | grep -i security
   ```

2. **Docker Image Updates**
   ```bash
   # Pull latest images
   docker-compose pull
   
   # Update containers
   docker-compose up -d
   ```

3. **Dependency Updates**
   ```bash
   # Update Python dependencies
   pip install --upgrade -r requirements.txt
   
   # Check for security vulnerabilities
   pip audit
   ```

#### Configuration Review

1. **Review Configuration Files**
   ```bash
   # Check for configuration changes
   git diff HEAD~30 -- config/
   
   # Validate configuration
   ./scripts/validate-config.sh
   ```

2. **Update Security Policies**
   ```bash
   # Review and update security policies
   ./scripts/update-security-policies.sh
   ```

### Third Monday: Capacity Planning

#### Resource Analysis

1. **Storage Usage Trend**
   ```bash
   # Analyze storage growth
   du -h /opt/autonomous-agent/ | sort -h | tail -20
   
   # Database growth
   SELECT pg_size_pretty(pg_database_size('autonomous_agent'));
   ```

2. **Performance Metrics**
   ```bash
   # Generate monthly performance report
   ./scripts/monthly-performance-report.sh
   ```

3. **Capacity Projections**
   ```bash
   # Analyze usage trends
   ./scripts/capacity-analysis.sh
   ```

## Incident Response

### Severity 1 - Critical Incidents

#### Immediate Actions (0-15 minutes)

1. **Acknowledge Incident**
   ```bash
   # Update status page
   curl -X POST https://status.autonomous-agent.com/incidents \
     -H "Authorization: Bearer $STATUS_TOKEN" \
     -d '{"title": "System Outage", "status": "investigating"}'
   ```

2. **Assess Impact**
   ```bash
   # Check system status
   ./scripts/production/health-check.sh --format json
   
   # Review recent changes
   git log --oneline -10
   
   # Check monitoring alerts
   curl -s http://localhost:9093/api/v1/alerts | jq .
   ```

3. **Emergency Response**
   ```bash
   # If complete system failure
   ./scripts/production/emergency-restart.sh
   
   # If partial failure
   ./scripts/production/isolate-issue.sh
   ```

#### Investigation and Resolution (15-60 minutes)

1. **Identify Root Cause**
   ```bash
   # Check application logs
   docker-compose logs --tail=1000 app | grep -i error
   
   # Check system resources
   docker stats --no-stream
   
   # Database status
   docker-compose exec postgres pg_isready
   ```

2. **Implement Fix**
   ```bash
   # Rollback if needed
   ./scripts/production/rollback.sh --force
   
   # Apply hotfix
   ./scripts/production/hotfix.sh
   ```

3. **Verify Resolution**
   ```bash
   # Run health checks
   ./scripts/production/health-check.sh
   
   # Test critical functionality
   ./scripts/testing/smoke-test.sh
   ```

### Severity 2 - High Priority Incidents

#### Response Procedure (0-60 minutes)

1. **Assess Situation**
   ```bash
   # Identify affected components
   ./scripts/production/component-status.sh
   
   # Check user impact
   curl -s http://localhost:8000/api/v1/admin/metrics/users | jq .active_users
   ```

2. **Containment**
   ```bash
   # Isolate affected service
   docker-compose stop affected_service
   
   # Redirect traffic if needed
   ./scripts/production/traffic-redirect.sh
   ```

3. **Resolution**
   ```bash
   # Restart affected service
   docker-compose up -d affected_service
   
   # Monitor recovery
   ./scripts/production/monitor-recovery.sh
   ```

### Post-Incident Activities

#### Incident Documentation

1. **Create Incident Report**
   ```bash
   # Generate incident report
   ./scripts/incident-report.sh --incident-id $INCIDENT_ID
   ```

2. **Update Runbook**
   ```bash
   # Document lessons learned
   git add docs/runbooks/
   git commit -m "Update runbook based on incident $INCIDENT_ID"
   ```

## System Maintenance

### Planned Maintenance

#### Pre-Maintenance Checklist

1. **Schedule Maintenance Window**
   - Notify users 48 hours in advance
   - Update status page
   - Coordinate with stakeholders

2. **Prepare for Maintenance**
   ```bash
   # Create maintenance backup
   ./scripts/production/maintenance-backup.sh
   
   # Prepare rollback plan
   ./scripts/production/prepare-rollback.sh
   ```

3. **Verify Prerequisites**
   ```bash
   # Check system health
   ./scripts/production/health-check.sh
   
   # Verify backup integrity
   ./scripts/verify-backup.sh
   ```

#### Maintenance Procedure

1. **Enable Maintenance Mode**
   ```bash
   # Enable maintenance mode
   ./scripts/production/maintenance-mode.sh enable
   
   # Verify maintenance page
   curl -s http://localhost:8000/ | grep -i maintenance
   ```

2. **Perform Maintenance**
   ```bash
   # Stop services
   docker-compose down
   
   # Perform maintenance tasks
   ./scripts/maintenance-tasks.sh
   
   # Start services
   docker-compose up -d
   ```

3. **Post-Maintenance Verification**
   ```bash
   # Health check
   ./scripts/production/health-check.sh
   
   # Smoke tests
   ./scripts/testing/smoke-test.sh
   
   # Disable maintenance mode
   ./scripts/production/maintenance-mode.sh disable
   ```

### Emergency Maintenance

#### Immediate Actions

1. **Assess Urgency**
   ```bash
   # Check current system status
   ./scripts/production/system-status.sh
   
   # Evaluate risk
   ./scripts/risk-assessment.sh
   ```

2. **Emergency Notification**
   ```bash
   # Send emergency notification
   ./scripts/emergency-notification.sh "Emergency maintenance required"
   ```

3. **Rapid Response**
   ```bash
   # Quick maintenance
   ./scripts/production/emergency-maintenance.sh
   ```

## Performance Monitoring

### Key Performance Indicators

#### System Metrics

1. **Response Time Monitoring**
   ```bash
   # Check average response time
   curl -s "http://localhost:9090/api/v1/query?query=avg(http_request_duration_seconds)" | jq .
   
   # Check 95th percentile
   curl -s "http://localhost:9090/api/v1/query?query=histogram_quantile(0.95, http_request_duration_seconds_bucket)" | jq .
   ```

2. **Throughput Monitoring**
   ```bash
   # Requests per second
   curl -s "http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m])" | jq .
   
   # Tasks per minute
   curl -s "http://localhost:9090/api/v1/query?query=rate(tasks_completed_total[5m])*60" | jq .
   ```

3. **Error Rate Monitoring**
   ```bash
   # Error rate
   curl -s "http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~\"5..\"}[5m])" | jq .
   
   # Agent error rate
   curl -s "http://localhost:9090/api/v1/query?query=rate(agent_errors_total[5m])" | jq .
   ```

#### Application Metrics

1. **Agent Performance**
   ```bash
   # Gmail agent metrics
   curl -s http://localhost:8000/api/v1/agents/gmail/metrics | jq .
   
   # Research agent metrics
   curl -s http://localhost:8000/api/v1/agents/research/metrics | jq .
   
   # Code agent metrics
   curl -s http://localhost:8000/api/v1/agents/code/metrics | jq .
   ```

2. **Database Performance**
   ```sql
   -- Connection pool usage
   SELECT count(*) FROM pg_stat_activity;
   
   -- Query performance
   SELECT query, mean_time, calls
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 5;
   ```

### Performance Alerts

#### Alert Configuration

1. **Response Time Alerts**
   ```yaml
   # Prometheus alert rule
   - alert: HighResponseTime
     expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
     for: 5m
     labels:
       severity: warning
     annotations:
       summary: "High response time detected"
   ```

2. **Error Rate Alerts**
   ```yaml
   # Prometheus alert rule
   - alert: HighErrorRate
     expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
     for: 3m
     labels:
       severity: critical
     annotations:
       summary: "High error rate detected"
   ```

## Backup and Recovery

### Backup Verification

#### Daily Backup Check

1. **Verify Backup Completion**
   ```bash
   # Check backup logs
   tail -50 /var/log/backup.log | grep -i error
   
   # Verify backup files
   ls -la /opt/autonomous-agent/backups/ | head -10
   ```

2. **Test Backup Integrity**
   ```bash
   # Test database backup
   ./scripts/test-backup.sh --type database --backup latest
   
   # Test volume backup
   ./scripts/test-backup.sh --type volume --backup latest
   ```

#### Weekly Recovery Test

1. **Test Recovery Procedure**
   ```bash
   # Run recovery test in isolated environment
   ./scripts/recovery-test.sh --environment staging
   ```

2. **Verify Recovery Time**
   ```bash
   # Measure recovery time
   ./scripts/measure-recovery-time.sh
   ```

### Disaster Recovery

#### Recovery Procedures

1. **Database Recovery**
   ```bash
   # Stop services
   docker-compose down
   
   # Restore database
   ./scripts/restore-database.sh --backup $BACKUP_FILE
   
   # Verify data integrity
   ./scripts/verify-data-integrity.sh
   ```

2. **Full System Recovery**
   ```bash
   # Complete system restore
   ./scripts/disaster-recovery.sh --backup-date $BACKUP_DATE
   
   # Verify system functionality
   ./scripts/testing/production-test.sh --test-type smoke
   ```

## Security Operations

### Security Monitoring

#### Daily Security Checks

1. **Access Log Analysis**
   ```bash
   # Check for suspicious activity
   docker-compose logs nginx | grep -E "(40[0-9]|50[0-9])" | tail -20
   
   # Monitor authentication attempts
   docker-compose logs app | grep "authentication" | tail -20
   ```

2. **Security Scan**
   ```bash
   # Run security scan
   ./scripts/security-scan.sh --quick
   
   # Check for vulnerabilities
   ./scripts/vulnerability-scan.sh
   ```

#### Weekly Security Review

1. **Access Review**
   ```bash
   # Review user access
   curl -s http://localhost:8000/api/v1/admin/users | jq '.[] | {username, role, last_login}'
   
   # Review API key usage
   curl -s http://localhost:8000/api/v1/admin/api-keys | jq '.[] | {name, last_used}'
   ```

2. **Security Configuration**
   ```bash
   # Check security configuration
   ./scripts/security-config-check.sh
   
   # Review firewall rules
   sudo iptables -L -n
   ```

### Security Incident Response

#### Security Breach Response

1. **Immediate Actions**
   ```bash
   # Isolate affected systems
   ./scripts/security-isolate.sh
   
   # Collect evidence
   ./scripts/security-evidence.sh
   
   # Notify security team
   ./scripts/security-notification.sh
   ```

2. **Investigation**
   ```bash
   # Analyze logs
   ./scripts/security-analysis.sh
   
   # Check for indicators of compromise
   ./scripts/ioc-check.sh
   ```

3. **Recovery**
   ```bash
   # Secure system
   ./scripts/security-hardening.sh
   
   # Update passwords/keys
   ./scripts/credential-rotation.sh
   ```

## Troubleshooting Guide

### Common Issues

#### Application Not Starting

**Symptoms:**
- Application containers fail to start
- Health check endpoints not responding
- Error messages in logs

**Diagnosis:**
```bash
# Check container status
docker-compose ps

# Check logs
docker-compose logs app

# Check resource usage
docker stats --no-stream
```

**Resolution:**
```bash
# Restart containers
docker-compose restart

# If still failing, check configuration
./scripts/validate-config.sh

# Check for port conflicts
netstat -tulpn | grep :8000
```

#### Database Connection Issues

**Symptoms:**
- Database connection timeouts
- Connection pool exhaustion
- Query performance degradation

**Diagnosis:**
```bash
# Check database status
docker-compose exec postgres pg_isready

# Check connections
docker-compose exec postgres psql -U agent -c "SELECT count(*) FROM pg_stat_activity;"

# Check for long-running queries
docker-compose exec postgres psql -U agent -c "SELECT pid, query, state, query_start FROM pg_stat_activity WHERE state = 'active';"
```

**Resolution:**
```bash
# Restart database if needed
docker-compose restart postgres

# Kill long-running queries
docker-compose exec postgres psql -U agent -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'active' AND query_start < now() - interval '1 hour';"

# Adjust connection pool settings
# Edit config/database.yml
```

#### High Memory Usage

**Symptoms:**
- System running out of memory
- Containers being killed by OOM killer
- Performance degradation

**Diagnosis:**
```bash
# Check memory usage
free -h

# Check container memory usage
docker stats --no-stream

# Check for memory leaks
./scripts/memory-analysis.sh
```

**Resolution:**
```bash
# Restart memory-intensive services
docker-compose restart app

# Adjust memory limits
# Edit docker-compose.prod.yml

# Scale services if needed
docker-compose up -d --scale app=2
```

### Emergency Contacts

#### Escalation Matrix

**Level 1 - On-Call Engineer**
- Initial response and basic troubleshooting
- Escalate after 30 minutes if unresolved

**Level 2 - Senior Engineer**
- Complex technical issues
- System architecture problems
- Escalate after 60 minutes if unresolved

**Level 3 - Engineering Manager**
- Major incidents
- Business impact assessment
- External communication

**Level 4 - CTO/VP Engineering**
- Critical business impact
- External vendor coordination
- Executive communication

### Documentation Updates

#### Runbook Maintenance

1. **Regular Review**
   ```bash
   # Monthly runbook review
   git log --since="1 month ago" -- docs/runbooks/
   ```

2. **Update Process**
   ```bash
   # Update runbook
   git add docs/runbooks/
   git commit -m "Update runbook procedures"
   git push origin main
   ```

3. **Version Control**
   ```bash
   # Tag major updates
   git tag -a runbook-v1.1 -m "Runbook update v1.1"
   git push origin runbook-v1.1
   ```

---

*This runbook is a living document and should be updated regularly based on operational experience and system changes.*