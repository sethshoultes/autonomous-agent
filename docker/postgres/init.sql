-- ============================================================================
-- PostgreSQL Initialization Script for Autonomous Agent System
-- ============================================================================

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE autonomous_agent'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'autonomous_agent');

-- Connect to the database
\c autonomous_agent;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS agents;
CREATE SCHEMA IF NOT EXISTS communication;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS security;

-- Create tables for agents
CREATE TABLE IF NOT EXISTS agents.agent_instances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'inactive',
    configuration JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_heartbeat TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS agents.agent_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents.agent_instances(id) ON DELETE CASCADE,
    task_type VARCHAR(100) NOT NULL,
    task_data JSONB,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);

-- Create tables for communication
CREATE TABLE IF NOT EXISTS communication.messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sender_id UUID,
    recipient_id UUID,
    message_type VARCHAR(100) NOT NULL,
    content JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS communication.email_threads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id VARCHAR(255) NOT NULL,
    subject VARCHAR(500),
    participants TEXT[],
    labels TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create tables for monitoring
CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    source VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS monitoring.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create tables for security
CREATE TABLE IF NOT EXISTS security.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    key_name VARCHAR(255) NOT NULL,
    permissions TEXT[],
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS security.rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identifier VARCHAR(255) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    window_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(identifier, endpoint)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_agent_instances_status ON agents.agent_instances(status);
CREATE INDEX IF NOT EXISTS idx_agent_instances_type ON agents.agent_instances(type);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_status ON agents.agent_tasks(status);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_priority ON agents.agent_tasks(priority);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_created_at ON agents.agent_tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON communication.messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_type ON communication.messages(message_type);
CREATE INDEX IF NOT EXISTS idx_email_threads_thread_id ON communication.email_threads(thread_id);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON monitoring.system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON monitoring.system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON monitoring.audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON monitoring.audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON security.api_keys(is_active);
CREATE INDEX IF NOT EXISTS idx_rate_limits_identifier ON security.rate_limits(identifier);

-- Create functions for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at
CREATE TRIGGER update_agent_instances_updated_at
    BEFORE UPDATE ON agents.agent_instances
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_email_threads_updated_at
    BEFORE UPDATE ON communication.email_threads
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create function for cleaning old data
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS VOID AS $$
BEGIN
    -- Clean up old metrics (older than 30 days)
    DELETE FROM monitoring.system_metrics 
    WHERE timestamp < NOW() - INTERVAL '30 days';
    
    -- Clean up old audit logs (older than 90 days)
    DELETE FROM monitoring.audit_logs 
    WHERE timestamp < NOW() - INTERVAL '90 days';
    
    -- Clean up old rate limit records (older than 1 hour)
    DELETE FROM security.rate_limits 
    WHERE window_start < NOW() - INTERVAL '1 hour';
    
    -- Clean up expired API keys
    UPDATE security.api_keys 
    SET is_active = FALSE 
    WHERE expires_at < NOW() AND is_active = TRUE;
    
    -- Clean up completed tasks older than 7 days
    DELETE FROM agents.agent_tasks 
    WHERE status = 'completed' 
    AND completed_at < NOW() - INTERVAL '7 days';
END;
$$ LANGUAGE plpgsql;

-- Create default admin user (for initial setup)
INSERT INTO security.api_keys (key_hash, key_name, permissions)
VALUES (
    crypt('admin-key-change-me', gen_salt('bf')),
    'Admin Key',
    ARRAY['admin', 'read', 'write', 'delete']
) ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA agents TO agent;
GRANT USAGE ON SCHEMA communication TO agent;
GRANT USAGE ON SCHEMA monitoring TO agent;
GRANT USAGE ON SCHEMA security TO agent;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA agents TO agent;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA communication TO agent;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA monitoring TO agent;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA security TO agent;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA agents TO agent;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA communication TO agent;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA monitoring TO agent;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA security TO agent;

-- Create materialized views for performance
CREATE MATERIALIZED VIEW IF NOT EXISTS monitoring.agent_status_summary AS
SELECT 
    type,
    status,
    COUNT(*) as count,
    AVG(EXTRACT(EPOCH FROM (NOW() - last_heartbeat))) as avg_heartbeat_age
FROM agents.agent_instances
GROUP BY type, status;

CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_status_summary 
ON monitoring.agent_status_summary(type, status);

-- Refresh materialized view function
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW monitoring.agent_status_summary;
END;
$$ LANGUAGE plpgsql;

-- Create a function to validate JSON schemas
CREATE OR REPLACE FUNCTION validate_json_schema(data JSONB, schema_name VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
    -- This is a placeholder for JSON schema validation
    -- In a real implementation, you would use a proper JSON schema validator
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully!';
END $$;