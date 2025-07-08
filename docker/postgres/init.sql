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

-- Create tables for security and user management
CREATE TABLE IF NOT EXISTS security.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    login_attempts INTEGER DEFAULT 0,
    lockout_until TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb,
    preferences JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS security.user_roles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES security.users(id) ON DELETE CASCADE,
    role_name VARCHAR(100) NOT NULL,
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    granted_by UUID REFERENCES security.users(id),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(user_id, role_name)
);

CREATE TABLE IF NOT EXISTS security.user_permissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES security.users(id) ON DELETE CASCADE,
    permission_name VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    granted_by UUID REFERENCES security.users(id),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS security.user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES security.users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    refresh_token VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    refresh_expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS security.user_mfa (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES security.users(id) ON DELETE CASCADE,
    mfa_type VARCHAR(50) NOT NULL, -- 'totp', 'sms', 'email'
    secret_key VARCHAR(255), -- For TOTP
    phone_number VARCHAR(50), -- For SMS
    backup_codes TEXT[], -- Array of backup codes
    is_enabled BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used TIMESTAMP WITH TIME ZONE,
    UNIQUE(user_id, mfa_type)
);

CREATE TABLE IF NOT EXISTS security.user_oauth (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES security.users(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL, -- 'google', 'github', 'microsoft'
    provider_user_id VARCHAR(255) NOT NULL,
    access_token TEXT,
    refresh_token TEXT,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(user_id, provider),
    UNIQUE(provider, provider_user_id)
);

CREATE TABLE IF NOT EXISTS security.password_reset_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES security.users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    used_at TIMESTAMP WITH TIME ZONE,
    is_used BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS security.email_verification_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES security.users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    verified_at TIMESTAMP WITH TIME ZONE,
    is_used BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS security.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES security.users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    key_name VARCHAR(255) NOT NULL,
    key_prefix VARCHAR(20) NOT NULL, -- First 8 chars for identification
    permissions TEXT[],
    scopes TEXT[], -- API scopes
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used_at TIMESTAMP WITH TIME ZONE,
    usage_count INTEGER DEFAULT 0,
    rate_limit_overrides JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS security.rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identifier VARCHAR(255) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    window_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(identifier, endpoint)
);

CREATE TABLE IF NOT EXISTS security.user_activity_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES security.users(id) ON DELETE SET NULL,
    activity_type VARCHAR(100) NOT NULL,
    activity_details JSONB,
    ip_address INET,
    user_agent TEXT,
    session_id UUID REFERENCES security.user_sessions(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS security.security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES security.users(id) ON DELETE SET NULL,
    event_type VARCHAR(100) NOT NULL, -- 'login_failure', 'suspicious_activity', 'account_lockout'
    event_data JSONB,
    severity VARCHAR(20) DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    ip_address INET,
    user_agent TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by UUID REFERENCES security.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS security.user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES security.users(id) ON DELETE CASCADE,
    category VARCHAR(100) NOT NULL, -- 'email', 'agents', 'notifications', 'privacy'
    preferences JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, category)
);

CREATE TABLE IF NOT EXISTS security.user_data_consent (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES security.users(id) ON DELETE CASCADE,
    consent_type VARCHAR(100) NOT NULL, -- 'data_processing', 'marketing', 'analytics'
    consent_given BOOLEAN NOT NULL,
    consent_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    revoked_date TIMESTAMP WITH TIME ZONE,
    ip_address INET,
    metadata JSONB DEFAULT '{}'::jsonb
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

-- User management indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON security.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON security.users(email);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON security.users(is_active);
CREATE INDEX IF NOT EXISTS idx_users_is_verified ON security.users(is_verified);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON security.users(created_at);
CREATE INDEX IF NOT EXISTS idx_users_last_login ON security.users(last_login);
CREATE INDEX IF NOT EXISTS idx_users_lockout_until ON security.users(lockout_until);

-- User roles and permissions indexes
CREATE INDEX IF NOT EXISTS idx_user_roles_user_id ON security.user_roles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_role_name ON security.user_roles(role_name);
CREATE INDEX IF NOT EXISTS idx_user_roles_is_active ON security.user_roles(is_active);
CREATE INDEX IF NOT EXISTS idx_user_permissions_user_id ON security.user_permissions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_permissions_permission_name ON security.user_permissions(permission_name);
CREATE INDEX IF NOT EXISTS idx_user_permissions_resource_type ON security.user_permissions(resource_type);
CREATE INDEX IF NOT EXISTS idx_user_permissions_is_active ON security.user_permissions(is_active);

-- Session management indexes
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON security.user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_session_token ON security.user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_refresh_token ON security.user_sessions(refresh_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON security.user_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_user_sessions_is_active ON security.user_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_user_sessions_ip_address ON security.user_sessions(ip_address);

-- MFA indexes
CREATE INDEX IF NOT EXISTS idx_user_mfa_user_id ON security.user_mfa(user_id);
CREATE INDEX IF NOT EXISTS idx_user_mfa_mfa_type ON security.user_mfa(mfa_type);
CREATE INDEX IF NOT EXISTS idx_user_mfa_is_enabled ON security.user_mfa(is_enabled);

-- OAuth indexes
CREATE INDEX IF NOT EXISTS idx_user_oauth_user_id ON security.user_oauth(user_id);
CREATE INDEX IF NOT EXISTS idx_user_oauth_provider ON security.user_oauth(provider);
CREATE INDEX IF NOT EXISTS idx_user_oauth_provider_user_id ON security.user_oauth(provider_user_id);

-- Token indexes
CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_user_id ON security.password_reset_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_token_hash ON security.password_reset_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_expires_at ON security.password_reset_tokens(expires_at);
CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_is_used ON security.password_reset_tokens(is_used);
CREATE INDEX IF NOT EXISTS idx_email_verification_tokens_user_id ON security.email_verification_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_email_verification_tokens_token_hash ON security.email_verification_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_email_verification_tokens_expires_at ON security.email_verification_tokens(expires_at);
CREATE INDEX IF NOT EXISTS idx_email_verification_tokens_is_used ON security.email_verification_tokens(is_used);

-- API key indexes
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON security.api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON security.api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_prefix ON security.api_keys(key_prefix);
CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON security.api_keys(is_active);
CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON security.api_keys(expires_at);
CREATE INDEX IF NOT EXISTS idx_api_keys_last_used_at ON security.api_keys(last_used_at);

-- Rate limiting indexes
CREATE INDEX IF NOT EXISTS idx_rate_limits_identifier ON security.rate_limits(identifier);
CREATE INDEX IF NOT EXISTS idx_rate_limits_endpoint ON security.rate_limits(endpoint);
CREATE INDEX IF NOT EXISTS idx_rate_limits_window_start ON security.rate_limits(window_start);

-- Activity and security event indexes
CREATE INDEX IF NOT EXISTS idx_user_activity_log_user_id ON security.user_activity_log(user_id);
CREATE INDEX IF NOT EXISTS idx_user_activity_log_activity_type ON security.user_activity_log(activity_type);
CREATE INDEX IF NOT EXISTS idx_user_activity_log_created_at ON security.user_activity_log(created_at);
CREATE INDEX IF NOT EXISTS idx_user_activity_log_session_id ON security.user_activity_log(session_id);
CREATE INDEX IF NOT EXISTS idx_security_events_user_id ON security.security_events(user_id);
CREATE INDEX IF NOT EXISTS idx_security_events_event_type ON security.security_events(event_type);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON security.security_events(severity);
CREATE INDEX IF NOT EXISTS idx_security_events_resolved ON security.security_events(resolved);
CREATE INDEX IF NOT EXISTS idx_security_events_created_at ON security.security_events(created_at);

-- User preferences indexes
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON security.user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_user_preferences_category ON security.user_preferences(category);
CREATE INDEX IF NOT EXISTS idx_user_data_consent_user_id ON security.user_data_consent(user_id);
CREATE INDEX IF NOT EXISTS idx_user_data_consent_consent_type ON security.user_data_consent(consent_type);
CREATE INDEX IF NOT EXISTS idx_user_data_consent_consent_given ON security.user_data_consent(consent_given);

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

-- User management triggers
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON security.users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_oauth_updated_at
    BEFORE UPDATE ON security.user_oauth
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at
    BEFORE UPDATE ON security.user_preferences
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
    
    -- Clean up old user sessions (expired)
    DELETE FROM security.user_sessions 
    WHERE expires_at < NOW() OR refresh_expires_at < NOW();
    
    -- Clean up old password reset tokens (older than 1 day)
    DELETE FROM security.password_reset_tokens 
    WHERE expires_at < NOW() OR created_at < NOW() - INTERVAL '1 day';
    
    -- Clean up old email verification tokens (older than 7 days)
    DELETE FROM security.email_verification_tokens 
    WHERE expires_at < NOW() OR created_at < NOW() - INTERVAL '7 days';
    
    -- Clean up old user activity logs (older than 180 days)
    DELETE FROM security.user_activity_log 
    WHERE created_at < NOW() - INTERVAL '180 days';
    
    -- Clean up resolved security events (older than 90 days)
    DELETE FROM security.security_events 
    WHERE resolved = TRUE AND resolved_at < NOW() - INTERVAL '90 days';
    
    -- Reset failed login attempts for users not locked out
    UPDATE security.users 
    SET login_attempts = 0 
    WHERE lockout_until IS NULL OR lockout_until < NOW();
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