"""
Security Monitoring Service.

This module provides comprehensive security monitoring, audit logging,
threat detection, and security event management for the autonomous agent system.
"""

import logging
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from collections import defaultdict, deque

from pydantic import BaseModel, Field
import geoip2.database
import geoip2.errors

from ..database.connection import get_database_connection
from ..monitoring.metrics import MetricsCollector


logger = logging.getLogger(__name__)


class SecurityEventType(str, Enum):
    """Security event types."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    MULTIPLE_LOGIN_FAILURES = "multiple_login_failures"
    ACCOUNT_LOCKED = "account_locked"
    SUSPICIOUS_LOGIN = "suspicious_login"
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    UNUSUAL_LOCATION = "unusual_location"
    UNUSUAL_DEVICE = "unusual_device"
    UNUSUAL_TIME = "unusual_time"
    PASSWORD_CHANGE = "password_change"
    MFA_BYPASS_ATTEMPT = "mfa_bypass_attempt"
    API_KEY_ABUSE = "api_key_abuse"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    MALICIOUS_REQUEST = "malicious_request"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SESSION_HIJACKING = "session_hijacking"
    ACCOUNT_TAKEOVER = "account_takeover"


class SecurityEventSeverity(str, Enum):
    """Security event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatLevel(str, Enum):
    """Threat levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event model."""
    id: UUID
    event_type: SecurityEventType
    severity: SecurityEventSeverity
    user_id: Optional[UUID] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    event_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityAlert:
    """Security alert model."""
    id: UUID
    title: str
    description: str
    severity: SecurityEventSeverity
    threat_level: ThreatLevel
    affected_users: List[UUID]
    events: List[UUID]
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_by: Optional[UUID] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatIntelligence:
    """Threat intelligence data."""
    ip_reputation: Dict[str, Any] = field(default_factory=dict)
    malicious_user_agents: Set[str] = field(default_factory=set)
    suspicious_patterns: List[Dict[str, Any]] = field(default_factory=list)
    known_attack_vectors: List[Dict[str, Any]] = field(default_factory=list)
    geo_anomalies: Dict[str, Any] = field(default_factory=dict)


class SecurityMetrics(BaseModel):
    """Security metrics model."""
    total_events: int
    events_by_severity: Dict[str, int]
    events_by_type: Dict[str, int]
    active_alerts: int
    resolved_alerts: int
    threat_level: ThreatLevel
    top_threats: List[Dict[str, Any]]
    affected_users: int
    suspicious_ips: int
    blocked_attempts: int


class SecurityMonitoringService:
    """Security monitoring and threat detection service."""
    
    def __init__(self, config: Dict[str, Any], metrics_collector: Optional[MetricsCollector] = None):
        self.config = config
        self.metrics_collector = metrics_collector
        self.db = get_database_connection()
        
        # Security configuration
        self.max_login_attempts = config.get('max_login_attempts', 5)
        self.login_attempt_window = config.get('login_attempt_window', 300)  # 5 minutes
        self.suspicious_location_threshold = config.get('suspicious_location_threshold', 1000)  # km
        self.unusual_time_threshold = config.get('unusual_time_threshold', 4)  # hours
        self.rate_limit_threshold = config.get('rate_limit_threshold', 100)
        self.alert_retention_days = config.get('alert_retention_days', 90)
        
        # Threat intelligence
        self.threat_intelligence = ThreatIntelligence()
        
        # Geolocation database
        self.geoip_db = None
        geoip_db_path = config.get('geoip_db_path')
        if geoip_db_path:
            try:
                self.geoip_db = geoip2.database.Reader(geoip_db_path)
            except Exception as e:
                logger.warning(f"Failed to load GeoIP database: {str(e)}")
        
        # In-memory tracking for real-time analysis
        self.login_attempts = defaultdict(lambda: deque(maxlen=self.max_login_attempts))
        self.user_locations = defaultdict(list)
        self.user_devices = defaultdict(set)
        self.suspicious_ips = set()
        
        # Initialize threat intelligence
        self._initialize_threat_intelligence()
    
    def _initialize_threat_intelligence(self):
        """Initialize threat intelligence data."""
        try:
            # Load known malicious user agents
            self.threat_intelligence.malicious_user_agents.update([
                'sqlmap',
                'nikto',
                'nmap',
                'burp',
                'hydra',
                'gobuster',
                'dirbuster',
                'python-requests',  # Often used in attacks
                'curl',  # Can be suspicious in some contexts
                'wget',  # Can be suspicious in some contexts
            ])
            
            # Load suspicious patterns
            self.threat_intelligence.suspicious_patterns.extend([
                {'pattern': r'(?i)(union|select|insert|update|delete|drop|create|alter)', 'type': 'sql_injection'},
                {'pattern': r'(?i)(<script|javascript:|vbscript:|onload|onerror)', 'type': 'xss'},
                {'pattern': r'(?i)(\.\.\/|\.\.\\\\|%2e%2e%2f|%2e%2e%5c)', 'type': 'path_traversal'},
                {'pattern': r'(?i)(cmd|exec|system|eval|passthru|shell_exec)', 'type': 'command_injection'},
                {'pattern': r'(?i)(etc\/passwd|boot\.ini|win\.ini)', 'type': 'file_disclosure'},
            ])
            
            # Load known attack vectors
            self.threat_intelligence.known_attack_vectors.extend([
                {'name': 'credential_stuffing', 'indicators': ['multiple_users', 'same_ip', 'rapid_attempts']},
                {'name': 'brute_force', 'indicators': ['same_user', 'multiple_attempts', 'different_passwords']},
                {'name': 'account_takeover', 'indicators': ['unusual_location', 'new_device', 'password_change']},
                {'name': 'session_hijacking', 'indicators': ['ip_change', 'user_agent_change', 'concurrent_sessions']},
            ])
            
        except Exception as e:
            logger.error(f"Failed to initialize threat intelligence: {str(e)}")
    
    async def log_security_event(self, event_type: SecurityEventType,
                               user_id: Optional[UUID] = None,
                               ip_address: Optional[str] = None,
                               user_agent: Optional[str] = None,
                               event_data: Optional[Dict[str, Any]] = None,
                               severity: Optional[SecurityEventSeverity] = None) -> SecurityEvent:
        """Log a security event."""
        try:
            # Determine severity if not provided
            if severity is None:
                severity = self._determine_event_severity(event_type, event_data or {})
            
            # Get location information
            location = None
            if ip_address:
                location = self._get_location_from_ip(ip_address)
            
            # Create security event
            event = SecurityEvent(
                id=uuid4(),
                event_type=event_type,
                severity=severity,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                location=location,
                event_data=event_data or {},
                created_at=datetime.utcnow()
            )
            
            # Store event
            await self._store_security_event(event)
            
            # Analyze for threats
            await self._analyze_security_event(event)
            
            # Update metrics
            if self.metrics_collector:
                await self.metrics_collector.increment_counter(
                    'security_events_total',
                    labels={
                        'event_type': event_type.value,
                        'severity': severity.value
                    }
                )
            
            return event
            
        except Exception as e:
            logger.error(f"Failed to log security event: {str(e)}")
            raise
    
    async def analyze_login_attempt(self, user_id: UUID, 
                                  ip_address: str,
                                  user_agent: str,
                                  success: bool,
                                  event_data: Optional[Dict[str, Any]] = None) -> List[SecurityEvent]:
        """Analyze login attempt for security threats."""
        events = []
        
        try:
            # Track login attempt
            self.login_attempts[f"{user_id}:{ip_address}"].append({
                'timestamp': datetime.utcnow(),
                'success': success,
                'user_agent': user_agent,
                'event_data': event_data or {}
            })
            
            if success:
                # Successful login - check for anomalies
                events.extend(await self._analyze_successful_login(
                    user_id, ip_address, user_agent, event_data
                ))
            else:
                # Failed login - check for brute force
                events.extend(await self._analyze_failed_login(
                    user_id, ip_address, user_agent, event_data
                ))
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to analyze login attempt: {str(e)}")
            return events
    
    async def analyze_api_request(self, user_id: Optional[UUID],
                                api_key: Optional[str],
                                ip_address: str,
                                user_agent: str,
                                endpoint: str,
                                method: str,
                                status_code: int,
                                request_data: Optional[Dict[str, Any]] = None) -> List[SecurityEvent]:
        """Analyze API request for security threats."""
        events = []
        
        try:
            # Check for malicious patterns
            if self._contains_malicious_patterns(endpoint, request_data):
                event = await self.log_security_event(
                    SecurityEventType.MALICIOUS_REQUEST,
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    event_data={
                        'endpoint': endpoint,
                        'method': method,
                        'status_code': status_code,
                        'api_key': api_key[:8] + '...' if api_key else None
                    },
                    severity=SecurityEventSeverity.HIGH
                )
                events.append(event)
            
            # Check for API key abuse
            if api_key and await self._detect_api_key_abuse(api_key, ip_address):
                event = await self.log_security_event(
                    SecurityEventType.API_KEY_ABUSE,
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    event_data={
                        'api_key': api_key[:8] + '...',
                        'endpoint': endpoint,
                        'method': method
                    },
                    severity=SecurityEventSeverity.HIGH
                )
                events.append(event)
            
            # Check for privilege escalation attempts
            if await self._detect_privilege_escalation(user_id, endpoint, method):
                event = await self.log_security_event(
                    SecurityEventType.PRIVILEGE_ESCALATION,
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    event_data={
                        'endpoint': endpoint,
                        'method': method,
                        'status_code': status_code
                    },
                    severity=SecurityEventSeverity.CRITICAL
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to analyze API request: {str(e)}")
            return events
    
    async def detect_anomalies(self, time_window: timedelta = timedelta(hours=24)) -> List[SecurityAlert]:
        """Detect security anomalies in the specified time window."""
        alerts = []
        
        try:
            # Get recent events
            events = await self._get_recent_events(time_window)
            
            # Analyze patterns
            alerts.extend(await self._detect_brute_force_attacks(events))
            alerts.extend(await self._detect_credential_stuffing(events))
            alerts.extend(await self._detect_suspicious_patterns(events))
            alerts.extend(await self._detect_account_takeover(events))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {str(e)}")
            return alerts
    
    async def get_security_metrics(self, time_window: timedelta = timedelta(hours=24)) -> SecurityMetrics:
        """Get security metrics for the specified time window."""
        try:
            # Get events in time window
            events = await self._get_recent_events(time_window)
            
            # Calculate metrics
            total_events = len(events)
            events_by_severity = defaultdict(int)
            events_by_type = defaultdict(int)
            
            for event in events:
                events_by_severity[event.severity.value] += 1
                events_by_type[event.event_type.value] += 1
            
            # Get alert metrics
            alerts = await self._get_recent_alerts(time_window)
            active_alerts = len([a for a in alerts if not a.resolved])
            resolved_alerts = len([a for a in alerts if a.resolved])
            
            # Determine overall threat level
            threat_level = self._calculate_threat_level(events, alerts)
            
            # Get top threats
            top_threats = self._get_top_threats(events)
            
            # Get affected users and suspicious IPs
            affected_users = len(set(e.user_id for e in events if e.user_id))
            suspicious_ips = len(self.suspicious_ips)
            
            # Calculate blocked attempts
            blocked_attempts = len([e for e in events if e.event_type in [
                SecurityEventType.BRUTE_FORCE_ATTACK,
                SecurityEventType.RATE_LIMIT_EXCEEDED,
                SecurityEventType.MALICIOUS_REQUEST
            ]])
            
            return SecurityMetrics(
                total_events=total_events,
                events_by_severity=dict(events_by_severity),
                events_by_type=dict(events_by_type),
                active_alerts=active_alerts,
                resolved_alerts=resolved_alerts,
                threat_level=threat_level,
                top_threats=top_threats,
                affected_users=affected_users,
                suspicious_ips=suspicious_ips,
                blocked_attempts=blocked_attempts
            )
            
        except Exception as e:
            logger.error(f"Failed to get security metrics: {str(e)}")
            return SecurityMetrics(
                total_events=0,
                events_by_severity={},
                events_by_type={},
                active_alerts=0,
                resolved_alerts=0,
                threat_level=ThreatLevel.NONE,
                top_threats=[],
                affected_users=0,
                suspicious_ips=0,
                blocked_attempts=0
            )
    
    async def get_security_events(self, 
                                filters: Optional[Dict[str, Any]] = None,
                                limit: int = 100,
                                offset: int = 0) -> Tuple[List[SecurityEvent], int]:
        """Get security events with filtering."""
        try:
            return await self._get_security_events(filters, limit, offset)
        except Exception as e:
            logger.error(f"Failed to get security events: {str(e)}")
            return [], 0
    
    async def resolve_security_event(self, event_id: UUID, 
                                   resolved_by: UUID) -> bool:
        """Resolve a security event."""
        try:
            return await self._resolve_security_event(event_id, resolved_by)
        except Exception as e:
            logger.error(f"Failed to resolve security event: {str(e)}")
            return False
    
    async def create_security_alert(self, title: str,
                                  description: str,
                                  severity: SecurityEventSeverity,
                                  threat_level: ThreatLevel,
                                  affected_users: List[UUID],
                                  related_events: List[UUID]) -> SecurityAlert:
        """Create a security alert."""
        try:
            alert = SecurityAlert(
                id=uuid4(),
                title=title,
                description=description,
                severity=severity,
                threat_level=threat_level,
                affected_users=affected_users,
                events=related_events,
                created_at=datetime.utcnow()
            )
            
            await self._store_security_alert(alert)
            
            # Update metrics
            if self.metrics_collector:
                await self.metrics_collector.increment_counter(
                    'security_alerts_total',
                    labels={
                        'severity': severity.value,
                        'threat_level': threat_level.value
                    }
                )
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create security alert: {str(e)}")
            raise
    
    async def cleanup_old_data(self) -> bool:
        """Clean up old security data."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.alert_retention_days)
            
            # Clean up old events
            await self._cleanup_old_events(cutoff_date)
            
            # Clean up old alerts
            await self._cleanup_old_alerts(cutoff_date)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old security data: {str(e)}")
            return False
    
    # Helper methods
    
    def _determine_event_severity(self, event_type: SecurityEventType, 
                                event_data: Dict[str, Any]) -> SecurityEventSeverity:
        """Determine event severity based on type and data."""
        severity_mapping = {
            SecurityEventType.LOGIN_SUCCESS: SecurityEventSeverity.LOW,
            SecurityEventType.LOGIN_FAILURE: SecurityEventSeverity.LOW,
            SecurityEventType.MULTIPLE_LOGIN_FAILURES: SecurityEventSeverity.MEDIUM,
            SecurityEventType.ACCOUNT_LOCKED: SecurityEventSeverity.MEDIUM,
            SecurityEventType.SUSPICIOUS_LOGIN: SecurityEventSeverity.MEDIUM,
            SecurityEventType.BRUTE_FORCE_ATTACK: SecurityEventSeverity.HIGH,
            SecurityEventType.UNUSUAL_LOCATION: SecurityEventSeverity.MEDIUM,
            SecurityEventType.UNUSUAL_DEVICE: SecurityEventSeverity.MEDIUM,
            SecurityEventType.UNUSUAL_TIME: SecurityEventSeverity.LOW,
            SecurityEventType.PASSWORD_CHANGE: SecurityEventSeverity.LOW,
            SecurityEventType.MFA_BYPASS_ATTEMPT: SecurityEventSeverity.HIGH,
            SecurityEventType.API_KEY_ABUSE: SecurityEventSeverity.HIGH,
            SecurityEventType.RATE_LIMIT_EXCEEDED: SecurityEventSeverity.MEDIUM,
            SecurityEventType.PRIVILEGE_ESCALATION: SecurityEventSeverity.CRITICAL,
            SecurityEventType.DATA_BREACH_ATTEMPT: SecurityEventSeverity.CRITICAL,
            SecurityEventType.MALICIOUS_REQUEST: SecurityEventSeverity.HIGH,
            SecurityEventType.UNAUTHORIZED_ACCESS: SecurityEventSeverity.HIGH,
            SecurityEventType.SESSION_HIJACKING: SecurityEventSeverity.CRITICAL,
            SecurityEventType.ACCOUNT_TAKEOVER: SecurityEventSeverity.CRITICAL,
        }
        
        return severity_mapping.get(event_type, SecurityEventSeverity.MEDIUM)
    
    def _get_location_from_ip(self, ip_address: str) -> Optional[Dict[str, Any]]:
        """Get location information from IP address."""
        if not self.geoip_db:
            return None
        
        try:
            response = self.geoip_db.city(ip_address)
            return {
                'country': response.country.name,
                'country_code': response.country.iso_code,
                'city': response.city.name,
                'latitude': float(response.location.latitude) if response.location.latitude else None,
                'longitude': float(response.location.longitude) if response.location.longitude else None,
                'timezone': response.location.time_zone,
                'accuracy_radius': response.location.accuracy_radius
            }
        except geoip2.errors.AddressNotFoundError:
            return None
        except Exception as e:
            logger.warning(f"Failed to get location for IP {ip_address}: {str(e)}")
            return None
    
    def _contains_malicious_patterns(self, endpoint: str, 
                                   request_data: Optional[Dict[str, Any]]) -> bool:
        """Check if request contains malicious patterns."""
        import re
        
        # Check endpoint
        for pattern_data in self.threat_intelligence.suspicious_patterns:
            if re.search(pattern_data['pattern'], endpoint):
                return True
        
        # Check request data
        if request_data:
            data_string = json.dumps(request_data)
            for pattern_data in self.threat_intelligence.suspicious_patterns:
                if re.search(pattern_data['pattern'], data_string):
                    return True
        
        return False
    
    def _calculate_threat_level(self, events: List[SecurityEvent], 
                              alerts: List[SecurityAlert]) -> ThreatLevel:
        """Calculate overall threat level."""
        if not events and not alerts:
            return ThreatLevel.NONE
        
        # Count critical and high severity events
        critical_events = len([e for e in events if e.severity == SecurityEventSeverity.CRITICAL])
        high_events = len([e for e in events if e.severity == SecurityEventSeverity.HIGH])
        
        # Count critical and high alerts
        critical_alerts = len([a for a in alerts if a.severity == SecurityEventSeverity.CRITICAL])
        high_alerts = len([a for a in alerts if a.severity == SecurityEventSeverity.HIGH])
        
        # Determine threat level
        if critical_events > 0 or critical_alerts > 0:
            return ThreatLevel.CRITICAL
        elif high_events > 5 or high_alerts > 2:
            return ThreatLevel.HIGH
        elif high_events > 0 or high_alerts > 0:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _get_top_threats(self, events: List[SecurityEvent]) -> List[Dict[str, Any]]:
        """Get top threats from events."""
        threat_counts = defaultdict(int)
        
        for event in events:
            if event.severity in [SecurityEventSeverity.HIGH, SecurityEventSeverity.CRITICAL]:
                threat_counts[event.event_type.value] += 1
        
        # Sort by count and return top 5
        top_threats = sorted(threat_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return [
            {
                'threat_type': threat_type,
                'count': count,
                'severity': 'high' if count > 5 else 'medium'
            }
            for threat_type, count in top_threats
        ]
    
    # Analysis methods
    
    async def _analyze_security_event(self, event: SecurityEvent) -> None:
        """Analyze security event for immediate threats."""
        # Check for IP-based threats
        if event.ip_address:
            if await self._is_suspicious_ip(event.ip_address):
                self.suspicious_ips.add(event.ip_address)
        
        # Check for user agent threats
        if event.user_agent:
            if self._is_malicious_user_agent(event.user_agent):
                # Could trigger additional monitoring
                pass
    
    async def _analyze_successful_login(self, user_id: UUID, 
                                      ip_address: str,
                                      user_agent: str,
                                      event_data: Optional[Dict[str, Any]]) -> List[SecurityEvent]:
        """Analyze successful login for anomalies."""
        events = []
        
        # Check for unusual location
        if await self._is_unusual_location(user_id, ip_address):
            event = await self.log_security_event(
                SecurityEventType.UNUSUAL_LOCATION,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                event_data=event_data,
                severity=SecurityEventSeverity.MEDIUM
            )
            events.append(event)
        
        # Check for unusual device
        if await self._is_unusual_device(user_id, user_agent):
            event = await self.log_security_event(
                SecurityEventType.UNUSUAL_DEVICE,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                event_data=event_data,
                severity=SecurityEventSeverity.MEDIUM
            )
            events.append(event)
        
        # Check for unusual time
        if await self._is_unusual_time(user_id):
            event = await self.log_security_event(
                SecurityEventType.UNUSUAL_TIME,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                event_data=event_data,
                severity=SecurityEventSeverity.LOW
            )
            events.append(event)
        
        return events
    
    async def _analyze_failed_login(self, user_id: UUID, 
                                  ip_address: str,
                                  user_agent: str,
                                  event_data: Optional[Dict[str, Any]]) -> List[SecurityEvent]:
        """Analyze failed login for brute force attacks."""
        events = []
        
        # Check recent attempts
        key = f"{user_id}:{ip_address}"
        recent_attempts = self.login_attempts[key]
        
        # Count failed attempts in the window
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.login_attempt_window)
        
        failed_attempts = [
            attempt for attempt in recent_attempts
            if not attempt['success'] and attempt['timestamp'] > window_start
        ]
        
        if len(failed_attempts) >= self.max_login_attempts:
            event = await self.log_security_event(
                SecurityEventType.BRUTE_FORCE_ATTACK,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                event_data={
                    'failed_attempts': len(failed_attempts),
                    'time_window': self.login_attempt_window,
                    **(event_data or {})
                },
                severity=SecurityEventSeverity.HIGH
            )
            events.append(event)
        
        return events
    
    # Threat detection methods
    
    async def _detect_brute_force_attacks(self, events: List[SecurityEvent]) -> List[SecurityAlert]:
        """Detect brute force attacks."""
        alerts = []
        
        # Group failed login attempts by IP
        ip_attempts = defaultdict(list)
        for event in events:
            if event.event_type == SecurityEventType.LOGIN_FAILURE:
                ip_attempts[event.ip_address].append(event)
        
        # Check for brute force patterns
        for ip_address, attempts in ip_attempts.items():
            if len(attempts) > self.max_login_attempts * 2:  # Multiple users targeted
                affected_users = list(set(e.user_id for e in attempts if e.user_id))
                
                alert = await self.create_security_alert(
                    title=f"Brute Force Attack from {ip_address}",
                    description=f"Multiple failed login attempts from {ip_address} targeting {len(affected_users)} users",
                    severity=SecurityEventSeverity.HIGH,
                    threat_level=ThreatLevel.HIGH,
                    affected_users=affected_users,
                    related_events=[e.id for e in attempts]
                )
                alerts.append(alert)
        
        return alerts
    
    async def _detect_credential_stuffing(self, events: List[SecurityEvent]) -> List[SecurityAlert]:
        """Detect credential stuffing attacks."""
        alerts = []
        
        # Group login attempts by IP
        ip_events = defaultdict(list)
        for event in events:
            if event.event_type in [SecurityEventType.LOGIN_SUCCESS, SecurityEventType.LOGIN_FAILURE]:
                ip_events[event.ip_address].append(event)
        
        # Check for credential stuffing patterns
        for ip_address, attempts in ip_events.items():
            unique_users = set(e.user_id for e in attempts if e.user_id)
            
            if len(unique_users) > 10:  # Many different users from same IP
                alert = await self.create_security_alert(
                    title=f"Potential Credential Stuffing from {ip_address}",
                    description=f"Login attempts for {len(unique_users)} different users from {ip_address}",
                    severity=SecurityEventSeverity.HIGH,
                    threat_level=ThreatLevel.HIGH,
                    affected_users=list(unique_users),
                    related_events=[e.id for e in attempts]
                )
                alerts.append(alert)
        
        return alerts
    
    async def _detect_suspicious_patterns(self, events: List[SecurityEvent]) -> List[SecurityAlert]:
        """Detect suspicious patterns in events."""
        alerts = []
        
        # Check for multiple MFA bypass attempts
        mfa_bypass_events = [e for e in events if e.event_type == SecurityEventType.MFA_BYPASS_ATTEMPT]
        if len(mfa_bypass_events) > 3:
            affected_users = list(set(e.user_id for e in mfa_bypass_events if e.user_id))
            
            alert = await self.create_security_alert(
                title="Multiple MFA Bypass Attempts",
                description=f"Multiple MFA bypass attempts detected affecting {len(affected_users)} users",
                severity=SecurityEventSeverity.CRITICAL,
                threat_level=ThreatLevel.HIGH,
                affected_users=affected_users,
                related_events=[e.id for e in mfa_bypass_events]
            )
            alerts.append(alert)
        
        return alerts
    
    async def _detect_account_takeover(self, events: List[SecurityEvent]) -> List[SecurityAlert]:
        """Detect potential account takeover."""
        alerts = []
        
        # Group events by user
        user_events = defaultdict(list)
        for event in events:
            if event.user_id:
                user_events[event.user_id].append(event)
        
        # Check for account takeover patterns
        for user_id, user_events_list in user_events.items():
            # Look for password change followed by unusual activity
            password_changes = [e for e in user_events_list if e.event_type == SecurityEventType.PASSWORD_CHANGE]
            unusual_activities = [e for e in user_events_list if e.event_type in [
                SecurityEventType.UNUSUAL_LOCATION,
                SecurityEventType.UNUSUAL_DEVICE,
                SecurityEventType.SUSPICIOUS_LOGIN
            ]]
            
            if password_changes and len(unusual_activities) > 2:
                alert = await self.create_security_alert(
                    title=f"Potential Account Takeover - User {user_id}",
                    description="Password change followed by unusual activity patterns",
                    severity=SecurityEventSeverity.CRITICAL,
                    threat_level=ThreatLevel.CRITICAL,
                    affected_users=[user_id],
                    related_events=[e.id for e in password_changes + unusual_activities]
                )
                alerts.append(alert)
        
        return alerts
    
    # Utility methods
    
    async def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is suspicious."""
        # Check against known malicious IPs
        # This would typically check against threat intelligence feeds
        return ip_address in self.suspicious_ips
    
    def _is_malicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is malicious."""
        user_agent_lower = user_agent.lower()
        return any(malicious_ua in user_agent_lower 
                  for malicious_ua in self.threat_intelligence.malicious_user_agents)
    
    async def _is_unusual_location(self, user_id: UUID, ip_address: str) -> bool:
        """Check if location is unusual for user."""
        # Get current location
        current_location = self._get_location_from_ip(ip_address)
        if not current_location:
            return False
        
        # Get user's typical locations
        user_locations = await self._get_user_typical_locations(user_id)
        
        # Check if current location is far from typical locations
        for location in user_locations:
            distance = self._calculate_distance(current_location, location)
            if distance < self.suspicious_location_threshold:
                return False
        
        return True
    
    async def _is_unusual_device(self, user_id: UUID, user_agent: str) -> bool:
        """Check if device is unusual for user."""
        # Get user's typical devices
        user_devices = await self._get_user_typical_devices(user_id)
        
        # Check if current device is in the list
        device_fingerprint = self._get_device_fingerprint(user_agent)
        return device_fingerprint not in user_devices
    
    async def _is_unusual_time(self, user_id: UUID) -> bool:
        """Check if current time is unusual for user."""
        # Get user's typical login times
        typical_times = await self._get_user_typical_times(user_id)
        
        # Check if current time is outside normal hours
        current_hour = datetime.utcnow().hour
        return not any(abs(current_hour - typical_hour) < self.unusual_time_threshold 
                      for typical_hour in typical_times)
    
    async def _detect_api_key_abuse(self, api_key: str, ip_address: str) -> bool:
        """Detect API key abuse."""
        # Check rate limiting for API key
        # This would typically check against rate limit data
        return False
    
    async def _detect_privilege_escalation(self, user_id: Optional[UUID], 
                                         endpoint: str, method: str) -> bool:
        """Detect privilege escalation attempts."""
        # Check if user is trying to access admin endpoints
        admin_endpoints = ['/admin/', '/api/admin/', '/users/admin/']
        
        if any(admin_endpoint in endpoint for admin_endpoint in admin_endpoints):
            if user_id:
                user_role = await self._get_user_role(user_id)
                return user_role != 'admin'
            return True
        
        return False
    
    def _calculate_distance(self, location1: Dict[str, Any], 
                          location2: Dict[str, Any]) -> float:
        """Calculate distance between two locations."""
        if not all(key in location1 for key in ['latitude', 'longitude']):
            return float('inf')
        if not all(key in location2 for key in ['latitude', 'longitude']):
            return float('inf')
        
        # Simple distance calculation (in reality, use proper geo library)
        lat_diff = abs(location1['latitude'] - location2['latitude'])
        lon_diff = abs(location1['longitude'] - location2['longitude'])
        return (lat_diff + lon_diff) * 111  # Rough conversion to km
    
    def _get_device_fingerprint(self, user_agent: str) -> str:
        """Get device fingerprint from user agent."""
        return hashlib.md5(user_agent.encode()).hexdigest()
    
    # Database operations (to be implemented with actual database calls)
    
    async def _store_security_event(self, event: SecurityEvent) -> None:
        """Store security event in database."""
        # Implementation depends on your database layer
        pass
    
    async def _store_security_alert(self, alert: SecurityAlert) -> None:
        """Store security alert in database."""
        # Implementation depends on your database layer
        pass
    
    async def _get_recent_events(self, time_window: timedelta) -> List[SecurityEvent]:
        """Get recent security events."""
        # Implementation depends on your database layer
        pass
    
    async def _get_recent_alerts(self, time_window: timedelta) -> List[SecurityAlert]:
        """Get recent security alerts."""
        # Implementation depends on your database layer
        pass
    
    async def _get_security_events(self, filters: Optional[Dict[str, Any]], 
                                 limit: int, offset: int) -> Tuple[List[SecurityEvent], int]:
        """Get security events with filtering."""
        # Implementation depends on your database layer
        pass
    
    async def _resolve_security_event(self, event_id: UUID, resolved_by: UUID) -> bool:
        """Resolve security event."""
        # Implementation depends on your database layer
        pass
    
    async def _cleanup_old_events(self, cutoff_date: datetime) -> None:
        """Clean up old security events."""
        # Implementation depends on your database layer
        pass
    
    async def _cleanup_old_alerts(self, cutoff_date: datetime) -> None:
        """Clean up old security alerts."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_typical_locations(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get user's typical locations."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_typical_devices(self, user_id: UUID) -> Set[str]:
        """Get user's typical devices."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_typical_times(self, user_id: UUID) -> List[int]:
        """Get user's typical login times."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_role(self, user_id: UUID) -> str:
        """Get user role."""
        # Implementation depends on your database layer
        pass