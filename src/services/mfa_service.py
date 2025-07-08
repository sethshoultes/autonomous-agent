"""
Multi-Factor Authentication Service.

This module provides comprehensive MFA functionality including TOTP, SMS, and email
verification methods with backup codes and recovery options.
"""

import hashlib
import secrets
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID

import pyotp
import qrcode
import io
import base64
from pydantic import BaseModel, Field

from ..database.models.users import User
from ..database.connection import get_database_connection


logger = logging.getLogger(__name__)


class MFAMethodType(str, Enum):
    """MFA method types."""
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"


class MFAStatus(str, Enum):
    """MFA status."""
    DISABLED = "disabled"
    PENDING = "pending"
    ENABLED = "enabled"
    SUSPENDED = "suspended"


class TOTPSetupResult(BaseModel):
    """TOTP setup result."""
    secret: str
    backup_codes: List[str]
    qr_code: str
    recovery_codes: List[str]


class MFAVerificationResult(BaseModel):
    """MFA verification result."""
    success: bool
    method_used: Optional[MFAMethodType] = None
    remaining_attempts: int = 0
    lockout_until: Optional[datetime] = None
    error_message: Optional[str] = None


class MFAService:
    """Multi-factor authentication service."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = get_database_connection()
        
        # MFA settings
        self.totp_issuer = config.get('totp_issuer', 'Autonomous Agent System')
        self.totp_window = config.get('totp_window', 2)
        self.backup_codes_count = config.get('backup_codes_count', 10)
        self.max_attempts = config.get('mfa_max_attempts', 3)
        self.lockout_duration = config.get('mfa_lockout_duration', 300)  # 5 minutes
        
        # SMS settings (if enabled)
        self.sms_enabled = config.get('sms_enabled', False)
        self.sms_provider = config.get('sms_provider')
        self.sms_api_key = config.get('sms_api_key')
        
        # Email settings (if enabled)
        self.email_enabled = config.get('email_enabled', True)
        self.email_provider = config.get('email_provider')
    
    async def setup_totp(self, user_id: UUID) -> TOTPSetupResult:
        """Set up TOTP for user."""
        try:
            # Generate TOTP secret
            secret = pyotp.random_base32()
            
            # Get user for QR code generation
            user = await self._get_user(user_id)
            if not user:
                raise ValueError("User not found")
            
            # Generate provisioning URI
            totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                name=user.email,
                issuer_name=self.totp_issuer
            )
            
            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(totp_uri)
            qr.make(fit=True)
            
            # Create QR code image
            qr_image = qr.make_image(fill_color="black", back_color="white")
            buffer = io.BytesIO()
            qr_image.save(buffer, format='PNG')
            qr_code_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Generate backup codes
            backup_codes = self._generate_backup_codes()
            
            # Generate recovery codes (different from backup codes)
            recovery_codes = self._generate_recovery_codes()
            
            # Store MFA setup (pending confirmation)
            await self._store_mfa_setup(user_id, {
                'method': MFAMethodType.TOTP,
                'secret': secret,
                'backup_codes': backup_codes,
                'recovery_codes': recovery_codes,
                'status': MFAStatus.PENDING,
                'created_at': datetime.utcnow()
            })
            
            return TOTPSetupResult(
                secret=secret,
                backup_codes=backup_codes,
                qr_code=qr_code_base64,
                recovery_codes=recovery_codes
            )
            
        except Exception as e:
            logger.error(f"TOTP setup failed for user {user_id}: {str(e)}")
            raise
    
    async def setup_sms(self, user_id: UUID, phone_number: str) -> bool:
        """Set up SMS MFA for user."""
        try:
            if not self.sms_enabled:
                raise ValueError("SMS MFA is not enabled")
            
            # Validate phone number format
            if not self._validate_phone_number(phone_number):
                raise ValueError("Invalid phone number format")
            
            # Generate verification code
            verification_code = self._generate_verification_code()
            
            # Store SMS setup (pending confirmation)
            await self._store_mfa_setup(user_id, {
                'method': MFAMethodType.SMS,
                'phone_number': phone_number,
                'verification_code': verification_code,
                'status': MFAStatus.PENDING,
                'created_at': datetime.utcnow(),
                'expires_at': datetime.utcnow() + timedelta(minutes=5)
            })
            
            # Send SMS verification code
            await self._send_sms_verification(phone_number, verification_code)
            
            return True
            
        except Exception as e:
            logger.error(f"SMS setup failed for user {user_id}: {str(e)}")
            raise
    
    async def setup_email(self, user_id: UUID, email: str) -> bool:
        """Set up email MFA for user."""
        try:
            if not self.email_enabled:
                raise ValueError("Email MFA is not enabled")
            
            # Generate verification code
            verification_code = self._generate_verification_code()
            
            # Store email setup (pending confirmation)
            await self._store_mfa_setup(user_id, {
                'method': MFAMethodType.EMAIL,
                'email': email,
                'verification_code': verification_code,
                'status': MFAStatus.PENDING,
                'created_at': datetime.utcnow(),
                'expires_at': datetime.utcnow() + timedelta(minutes=5)
            })
            
            # Send email verification code
            await self._send_email_verification(email, verification_code)
            
            return True
            
        except Exception as e:
            logger.error(f"Email setup failed for user {user_id}: {str(e)}")
            raise
    
    async def verify_and_enable_mfa(self, user_id: UUID, verification_code: str) -> bool:
        """Verify MFA setup and enable it."""
        try:
            # Get pending MFA setup
            mfa_setup = await self._get_pending_mfa_setup(user_id)
            if not mfa_setup:
                return False
            
            # Verify based on method type
            if mfa_setup['method'] == MFAMethodType.TOTP:
                # Verify TOTP code
                totp = pyotp.TOTP(mfa_setup['secret'])
                if not totp.verify(verification_code, valid_window=self.totp_window):
                    return False
            
            elif mfa_setup['method'] == MFAMethodType.SMS:
                # Verify SMS code
                if mfa_setup.get('verification_code') != verification_code:
                    return False
                
                # Check if code has expired
                if datetime.utcnow() > mfa_setup.get('expires_at'):
                    return False
            
            elif mfa_setup['method'] == MFAMethodType.EMAIL:
                # Verify email code
                if mfa_setup.get('verification_code') != verification_code:
                    return False
                
                # Check if code has expired
                if datetime.utcnow() > mfa_setup.get('expires_at'):
                    return False
            
            # Enable MFA
            await self._enable_mfa(user_id, mfa_setup)
            
            # Clean up pending setup
            await self._cleanup_pending_mfa_setup(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"MFA verification failed for user {user_id}: {str(e)}")
            return False
    
    async def verify_mfa_code(self, user_id: UUID, code: str, 
                            backup_code: Optional[str] = None) -> MFAVerificationResult:
        """Verify MFA code for authentication."""
        try:
            # Check if user is locked out
            lockout_info = await self._check_mfa_lockout(user_id)
            if lockout_info and lockout_info > datetime.utcnow():
                return MFAVerificationResult(
                    success=False,
                    lockout_until=lockout_info,
                    error_message="Account temporarily locked due to too many failed attempts"
                )
            
            # Get user's MFA configuration
            mfa_config = await self._get_mfa_config(user_id)
            if not mfa_config or mfa_config['status'] != MFAStatus.ENABLED:
                return MFAVerificationResult(
                    success=False,
                    error_message="MFA is not enabled for this account"
                )
            
            # Try backup code first if provided
            if backup_code:
                if await self._verify_backup_code(user_id, backup_code):
                    await self._clear_mfa_attempts(user_id)
                    return MFAVerificationResult(
                        success=True,
                        method_used=mfa_config['method']
                    )
            
            # Verify based on method type
            if mfa_config['method'] == MFAMethodType.TOTP:
                totp = pyotp.TOTP(mfa_config['secret'])
                if totp.verify(code, valid_window=self.totp_window):
                    await self._clear_mfa_attempts(user_id)
                    return MFAVerificationResult(
                        success=True,
                        method_used=MFAMethodType.TOTP
                    )
            
            elif mfa_config['method'] == MFAMethodType.SMS:
                # For SMS, we need to send a code first
                if await self._verify_sms_code(user_id, code):
                    await self._clear_mfa_attempts(user_id)
                    return MFAVerificationResult(
                        success=True,
                        method_used=MFAMethodType.SMS
                    )
            
            elif mfa_config['method'] == MFAMethodType.EMAIL:
                # For email, we need to send a code first
                if await self._verify_email_code(user_id, code):
                    await self._clear_mfa_attempts(user_id)
                    return MFAVerificationResult(
                        success=True,
                        method_used=MFAMethodType.EMAIL
                    )
            
            # Failed verification - record attempt
            remaining_attempts = await self._record_mfa_attempt(user_id)
            
            # Check if user should be locked out
            lockout_until = None
            if remaining_attempts <= 0:
                lockout_until = datetime.utcnow() + timedelta(seconds=self.lockout_duration)
                await self._set_mfa_lockout(user_id, lockout_until)
            
            return MFAVerificationResult(
                success=False,
                remaining_attempts=remaining_attempts,
                lockout_until=lockout_until,
                error_message="Invalid verification code"
            )
            
        except Exception as e:
            logger.error(f"MFA verification failed for user {user_id}: {str(e)}")
            return MFAVerificationResult(
                success=False,
                error_message="MFA verification failed"
            )
    
    async def send_mfa_code(self, user_id: UUID) -> bool:
        """Send MFA code via configured method."""
        try:
            mfa_config = await self._get_mfa_config(user_id)
            if not mfa_config or mfa_config['status'] != MFAStatus.ENABLED:
                return False
            
            if mfa_config['method'] == MFAMethodType.SMS:
                # Generate and send SMS code
                code = self._generate_verification_code()
                await self._store_temporary_code(user_id, code, MFAMethodType.SMS)
                await self._send_sms_verification(mfa_config['phone_number'], code)
                return True
            
            elif mfa_config['method'] == MFAMethodType.EMAIL:
                # Generate and send email code
                code = self._generate_verification_code()
                await self._store_temporary_code(user_id, code, MFAMethodType.EMAIL)
                await self._send_email_verification(mfa_config['email'], code)
                return True
            
            # TOTP doesn't need codes to be sent
            return mfa_config['method'] == MFAMethodType.TOTP
            
        except Exception as e:
            logger.error(f"Failed to send MFA code for user {user_id}: {str(e)}")
            return False
    
    async def disable_mfa(self, user_id: UUID) -> bool:
        """Disable MFA for user."""
        try:
            # Get current MFA configuration
            mfa_config = await self._get_mfa_config(user_id)
            if not mfa_config:
                return True  # Already disabled
            
            # Disable MFA
            await self._disable_mfa(user_id)
            
            # Clean up any pending setups
            await self._cleanup_pending_mfa_setup(user_id)
            
            # Clear any lockouts
            await self._clear_mfa_lockout(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable MFA for user {user_id}: {str(e)}")
            return False
    
    async def get_mfa_status(self, user_id: UUID) -> Dict[str, Any]:
        """Get MFA status for user."""
        try:
            mfa_config = await self._get_mfa_config(user_id)
            if not mfa_config:
                return {
                    'enabled': False,
                    'method': None,
                    'status': MFAStatus.DISABLED
                }
            
            return {
                'enabled': mfa_config['status'] == MFAStatus.ENABLED,
                'method': mfa_config['method'],
                'status': mfa_config['status'],
                'created_at': mfa_config.get('created_at'),
                'last_used': mfa_config.get('last_used')
            }
            
        except Exception as e:
            logger.error(f"Failed to get MFA status for user {user_id}: {str(e)}")
            return {
                'enabled': False,
                'method': None,
                'status': MFAStatus.DISABLED
            }
    
    async def regenerate_backup_codes(self, user_id: UUID) -> List[str]:
        """Regenerate backup codes for user."""
        try:
            mfa_config = await self._get_mfa_config(user_id)
            if not mfa_config or mfa_config['status'] != MFAStatus.ENABLED:
                raise ValueError("MFA is not enabled")
            
            # Generate new backup codes
            new_backup_codes = self._generate_backup_codes()
            
            # Update MFA configuration
            await self._update_backup_codes(user_id, new_backup_codes)
            
            return new_backup_codes
            
        except Exception as e:
            logger.error(f"Failed to regenerate backup codes for user {user_id}: {str(e)}")
            raise
    
    async def get_recovery_codes(self, user_id: UUID) -> List[str]:
        """Get recovery codes for user."""
        try:
            mfa_config = await self._get_mfa_config(user_id)
            if not mfa_config or mfa_config['status'] != MFAStatus.ENABLED:
                raise ValueError("MFA is not enabled")
            
            return mfa_config.get('recovery_codes', [])
            
        except Exception as e:
            logger.error(f"Failed to get recovery codes for user {user_id}: {str(e)}")
            raise
    
    async def use_recovery_code(self, user_id: UUID, recovery_code: str) -> bool:
        """Use recovery code to disable MFA."""
        try:
            mfa_config = await self._get_mfa_config(user_id)
            if not mfa_config or mfa_config['status'] != MFAStatus.ENABLED:
                return False
            
            # Check if recovery code is valid
            recovery_codes = mfa_config.get('recovery_codes', [])
            if recovery_code not in recovery_codes:
                return False
            
            # Disable MFA
            await self._disable_mfa(user_id)
            
            # Log recovery code usage
            await self._log_recovery_code_usage(user_id, recovery_code)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to use recovery code for user {user_id}: {str(e)}")
            return False
    
    # Helper methods
    
    def _generate_backup_codes(self) -> List[str]:
        """Generate backup codes."""
        return [secrets.token_urlsafe(8) for _ in range(self.backup_codes_count)]
    
    def _generate_recovery_codes(self) -> List[str]:
        """Generate recovery codes."""
        return [secrets.token_urlsafe(12) for _ in range(5)]
    
    def _generate_verification_code(self) -> str:
        """Generate 6-digit verification code."""
        return str(secrets.randbelow(900000) + 100000)
    
    def _validate_phone_number(self, phone_number: str) -> bool:
        """Validate phone number format."""
        # Basic validation - in production, use a proper phone number library
        import re
        pattern = r'^\+?1?\d{9,15}$'
        return re.match(pattern, phone_number) is not None
    
    # Database operations (to be implemented with actual database calls)
    
    async def _get_user(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        # Implementation depends on your database layer
        pass
    
    async def _store_mfa_setup(self, user_id: UUID, setup_data: Dict[str, Any]) -> None:
        """Store MFA setup configuration."""
        # Implementation depends on your database layer
        pass
    
    async def _get_pending_mfa_setup(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get pending MFA setup."""
        # Implementation depends on your database layer
        pass
    
    async def _cleanup_pending_mfa_setup(self, user_id: UUID) -> None:
        """Clean up pending MFA setup."""
        # Implementation depends on your database layer
        pass
    
    async def _enable_mfa(self, user_id: UUID, setup_data: Dict[str, Any]) -> None:
        """Enable MFA for user."""
        # Implementation depends on your database layer
        pass
    
    async def _disable_mfa(self, user_id: UUID) -> None:
        """Disable MFA for user."""
        # Implementation depends on your database layer
        pass
    
    async def _get_mfa_config(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get MFA configuration for user."""
        # Implementation depends on your database layer
        pass
    
    async def _verify_backup_code(self, user_id: UUID, backup_code: str) -> bool:
        """Verify and consume backup code."""
        # Implementation depends on your database layer
        pass
    
    async def _verify_sms_code(self, user_id: UUID, code: str) -> bool:
        """Verify SMS code."""
        # Implementation depends on your database layer
        pass
    
    async def _verify_email_code(self, user_id: UUID, code: str) -> bool:
        """Verify email code."""
        # Implementation depends on your database layer
        pass
    
    async def _store_temporary_code(self, user_id: UUID, code: str, method: MFAMethodType) -> None:
        """Store temporary verification code."""
        # Implementation depends on your database layer
        pass
    
    async def _check_mfa_lockout(self, user_id: UUID) -> Optional[datetime]:
        """Check if user is locked out from MFA attempts."""
        # Implementation depends on your database layer
        pass
    
    async def _set_mfa_lockout(self, user_id: UUID, lockout_until: datetime) -> None:
        """Set MFA lockout for user."""
        # Implementation depends on your database layer
        pass
    
    async def _clear_mfa_lockout(self, user_id: UUID) -> None:
        """Clear MFA lockout for user."""
        # Implementation depends on your database layer
        pass
    
    async def _record_mfa_attempt(self, user_id: UUID) -> int:
        """Record failed MFA attempt and return remaining attempts."""
        # Implementation depends on your database layer
        pass
    
    async def _clear_mfa_attempts(self, user_id: UUID) -> None:
        """Clear MFA attempts for user."""
        # Implementation depends on your database layer
        pass
    
    async def _update_backup_codes(self, user_id: UUID, backup_codes: List[str]) -> None:
        """Update backup codes for user."""
        # Implementation depends on your database layer
        pass
    
    async def _log_recovery_code_usage(self, user_id: UUID, recovery_code: str) -> None:
        """Log recovery code usage."""
        # Implementation depends on your logging system
        pass
    
    # External service integrations
    
    async def _send_sms_verification(self, phone_number: str, code: str) -> None:
        """Send SMS verification code."""
        # Implementation depends on your SMS provider
        if not self.sms_enabled:
            return
        
        # Example implementation for Twilio
        if self.sms_provider == 'twilio':
            # Implementation would go here
            pass
    
    async def _send_email_verification(self, email: str, code: str) -> None:
        """Send email verification code."""
        # Implementation depends on your email provider
        if not self.email_enabled:
            return
        
        # Example implementation for SendGrid
        if self.email_provider == 'sendgrid':
            # Implementation would go here
            pass