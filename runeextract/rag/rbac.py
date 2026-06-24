"""
Feature 11: Role-Based Access Control (RBAC) for Documents

Document-level and field-level RBAC with automatic filtering.
Enterprise-ready RAG with security built-in.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AccessRule:
    """Access control rule for documents or fields."""
    pattern: str  # Glob pattern or field name
    allowed_roles: Set[str]
    allowed_users: Set[str] = field(default_factory=set)
    rule_type: str = "document"  # "document" or "field"
    
    def matches(self, path_or_field: str) -> bool:
        """Check if rule matches path or field name."""
        import fnmatch
        return fnmatch.fnmatch(path_or_field, self.pattern)
    
    def allows(self, user: str, roles: List[str]) -> bool:
        """Check if user/roles have access."""
        # Wildcard allows everyone
        if "*" in self.allowed_roles:
            return True
        
        # Check user
        if user in self.allowed_users:
            return True
        
        # Check roles
        return bool(set(roles) & self.allowed_roles)


@dataclass
class AuditLog:
    """Audit log entry."""
    timestamp: float
    user: str
    roles: List[str]
    query: str
    documents_accessed: List[str]
    fields_accessed: List[str]
    denied: bool = False
    reason: Optional[str] = None


class RBACManager:
    """Role-Based Access Control manager for RAG.
    
    Features:
    - Document-level permissions (glob patterns)
    - Field-level redaction
    - User and role-based access
    - Audit logging
    - Dynamic rule updates (no re-indexing)
    
    Usage::
    
        rbac = RBACManager()
        
        # Set document permissions
        rbac.set_permissions({
            "finance/*.pdf": ["finance_team", "executives"],
            "hr/*.docx": ["hr_team", "managers"],
            "public/*": ["*"]  # Everyone
        })
        
        # Set field-level rules
        rbac.set_field_rules({
            "salary": ["hr_team", "executives"],
            "ssn": ["hr_team"],
            "confidential": ["executives"]
        })
        
        # Check access
        allowed = rbac.check_access(
            path="finance/q4_report.pdf",
            user="alice@company.com",
            roles=["finance_team"]
        )
        
        # Filter chunks
        filtered = rbac.filter_chunks(
            chunks, 
            user="bob@company.com",
            roles=["engineer"]
        )
    """
    
    def __init__(
        self,
        enable_audit: bool = True,
        max_audit_entries: int = 10000
    ):
        """Initialize RBAC manager.
        
        Args:
            enable_audit: Enable audit logging
            max_audit_entries: Max audit log size
        """
        self.enable_audit = enable_audit
        self.max_audit_entries = max_audit_entries
        
        self._document_rules: List[AccessRule] = []
        self._field_rules: List[AccessRule] = []
        self._audit_log: List[AuditLog] = []
        
        # PII patterns for auto-detection
        self._pii_patterns = {
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        }
        
        logger.info(f"RBAC initialized (audit={enable_audit})")
    
    def set_permissions(self, rules: Dict[str, List[str]]) -> None:
        """Set document-level permissions.
        
        Args:
            rules: Dict of {pattern: [roles]} where pattern is glob-style
        
        Example::
        
            rbac.set_permissions({
                "finance/*.pdf": ["finance_team", "executives"],
                "hr/*.docx": ["hr_team"],
                "public/*": ["*"]
            })
        """
        self._document_rules.clear()
        
        for pattern, roles in rules.items():
            rule = AccessRule(
                pattern=pattern,
                allowed_roles=set(roles),
                rule_type="document"
            )
            self._document_rules.append(rule)
        
        logger.info(f"Set {len(self._document_rules)} document permission rules")
    
    def set_field_rules(self, rules: Dict[str, List[str]]) -> None:
        """Set field-level access rules.
        
        Args:
            rules: Dict of {field_name: [roles]}
        
        Example::
        
            rbac.set_field_rules({
                "salary": ["hr_team", "executives"],
                "ssn": ["hr_team"],
                "confidential": ["executives"]
            })
        """
        self._field_rules.clear()
        
        for field_name, roles in rules.items():
            rule = AccessRule(
                pattern=field_name,
                allowed_roles=set(roles),
                rule_type="field"
            )
            self._field_rules.append(rule)
        
        logger.info(f"Set {len(self._field_rules)} field-level rules")
    
    def add_user_permission(self, pattern: str, user: str) -> None:
        """Grant access to specific user (bypasses role check).
        
        Args:
            pattern: Document pattern or field name
            user: User email/ID
        """
        for rule in self._document_rules + self._field_rules:
            if rule.pattern == pattern:
                rule.allowed_users.add(user)
                logger.info(f"Granted {user} access to {pattern}")
                return
        
        logger.warning(f"Pattern {pattern} not found in rules")
    
    def check_access(
        self,
        path: str,
        user: str,
        roles: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """Check if user has access to document.
        
        Args:
            path: Document path
            user: User email/ID
            roles: List of user roles
        
        Returns:
            (allowed: bool, reason: Optional[str])
        """
        # If no rules defined, allow access
        if not self._document_rules:
            return True, None
        
        # Check each rule
        for rule in self._document_rules:
            if rule.matches(path):
                if rule.allows(user, roles):
                    return True, None
                else:
                    reason = f"Access denied: {path} requires roles {rule.allowed_roles}"
                    return False, reason
        
        # No matching rule - deny by default
        reason = f"Access denied: No permission rule for {path}"
        return False, reason
    
    def filter_chunks(
        self,
        chunks: List[Any],
        user: str,
        roles: List[str],
        redact_fields: bool = True
    ) -> List[Any]:
        """Filter chunks based on user permissions.
        
        Args:
            chunks: List of chunk objects with .source and .text attributes
            user: User email/ID
            roles: User roles
            redact_fields: Whether to redact sensitive fields
        
        Returns:
            Filtered list of chunks (modified in-place if redacting)
        """
        filtered_chunks = []
        documents_accessed = []
        
        for chunk in chunks:
            source = getattr(chunk, 'source', '')
            
            # Check document-level access
            allowed, reason = self.check_access(source, user, roles)
            
            if not allowed:
                logger.debug(f"Filtered chunk from {source}: {reason}")
                continue
            
            # Apply field-level redaction
            if redact_fields and hasattr(chunk, 'text'):
                chunk.text = self._redact_fields(chunk.text, user, roles)
            
            filtered_chunks.append(chunk)
            if source:
                documents_accessed.append(source)
        
        # Audit log
        if self.enable_audit:
            self._log_access(
                user=user,
                roles=roles,
                query="",
                documents_accessed=list(set(documents_accessed)),
                denied=len(filtered_chunks) < len(chunks)
            )
        
        logger.info(
            f"Filtered {len(chunks)} chunks → {len(filtered_chunks)} "
            f"for user {user} with roles {roles}"
        )
        
        return filtered_chunks
    
    def _redact_fields(self, text: str, user: str, roles: List[str]) -> str:
        """Redact sensitive fields from text.
        
        Args:
            text: Original text
            user: User email/ID
            roles: User roles
        
        Returns:
            Text with redacted fields
        """
        redacted = text
        
        # Check field-level rules
        for rule in self._field_rules:
            if not rule.allows(user, roles):
                # Redact this field
                field_pattern = re.compile(
                    rf'\b{rule.pattern}\b\s*[:=]\s*[^\n,;]+',
                    re.IGNORECASE
                )
                redacted = field_pattern.sub(
                    f'{rule.pattern}: [REDACTED]',
                    redacted
                )
        
        # Auto-redact PII patterns
        redacted = self._auto_redact_pii(redacted, user, roles)
        
        return redacted
    
    def _auto_redact_pii(self, text: str, user: str, roles: List[str]) -> str:
        """Automatically redact PII if user doesn't have access.
        
        Args:
            text: Original text
            user: User email/ID
            roles: User roles
        
        Returns:
            Text with PII redacted
        """
        # Check if user has PII access
        pii_allowed = any(
            role in ["hr_team", "legal", "executives", "admin"]
            for role in roles
        )
        
        if pii_allowed:
            return text
        
        # Redact PII
        redacted = text
        for pii_type, pattern in self._pii_patterns.items():
            redacted = pattern.sub(f'[{pii_type.upper()}_REDACTED]', redacted)
        
        return redacted
    
    def _log_access(
        self,
        user: str,
        roles: List[str],
        query: str,
        documents_accessed: List[str],
        denied: bool = False,
        reason: Optional[str] = None
    ) -> None:
        """Log access event for audit.
        
        Args:
            user: User email/ID
            roles: User roles
            query: Query text
            documents_accessed: List of document paths
            denied: Whether access was denied
            reason: Denial reason if applicable
        """
        import time
        
        entry = AuditLog(
            timestamp=time.time(),
            user=user,
            roles=roles,
            query=query,
            documents_accessed=documents_accessed,
            fields_accessed=[],
            denied=denied,
            reason=reason
        )
        
        self._audit_log.append(entry)
        
        # Trim log if too large
        if len(self._audit_log) > self.max_audit_entries:
            self._audit_log = self._audit_log[-self.max_audit_entries:]
        
        if denied:
            logger.warning(
                f"Access denied for {user} ({roles}): {reason}"
            )
    
    def get_audit_log(
        self,
        user: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit log entries.
        
        Args:
            user: Filter by user (optional)
            limit: Max entries to return
        
        Returns:
            List of audit log entries
        """
        logs = self._audit_log
        
        if user:
            logs = [log for log in logs if log.user == user]
        
        return logs[-limit:]
    
    def export_audit_log(self, filepath: str) -> None:
        """Export audit log to JSON file.
        
        Args:
            filepath: Output file path
        """
        import json
        from datetime import datetime
        
        data = []
        for log in self._audit_log:
            data.append({
                "timestamp": datetime.fromtimestamp(log.timestamp).isoformat(),
                "user": log.user,
                "roles": log.roles,
                "query": log.query,
                "documents_accessed": log.documents_accessed,
                "denied": log.denied,
                "reason": log.reason
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(data)} audit entries to {filepath}")
    
    def check_compliance(self) -> Dict[str, Any]:
        """Check RBAC compliance status.
        
        Returns:
            Dict with compliance metrics
        """
        total_access_attempts = len(self._audit_log)
        denied_attempts = sum(1 for log in self._audit_log if log.denied)
        
        return {
            "total_rules": len(self._document_rules) + len(self._field_rules),
            "document_rules": len(self._document_rules),
            "field_rules": len(self._field_rules),
            "audit_entries": len(self._audit_log),
            "total_access_attempts": total_access_attempts,
            "denied_attempts": denied_attempts,
            "denial_rate": denied_attempts / total_access_attempts if total_access_attempts > 0 else 0,
            "compliance_ready": len(self._document_rules) > 0 and self.enable_audit
        }
