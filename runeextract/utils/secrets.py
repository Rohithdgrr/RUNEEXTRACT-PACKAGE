"""
Secret scanning utility.

Detects API keys, tokens, passwords, and other secrets in extracted text.
Integrates with the security logging system for audit trails.
Includes input size limits to prevent ReDoS attacks.
"""

import logging
import re
import time
from typing import List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_MAX_SCAN_SIZE = 1_000_000  # 1MB max input to prevent ReDoS
_REGEX_TIMEOUT = 5.0  # seconds max for all regex operations


@dataclass
class SecretFinding:
    """A detected secret in text content."""
    secret_type: str
    pattern_name: str
    context: str
    start: int
    end: int
    severity: str  # "low", "medium", "high", "critical"
    line_number: int = 0
    redacted: str = ""


# Ordered list of (pattern_name, regex, severity, redaction_template)
_SECRET_PATTERNS = [
    # Critical severity — full credential exposure
    ("AWS Access Key", re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])"), "critical", "[AWS_KEY]"),
    ("GitHub PAT", re.compile(r"(?<![A-Za-z0-9])ghp_[a-zA-Z0-9]{36}(?![A-Za-z0-9])"), "critical", "[GH_TOKEN]"),
    ("GitHub Fine-Grained PAT", re.compile(r"(?<![A-Za-z0-9])github_pat_[a-zA-Z0-9]{84}(?![A-Za-z0-9])"), "critical", "[GH_TOKEN]"),
    ("GitHub OAuth", re.compile(r"(?<![A-Za-z0-9])gho_[a-zA-Z0-9]{36}(?![A-Za-z0-9])"), "critical", "[GH_TOKEN]"),
    ("GitHub Refresh", re.compile(r"(?<![A-Za-z0-9])ghr_[a-zA-Z0-9]{36}(?![A-Za-z0-9])"), "critical", "[GH_TOKEN]"),
    ("Slack Bot Token", re.compile(r"(?<![A-Za-z0-9])xoxb-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}(?![A-Za-z0-9])"), "critical", "[SLACK_TOKEN]"),
    ("Slack App Token", re.compile(r"(?<![A-Za-z0-9])xapp-[0-9]{10,13}-[a-zA-Z0-9]{24}(?![A-Za-z0-9])"), "critical", "[SLACK_TOKEN]"),
    ("Slack Webhook", re.compile(r"https://hooks\.slack\.com/services/[A-Z0-9a-z/]{40,60}"), "critical", "[SLACK_WEBHOOK]"),
    ("Stripe Live Key", re.compile(r"(?<![A-Za-z0-9])[rs]k_live_[a-zA-Z0-9]{24,40}(?![A-Za-z0-9])"), "critical", "[STRIPE_KEY]"),
    ("Stripe Live Publishable", re.compile(r"(?<![A-Za-z0-9])pk_live_[a-zA-Z0-9]{24,40}(?![A-Za-z0-9])"), "critical", "[STRIPE_KEY]"),
    ("Discord Bot Token", re.compile(r"(?<![A-Za-z0-9])[MN][A-Za-z\d]{23,25}\.[A-Za-z\d_\-]{6}\.[A-Za-z\d_\-]{27,38}(?![A-Za-z0-9])"), "critical", "[DISCORD_TOKEN]"),
    ("Google API Key", re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z\-_]{35}(?![A-Za-z0-9])"), "critical", "[GOOGLE_KEY]"),
    ("Heroku API Key", re.compile(r"(?<![A-Za-z0-9])[hH][eE][rR][oO][kK][uU]-[a-zA-Z0-9]{36}(?![A-Za-z0-9])"), "critical", "[HEROKU_KEY]"),
    ("Mailchimp API Key", re.compile(r"(?<![A-Za-z0-9])[a-f0-9]{32}-us[0-9]{1,2}(?![A-Za-z0-9])"), "critical", "[MAILCHIMP_KEY]"),
    ("Mailgun API Key", re.compile(r"(?<![A-Za-z0-9])key-[a-f0-9]{32}(?![A-Za-z0-9])"), "critical", "[MAILGUN_KEY]"),
    ("Twilio API Key", re.compile(r"(?<![A-Za-z0-9])SK[a-f0-9]{32}(?![A-Za-z0-9])"), "critical", "[TWILIO_KEY]"),
    ("SendGrid API Key", re.compile(r"(?<![A-Za-z0-9])SG\.[a-zA-Z0-9\-_]{22}\.[a-zA-Z0-9\-_]{43}(?![A-Za-z0-9])"), "critical", "[SENDGRID_KEY]"),

    # High severity — tokens / bearer auth
    ("JWT Token", re.compile(r"(?<![A-Za-z0-9\-_])eyJ[a-zA-Z0-9\-_]{10,}\.(eyJ[a-zA-Z0-9\-_]{10,}|[a-zA-Z0-9\-_]{10,})\.[a-zA-Z0-9\-_]{10,}(?![A-Za-z0-9\-_])"), "high", "[JWT]"),
    ("Bearer Token", re.compile(r"(?i)bearer\s+[a-zA-Z0-9\-_.]{16,120}"), "high", "[BEARER_TOKEN]"),
    ("Basic Auth", re.compile(r"(?i)basic\s+[a-zA-Z0-9=+/]{8,120}"), "high", "[BASIC_AUTH]"),
    ("Private SSH Key", re.compile(r"-----BEGIN\s+(?:RSA|DSA|EC|OPENSSH)\s+PRIVATE\s+KEY-----"), "high", "[SSH_KEY]"),
    ("PGP Private Key", re.compile(r"-----BEGIN\s+PGP\s+PRIVATE\s+KEY\s+BLOCK-----"), "high", "[PGP_KEY]"),
    ("PKCS8 Private Key", re.compile(r"-----BEGIN\s+PRIVATE\s+KEY-----"), "high", "[PRIVATE_KEY]"),
    ("PKCS8 Encrypted Key", re.compile(r"-----BEGIN\s+ENCRYPTED\s+PRIVATE\s+KEY-----"), "high", "[ENCRYPTED_KEY]"),
    ("Certificate", re.compile(r"-----BEGIN\s+CERTIFICATE-----"), "medium", "[CERTIFICATE]"),
    ("RSA Public Key", re.compile(r"-----BEGIN\s+RSA\s+PUBLIC\s+KEY-----"), "medium", "[PUBKEY]"),

    # Medium severity — configurable credentials
    ("Password in URL", re.compile(r"(?i)(?:password|passwd|pwd|secret)=[^&\s]{4,80}"), "medium", "[PASSWORD]"),
    ("Database URL", re.compile(r"(?i)(?:postgres(?:ql)?|mysql|mongodb|redis|rediss)://[^@\s]+@"), "medium", "[DB_URL]"),
    ("Connection String", re.compile(r"(?i)(?:Server|Host)=[^;]+;.*(?:User\s*Id|Password)="), "medium", "[CONN_STR]"),
    ("Azure connection string", re.compile(r"(?i)DefaultEndpointsProtocol=https;AccountName=[^;]+;AccountKey=[^;]+"), "critical", "[AZURE_CONN]"),
    ("S3 URL with credentials", re.compile(r"(?i)s3://[A-Z0-9]{16,20}:[a-zA-Z0-9+/=]{30,50}@"), "critical", "[S3_CRED]"),
]


def scan_secrets(text: str) -> List[SecretFinding]:
    """Scan text for known secret patterns.

    Includes input size limits (1 MB) and regex timeout (5 s) to prevent ReDoS.

    Args:
        text: The text content to scan.

    Returns:
        List of SecretFinding dataclass instances, ordered by position.
    """
    if len(text) > _MAX_SCAN_SIZE:
        logger.warning(
            "Input too large for secret scanning (%d bytes, max %d); truncating",
            len(text), _MAX_SCAN_SIZE,
        )
        text = text[:_MAX_SCAN_SIZE]

    findings: List[SecretFinding] = []
    seen_spans: set = set()
    deadline = time.monotonic() + _REGEX_TIMEOUT

    for pattern_name, regex, severity, redacted in _SECRET_PATTERNS:
        if time.monotonic() > deadline:
            logger.warning("Secret scanning timed out after %.1fs — returning partial results", _REGEX_TIMEOUT)
            break
        try:
            for match in regex.finditer(text):
                if time.monotonic() > deadline:
                    break
                span = match.span()
                if span in seen_spans:
                    continue
                seen_spans.add(span)

                start = span[0]
                end = span[1]
                context_start = max(0, start - 30)
                context_end = min(len(text), end + 30)

                line_number = text[:start].count("\n") + 1

                findings.append(SecretFinding(
                    secret_type=pattern_name,
                    pattern_name=pattern_name,
                    context=text[context_start:context_end].replace("\n", " "),
                    start=start,
                    end=end,
                    severity=severity,
                    line_number=line_number,
                    redacted=redacted,
                ))
        except re.error as exc:
            logger.warning("Regex error on pattern '%s': %s — skipping", pattern_name, exc)

    findings.sort(key=lambda f: f.start)
    return findings


def redact_secrets(text: str, findings: List[SecretFinding]) -> str:
    """Redact secret findings from text by replacing matches with placeholders.

    Args:
        text: Original text
        findings: List of SecretFinding instances from scan_secrets()

    Returns:
        Text with secrets replaced by their redacted placeholders.
    """
    if not findings:
        return text

    # Sort by start position descending to avoid offset shifts
    sorted_findings = sorted(findings, key=lambda f: f.start, reverse=True)
    result = text
    for f in sorted_findings:
        result = result[:f.start] + f.redacted + result[f.end:]
    return result
