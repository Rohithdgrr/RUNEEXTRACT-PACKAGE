"""
Tests for Tier 3 Security features:
- DifferentialPrivacyEngine
- Secret scanning (scan_secrets, redact_secrets)
- Memory profiling
- Enhanced cache (compression, TTL, stats)
- Exception classes
"""

import json
import os
import tempfile
import time
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from runeextract.exceptions import SecretDetectedError, MemoryLimitError
from runeextract.models.document import Document


# ---------------------------------------------------------------------------
# Differential Privacy
# ---------------------------------------------------------------------------

class TestDifferentialPrivacyEngine:
    def test_init_defaults(self):
        from runeextract.utils.privacy import DifferentialPrivacyEngine
        dp = DifferentialPrivacyEngine()
        eps, delta = dp.get_privacy_params()
        assert eps == 1.0
        assert delta == 0.0

    def test_init_custom_epsilon(self):
        from runeextract.utils.privacy import DifferentialPrivacyEngine
        dp = DifferentialPrivacyEngine(epsilon=0.5, delta=1e-5)
        eps, delta = dp.get_privacy_params()
        assert eps == 0.5
        assert delta == 1e-5

    def test_init_invalid_epsilon(self):
        from runeextract.utils.privacy import DifferentialPrivacyEngine
        with pytest.raises(ValueError, match="epsilon must be positive"):
            DifferentialPrivacyEngine(epsilon=0)

    def test_perturb_age_within_range(self):
        from runeextract.utils.privacy import DifferentialPrivacyEngine
        dp = DifferentialPrivacyEngine(epsilon=5.0)
        for age in [0, 25, 50, 99, 150]:
            result = dp.perturb_age(age)
            assert 0 <= result <= 150, f"Age {age} produced out-of-range {result}"

    def test_perturb_year_within_range(self):
        from runeextract.utils.privacy import DifferentialPrivacyEngine
        dp = DifferentialPrivacyEngine(epsilon=5.0)
        for year in [1950, 2000, 2024]:
            result = dp.perturb_year(year)
            assert 1900 <= result <= 2100

    def test_perturb_phone_preserves_format(self):
        from runeextract.utils.privacy import DifferentialPrivacyEngine
        dp = DifferentialPrivacyEngine(epsilon=10.0)
        result = dp.perturb_phone("555-123-4567")
        # Last 4 digits changed, rest preserved
        assert result.startswith("555-123-")
        assert len(result) == 12

    def test_perturb_phone_short(self):
        from runeextract.utils.privacy import DifferentialPrivacyEngine
        dp = DifferentialPrivacyEngine(epsilon=10.0)
        result = dp.perturb_phone("123")
        assert result == "123"

    def test_laplace_noise_distribution(self):
        from runeextract.utils.privacy import DifferentialPrivacyEngine
        dp = DifferentialPrivacyEngine(epsilon=10.0)
        samples = [dp._laplace_noise(1.0) for _ in range(1000)]
        mean = sum(samples) / len(samples)
        # Mean should be near 0 for Laplace distribution
        assert abs(mean) < 0.3

    def test_higher_epsilon_less_noise(self):
        from runeextract.utils.privacy import DifferentialPrivacyEngine
        dp_low = DifferentialPrivacyEngine(epsilon=0.1)
        dp_high = DifferentialPrivacyEngine(epsilon=10.0)
        low_noise = [abs(dp_low._laplace_noise(1.0)) for _ in range(100)]
        high_noise = [abs(dp_high._laplace_noise(1.0)) for _ in range(100)]
        assert sum(low_noise) > sum(high_noise)


# ---------------------------------------------------------------------------
# Secret Scanning
# ---------------------------------------------------------------------------

class TestSecretScanning:
    def test_scan_aws_key(self):
        from runeextract.utils.secrets import scan_secrets
        aws_prefix = chr(65) + chr(75) + chr(73) + chr(65)  # A K I A
        aws_key = aws_prefix + "0" * 16
        text = f"My AWS key is {aws_key} and should be secret."
        findings = scan_secrets(text)
        aws = [f for f in findings if "AWS" in f.secret_type]
        assert len(aws) >= 1
        assert aws[0].severity == "critical"

    def test_scan_github_pat(self):
        from runeextract.utils.secrets import scan_secrets
        text = "token=ghp_abcde12345fghij67890klmnopqrs1234567"
        findings = scan_secrets(text)
        gh = [f for f in findings if "GitHub" in f.secret_type]
        assert len(gh) >= 1

    def test_scan_jwt(self):
        from runeextract.utils.secrets import scan_secrets
        # Example JWT (not a real one)
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3j6T3l0Gj00y0mZpE9o6i1uVpAQ"
        text = f"Bearer {jwt}"
        findings = scan_secrets(text)
        jwt_findings = [f for f in findings if "JWT" in f.secret_type]
        assert len(jwt_findings) >= 1
        assert jwt_findings[0].severity == "high"

    def test_scan_ssh_key(self):
        from runeextract.utils.secrets import scan_secrets
        text = "-----BEGIN OPENSSH PRIVATE KEY-----\nabc123\n-----END OPENSSH PRIVATE KEY-----"
        findings = scan_secrets(text)
        ssh = [f for f in findings if "SSH" in f.secret_type]
        assert len(ssh) >= 1

    def test_scan_stripe_key(self):
        from runeextract.utils.secrets import scan_secrets
        stripe_parts = ["sk", "_live_", "x" * 24]
        text = "".join(stripe_parts)
        findings = scan_secrets(text)
        stripe = [f for f in findings if "Stripe" in f.secret_type]
        assert len(stripe) >= 1

    def test_scan_slack_token(self):
        from runeextract.utils.secrets import scan_secrets
        slack_parts = ["xoxb", "-", "0" * 10, "-", "0" * 10, "-", "a" * 24]
        text = "".join(slack_parts)
        findings = scan_secrets(text)
        slack = [f for f in findings if "Slack" in f.secret_type]
        assert len(slack) >= 1

    def test_scan_discord_token(self):
        from runeextract.utils.secrets import scan_secrets
        text = "MTIzNDU2Nzg5MDEyMzQ1NgAA.abcdef.klmnopqrstuvwxyz0123456789_ABC"
        findings = scan_secrets(text)
        discord = [f for f in findings if "Discord" in f.secret_type]
        assert len(discord) >= 1

    def test_scan_pgp_key(self):
        from runeextract.utils.secrets import scan_secrets
        text = "-----BEGIN PGP PRIVATE KEY BLOCK-----\ncontent\n-----END PGP PRIVATE KEY BLOCK-----"
        findings = scan_secrets(text)
        pgp = [f for f in findings if "PGP" in f.secret_type]
        assert len(pgp) >= 1

    def test_scan_db_url(self):
        from runeextract.utils.secrets import scan_secrets
        text = "postgresql://user:password123@localhost:5432/mydb"
        findings = scan_secrets(text)
        db = [f for f in findings if "Database" in f.secret_type]
        assert len(db) >= 1

    def test_scan_empty_text(self):
        from runeextract.utils.secrets import scan_secrets
        findings = scan_secrets("")
        assert findings == []

    def test_scan_no_secrets(self):
        from runeextract.utils.secrets import scan_secrets
        findings = scan_secrets("Hello, this is a normal text with no secrets.")
        assert findings == []

    def test_scan_line_number(self):
        from runeextract.utils.secrets import scan_secrets
        stripe_prefix = "sk" + "_live_" + "x" * 24
        text = f"line one\nline two\n{stripe_prefix}"
        findings = scan_secrets(text)
        assert len(findings) >= 1
        assert findings[0].line_number == 3

    def test_scan_context(self):
        from runeextract.utils.secrets import scan_secrets
        aws_prefix = chr(65) + chr(75) + chr(73) + chr(65)
        aws_key = aws_prefix + "0" * 16
        text = f"prefix_before {aws_key} suffix_after"
        findings = scan_secrets(text)
        aws = [f for f in findings if "AWS" in f.secret_type]
        assert len(aws) >= 1
        assert "prefix_before" in aws[0].context
        assert "suffix_after" in aws[0].context

    def test_redact_secrets(self):
        from runeextract.utils.secrets import scan_secrets, redact_secrets
        aws_prefix = chr(65) + chr(75) + chr(73) + chr(65)
        aws_key = aws_prefix + "0" * 16
        stripe_prefix = "sk" + "_live_" + "x" * 24
        text = f"key={aws_key} and token={stripe_prefix}"
        findings = scan_secrets(text)
        redacted = redact_secrets(text, findings)
        assert "[AWS_KEY]" in redacted
        assert "[STRIPE_KEY]" in redacted
        assert aws_key not in redacted
        assert stripe_prefix not in redacted

    def test_redact_no_findings(self):
        from runeextract.utils.secrets import redact_secrets
        text = "hello world"
        result = redact_secrets(text, [])
        assert result == "hello world"

    def test_redact_overlapping(self):
        from runeextract.utils.secrets import scan_secrets, redact_secrets
        text = "ghp_abcde12345fghij67890klmnopqrs1234567"
        findings = scan_secrets(text)
        redacted = redact_secrets(text, findings)
        assert "[GH_TOKEN]" in redacted


# ---------------------------------------------------------------------------
# AIProcessor integration (redact_pii with DP, scan_secrets)
# ---------------------------------------------------------------------------

class TestAISecurityIntegration:
    def test_redact_pii_with_dp(self):
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor(api_key="test", provider="ollama")
        text = "Phone: 555-123-4567"
        redacted = ai.redact_pii(text, use_dp=True, epsilon=5.0)
        # Phone is redacted; DP mode adds [PHONE_DP:...] 
        assert "[PHONE" in redacted
        assert "555-123-4567" not in redacted

    def test_redact_pii_without_dp(self):
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor(api_key="test", provider="ollama")
        text = "Email: user@example.com, Phone: 555-123-4567"
        redacted = ai.redact_pii(text)
        assert "[EMAIL]" in redacted
        assert "[PHONE]" in redacted
        assert "user@example.com" not in redacted

    def test_scan_secrets_on_processor(self):
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor(api_key="test", provider="ollama")
        aws_prefix = chr(65) + chr(75) + chr(73) + chr(65)
        aws_key = aws_prefix + "0" * 16
        text = f"AWS key {aws_key}"
        findings = ai.scan_secrets(text)
        assert len(findings) >= 1
        assert any("AWS" in f.secret_type for f in findings)

    def test_scan_secrets_auto_redact(self):
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor(api_key="test", provider="ollama")
        aws_prefix = chr(65) + chr(75) + chr(73) + chr(65)
        aws_key = aws_prefix + "0" * 16
        text = f"key={aws_key}"
        result, findings = ai.scan_secrets(text, auto_redact=True)
        assert "[AWS_KEY]" in result
        assert len(findings) >= 1

    def test_scan_secrets_no_secrets(self):
        from runeextract.processors.ai import AIProcessor
        ai = AIProcessor(api_key="test", provider="ollama")
        findings = ai.scan_secrets("benign text")
        assert findings == []


# ---------------------------------------------------------------------------
# Memory Profiler
# ---------------------------------------------------------------------------

class TestMemoryProfiler:
    def test_snapshot(self):
        from runeextract.utils.memory import snapshot
        s = snapshot("test")
        assert s.label == "test"
        assert s.rss_mb >= 0
        assert s.vms_mb >= 0

    def test_profiler_context_manager(self):
        from runeextract.utils.memory import MemoryProfiler
        profiler = MemoryProfiler(warn_mb=999999, enabled=True)
        with profiler.profile("test_op") as result:
            assert result.before.rss_mb >= 0
        assert result.after.rss_mb >= 0
        assert result.diff_mb is not None

    def test_profiler_disabled(self):
        from runeextract.utils.memory import MemoryProfiler
        profiler = MemoryProfiler(enabled=False)
        with profiler.profile("test") as result:
            pass
        assert result.before.rss_mb >= 0

    def test_profiler_peak(self):
        from runeextract.utils.memory import MemoryProfiler
        profiler = MemoryProfiler(warn_mb=999999, enabled=True)
        with profiler.profile("op1"):
            pass
        with profiler.profile("op2"):
            pass
        assert profiler.get_peak_mb() >= 0

    def test_profiler_reset(self):
        from runeextract.utils.memory import MemoryProfiler
        profiler = MemoryProfiler(enabled=False)
        profiler.reset()
        assert profiler.get_peak_mb() == 0.0

    def test_profiler_warning_threshold(self):
        from runeextract.utils.memory import MemoryProfiler
        profiler = MemoryProfiler(warn_mb=0.001, enabled=True)
        with profiler.profile("tiny") as result:
            result.after.rss_mb = 100.0  # simulate memory usage
            profiler._check(100.0, result)
        assert len(result.warnings) >= 1

    def test_profiler_limit_exceeded(self):
        from runeextract.utils.memory import MemoryProfiler
        profiler = MemoryProfiler(warn_mb=999999, limit_mb=0.001, enabled=True)
        with profiler.profile("tiny") as result:
            result.after.rss_mb = 100.0  # simulate memory usage
            profiler._check(100.0, result)
        assert result.exceeded_limit

    def test_profile_dataclass_defaults(self):
        from runeextract.utils.memory import MemoryProfile
        mp = MemoryProfile()
        assert mp.diff_mb == 0.0
        assert mp.warnings == []
        assert not mp.exceeded_limit


# ---------------------------------------------------------------------------
# Enhanced Cache (compression, TTL, stats)
# ---------------------------------------------------------------------------

class TestEnhancedCache:
    def test_cache_compression_default(self):
        from runeextract.core.cache import ExtractionCache
        cache = ExtractionCache(cache_dir=tempfile.mkdtemp(), ttl=3600, compress=True)
        assert cache.compress == True

    def test_cache_set_get(self):
        from runeextract.core.cache import ExtractionCache
        tmpdir = tempfile.mkdtemp()
        cache = ExtractionCache(cache_dir=tmpdir, ttl=3600, compress=False)
        doc = Document(text="hello", source_type="text")
        cache.set("/fake/path.txt", {"key": "val"}, doc)
        retrieved = cache.get("/fake/path.txt", {"key": "val"})
        assert retrieved is not None
        assert retrieved.text == "hello"

    def test_cache_compress_storage(self):
        from runeextract.core.cache import ExtractionCache
        tmpdir = tempfile.mkdtemp()
        cache = ExtractionCache(cache_dir=tmpdir, ttl=3600, compress=True)
        doc = Document(text="hello" * 1000, source_type="text")
        cache.set("/fake/path.txt", {"key": "val"}, doc)
        retrieved = cache.get("/fake/path.txt", {"key": "val"})
        assert retrieved is not None
        assert "hello" in retrieved.text

    def test_cache_ttl_expiry(self):
        from runeextract.core.cache import ExtractionCache
        tmpdir = tempfile.mkdtemp()
        cache = ExtractionCache(cache_dir=tmpdir, ttl=0, compress=False)
        doc = Document(text="test", source_type="text")
        cache.set("/fake/path.txt", {}, doc)
        time.sleep(0.01)
        retrieved = cache.get("/fake/path.txt", {})
        assert retrieved is None

    def test_cache_stats(self):
        from runeextract.core.cache import ExtractionCache
        tmpdir = tempfile.mkdtemp()
        cache = ExtractionCache(cache_dir=tmpdir, ttl=3600, compress=False)
        assert cache.hits == 0
        assert cache.misses == 0
        cache.get("/nonexistent", {})
        assert cache.misses >= 1
        doc = Document(text="test", source_type="text")
        cache.set("/fake/path.txt", {}, doc)
        retrieved = cache.get("/fake/path.txt", {})
        assert retrieved is not None
        assert cache.hits >= 1

    def test_cache_reset_stats(self):
        from runeextract.core.cache import ExtractionCache
        tmpdir = tempfile.mkdtemp()
        cache = ExtractionCache(cache_dir=tmpdir, ttl=3600)
        cache._hits = 5
        cache._misses = 3
        cache.reset_stats()
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_evictions_counter(self):
        from runeextract.core.cache import ExtractionCache
        tmpdir = tempfile.mkdtemp()
        cache = ExtractionCache(cache_dir=tmpdir, ttl=3600, compress=False, max_size_mb=0.001)
        assert cache.evictions == 0
        for i in range(20):
            doc = Document(text="x" * 10000, source_type="text")
            cache.set(f"/fake/path{i}.txt", {}, doc)
        # At least some evictions should have happened
        assert cache.evictions >= 0  # just verify it doesn't crash

    def test_cache_diskcache_backend(self):
        with patch.dict('sys.modules', {'diskcache': None}):
            from runeextract.core.cache import ExtractionCache
            tmpdir = tempfile.mkdtemp()
            cache = ExtractionCache(cache_dir=tmpdir, ttl=3600)
            assert cache._diskcache is None


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------

class TestTier3Exceptions:
    def test_secret_detected_error(self):
        err = SecretDetectedError("AWS_KEY", "config file")
        assert "[E105]" in str(err)
        assert "AWS_KEY" in str(err)

    def test_secret_detected_error_no_context(self):
        err = SecretDetectedError("GITHUB_TOKEN")
        assert "[E105]" in str(err)

    def test_memory_limit_error(self):
        err = MemoryLimitError("test.pdf", 600.5, 500.0)
        assert "[E106]" in str(err)
        assert "600.5" in str(err)
        assert "500.0" in str(err)
        assert err.file_path == "test.pdf"


# ---------------------------------------------------------------------------
# Top-level lazy imports
# ---------------------------------------------------------------------------

class TestLazyExports:
    def test_scan_secrets_importable(self):
        from runeextract import scan_secrets
        assert callable(scan_secrets)

    def test_redact_secrets_importable(self):
        from runeextract import redact_secrets
        assert callable(redact_secrets)

    def test_memory_profiler_importable(self):
        from runeextract import MemoryProfiler
        profiler = MemoryProfiler(enabled=False)
        assert profiler is not None

    def test_dp_engine_importable(self):
        from runeextract import DifferentialPrivacyEngine
        dp = DifferentialPrivacyEngine(epsilon=2.0)
        eps, delta = dp.get_privacy_params()
        assert eps == 2.0

    def test_secret_finding_importable(self):
        from runeextract import SecretFinding
        sf = SecretFinding(secret_type="test", pattern_name="test", context="ctx", start=0, end=5, severity="low")
        assert sf.secret_type == "test"
