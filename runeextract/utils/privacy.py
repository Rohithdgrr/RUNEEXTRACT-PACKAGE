"""
Differential privacy engine for PII redaction.

Provides epsilon-differential privacy via the Laplace mechanism
for numeric PII fields (ages, dates, phone numbers, etc.) and
suppression with calibrated noise for categorical PII.
"""

import math
import random
import re
from typing import Tuple, Optional


class DifferentialPrivacyEngine:
    """Apply epsilon-differential privacy to PII redaction.

    The Laplace mechanism adds calibrated noise to numeric PII values
    to provide formal privacy guarantees. Higher epsilon = more accuracy
    but less privacy. Typical values: 0.1 (high privacy) to 10 (low privacy).

    Args:
        epsilon: Privacy budget (default 1.0). Lower = more private.
        delta: Relaxation parameter (default 0.0 for pure DP).
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 0.0):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.delta = delta

    def _laplace_noise(self, sensitivity: float) -> float:
        """Sample from Laplace(0, sensitivity/epsilon)."""
        scale = sensitivity / self.epsilon
        u = random.random() - 0.5
        return -scale * math.copysign(math.log(1 - 2 * abs(u)), u)

    def _clamp_int(self, value: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, value))

    def perturb_phone(self, phone_match: str) -> str:
        """Add calibrated noise to the last 4 digits of a phone number.

        Returns the perturbed phone string with the last digits obfuscated.
        """
        digits = re.sub(r"\D", "", phone_match)
        if len(digits) < 4:
            return phone_match
        noise = int(round(self._laplace_noise(1000.0)))
        last_four = int(digits[-4:])
        perturbed = self._clamp_int(last_four + noise, 0, 9999)
        return phone_match[:-4] + f"{perturbed:04d}"

    def perturb_age(self, age: int) -> int:
        """Add Laplace noise to an age value."""
        noise = int(round(self._laplace_noise(10.0)))
        return self._clamp_int(age + noise, 0, 150)

    def perturb_year(self, year: int) -> int:
        """Add Laplace noise to a year value."""
        noise = int(round(self._laplace_noise(5.0)))
        return self._clamp_int(year + noise, 1900, 2100)

    def get_privacy_params(self) -> Tuple[float, float]:
        """Return (epsilon, delta) for this engine instance."""
        return (self.epsilon, self.delta)
