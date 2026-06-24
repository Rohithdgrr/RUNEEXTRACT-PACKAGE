"""Maturity labels for RuneExtract features.

Provides decorators to mark features as stable, beta, or experimental.
Users can inspect maturity via ``get_feature_maturity()`` or at runtime
via the ``__maturity__`` attribute.
"""

import enum
import functools
import warnings
from typing import Callable, Optional


class MaturityLevel(str, enum.Enum):
    STABLE = "stable"
    BETA = "beta"
    EXPERIMENTAL = "experimental"


_MATURITY_REGISTRY: dict[str, MaturityLevel] = {}


def get_feature_maturity(name: str) -> Optional[MaturityLevel]:
    """Look up the maturity level of a named feature.

    Args:
        name: Fully-qualified feature name (e.g. ``rag.hierarchical``).

    Returns:
        The ``MaturityLevel`` if registered, else ``None``.
    """
    return _MATURITY_REGISTRY.get(name)


def list_experimental_features() -> list[str]:
    """Return names of all registered experimental features."""
    return [k for k, v in _MATURITY_REGISTRY.items() if v == MaturityLevel.EXPERIMENTAL]


def list_beta_features() -> list[str]:
    """Return names of all registered beta features."""
    return [k for k, v in _MATURITY_REGISTRY.items() if v == MaturityLevel.BETA]


def experimental(name: Optional[str] = None) -> Callable:
    """Decorator marking a class or function as experimental.

    Experimental features may have bugs, incomplete APIs, or change
    without notice. Warnings can be enabled via
    ``warnings.filterwarnings("always", category=UserWarning)``.

    Args:
        name: Optional feature name for the registry (defaults to
            ``module.qualname``).
    """
    def decorator(obj: Callable) -> Callable:
        feature_name = name or f"{obj.__module__}.{obj.__qualname__}"
        _MATURITY_REGISTRY[feature_name] = MaturityLevel.EXPERIMENTAL
        obj.__maturity__ = MaturityLevel.EXPERIMENTAL

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"'{feature_name}' is experimental — API may change without notice.",
                UserWarning,
                stacklevel=2,
            )
            return obj(*args, **kwargs)

        return wrapper
    return decorator


def beta(name: Optional[str] = None) -> Callable:
    """Decorator marking a class or function as beta.

    Beta features are functional but may have edge cases or API
    refinements before becoming stable.
    """
    def decorator(obj: Callable) -> Callable:
        feature_name = name or f"{obj.__module__}.{obj.__qualname__}"
        _MATURITY_REGISTRY[feature_name] = MaturityLevel.BETA
        obj.__maturity__ = MaturityLevel.BETA

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"'{feature_name}' is in beta — may have edge cases.",
                UserWarning,
                stacklevel=2,
            )
            return obj(*args, **kwargs)

        return wrapper
    return decorator


def stable(name: Optional[str] = None) -> Callable:
    """Decorator marking a class or function as stable."""
    def decorator(obj: Callable) -> Callable:
        feature_name = name or f"{obj.__module__}.{obj.__qualname__}"
        _MATURITY_REGISTRY[feature_name] = MaturityLevel.STABLE
        obj.__maturity__ = MaturityLevel.STABLE
        return obj
    return decorator
