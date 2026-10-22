"""Model adapters: translate canonical types to/from each provider. The ONLY place provider-specific code may live."""

from harness.adapters.registry import AdapterSpec, Provider, build_adapter

__all__ = ["AdapterSpec", "Provider", "build_adapter"]
