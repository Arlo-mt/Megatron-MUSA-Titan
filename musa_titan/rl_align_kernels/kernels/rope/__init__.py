"""Paired RoPE operators; importing this package installs no patches."""

from .triton_rope import rope_forward

__all__ = ["rope_forward"]
