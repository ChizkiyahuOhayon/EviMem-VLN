from __future__ import annotations

from .base import MemoryBackend, MemoryBackendConfig, MemoryBatch, MemoryTokens


def create_memory_backend(config: MemoryBackendConfig) -> MemoryBackend:
    if config.name == "gavln":
        from .gavln_backend import GAVLNMemoryBackend

        return GAVLNMemoryBackend(config)
    if config.name == "evimem":
        from evimem.torch_backend import EviMemTorchBackend

        return EviMemTorchBackend(config)
    raise ValueError(f"unsupported memory backend: {config.name}")


__all__ = [
    "MemoryBackend",
    "MemoryBackendConfig",
    "MemoryBatch",
    "MemoryTokens",
    "create_memory_backend",
]
