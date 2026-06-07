"""Tests for cross-restart rehydration of the disk-backed prompt KV cache."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import types
from typing import Any


class _FakeCacheLayer:
    """Serializable fake cache layer with MLX-like byte accounting."""

    def __init__(self, value: str, nbytes: int = 10) -> None:
        self.value = value
        self.nbytes = nbytes


def _load_prompt_cache_module(monkeypatch: Any, *, trimmable: bool = False) -> Any:
    """Import ``app.utils.prompt_cache`` with a fake MLX cache module."""
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_models = types.ModuleType("mlx_lm.models")
    fake_cache = types.ModuleType("mlx_lm.models.cache")

    fake_cache.can_trim_prompt_cache = lambda _cache: trimmable

    def trim_prompt_cache(cache: list[Any], n: int) -> int:
        del cache[-n:]
        return n

    fake_cache.trim_prompt_cache = trim_prompt_cache
    fake_models.cache = fake_cache
    fake_mlx_lm.models = fake_models

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", fake_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache)
    sys.modules.pop("app.utils.prompt_cache", None)
    module = importlib.import_module("app.utils.prompt_cache")
    return importlib.reload(module)


def _sidecars(cache_dir: Path) -> list[Path]:
    """Return the metadata sidecars present in ``cache_dir``."""
    return sorted(cache_dir.glob("*.meta.json"))


def test_insert_writes_sidecar_next_to_payload(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Each inserted entry should persist a JSON sidecar with its trie key."""
    module = _load_prompt_cache_module(monkeypatch)
    cache = module.LRUPromptCache(max_size=10, cache_dir=tmp_path, fingerprint="fp-A")

    cache.insert_cache([1, 2, 3], [_FakeCacheLayer("a")], cache_type="user", source="batch")

    sidecars = _sidecars(tmp_path)
    assert len(sidecars) == 1
    meta = json.loads(sidecars[0].read_text())
    assert meta["fingerprint"] == "fp-A"
    assert meta["tokens"] == [1, 2, 3]
    assert meta["cache_type"] == "user"
    assert meta["source"] == "batch"
    assert meta["version"] == module.CACHE_FORMAT_VERSION


def test_rehydrate_adopts_matching_entries(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A new instance over the same dir should serve a prior run's payload."""
    module = _load_prompt_cache_module(monkeypatch)

    first = module.LRUPromptCache(max_size=10, cache_dir=tmp_path, fingerprint="fp-A")
    first.insert_cache([1, 2, 3], [_FakeCacheLayer("hello")])

    # Simulate a restart: a brand-new instance over the same directory.
    second = module.LRUPromptCache(max_size=10, cache_dir=tmp_path, fingerprint="fp-A")

    assert len(second) == 1
    result, rest = second.fetch_nearest_cache([1, 2, 3])
    assert rest == []
    assert result is not None
    assert result[0].value == "hello"


def test_rehydrate_does_not_eagerly_load_payloads(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Rehydration should index metadata only, never deserialize payloads."""
    module = _load_prompt_cache_module(monkeypatch)

    first = module.LRUPromptCache(max_size=10, cache_dir=tmp_path, fingerprint="fp-A")
    first.insert_cache([1, 2, 3], [_FakeCacheLayer("lazy")])

    second = module.LRUPromptCache(max_size=10, cache_dir=tmp_path, fingerprint="fp-A")

    # The adopted entry carries a file path but no in-memory payload.
    entry = second._trie.get([1, 2, 3])
    assert entry.file_path.exists()
    assert not hasattr(entry, "prompt_cache")
    assert second.nbytes == 10


def test_rehydrate_skips_and_cleans_fingerprint_mismatch(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Entries from a different model/config must not be adopted or retained."""
    module = _load_prompt_cache_module(monkeypatch)

    first = module.LRUPromptCache(max_size=10, cache_dir=tmp_path, fingerprint="fp-A")
    first.insert_cache([1, 2, 3], [_FakeCacheLayer("foreign")])

    # Restart under a different fingerprint (e.g. a model or quantization change).
    second = module.LRUPromptCache(max_size=10, cache_dir=tmp_path, fingerprint="fp-B")

    assert len(second) == 0
    assert second.fetch_nearest_cache([1, 2, 3]) == (None, [1, 2, 3])
    # The mismatched payload and its sidecar are swept on startup.
    assert list(tmp_path.glob("*.pkl")) == []
    assert _sidecars(tmp_path) == []


def test_rehydrate_skips_and_cleans_version_mismatch(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Sidecars from an incompatible schema version are discarded."""
    module = _load_prompt_cache_module(monkeypatch)

    first = module.LRUPromptCache(max_size=10, cache_dir=tmp_path, fingerprint="fp-A")
    first.insert_cache([4, 5], [_FakeCacheLayer("stale")])

    # Rewrite the sidecar with a bumped (incompatible) version.
    sidecar = _sidecars(tmp_path)[0]
    meta = json.loads(sidecar.read_text())
    meta["version"] = module.CACHE_FORMAT_VERSION + 1
    sidecar.write_text(json.dumps(meta))

    second = module.LRUPromptCache(max_size=10, cache_dir=tmp_path, fingerprint="fp-A")

    assert len(second) == 0
    assert list(tmp_path.glob("*.pkl")) == []
    assert _sidecars(tmp_path) == []


def test_rehydrate_discards_corrupt_sidecar(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """An unreadable sidecar should be discarded without aborting rehydrate."""
    module = _load_prompt_cache_module(monkeypatch)

    first = module.LRUPromptCache(max_size=10, cache_dir=tmp_path, fingerprint="fp-A")
    first.insert_cache([7], [_FakeCacheLayer("ok")])
    first.insert_cache([8], [_FakeCacheLayer("corrupt")])

    # Corrupt one sidecar's JSON; the other should still be adopted.
    corrupt = next(s for s in _sidecars(tmp_path) if json.loads(s.read_text())["tokens"] == [8])
    corrupt.write_text("{not json")

    second = module.LRUPromptCache(max_size=10, cache_dir=tmp_path, fingerprint="fp-A")

    assert len(second) == 1
    result, rest = second.fetch_nearest_cache([7])
    assert rest == []
    assert result is not None and result[0].value == "ok"
    assert second.fetch_nearest_cache([8]) == (None, [8])


def test_rehydrate_drops_sidecar_without_payload(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A sidecar whose payload is missing should be treated as an orphan."""
    module = _load_prompt_cache_module(monkeypatch)

    first = module.LRUPromptCache(max_size=10, cache_dir=tmp_path, fingerprint="fp-A")
    first.insert_cache([9], [_FakeCacheLayer("vanishing")])
    payload = next(iter(tmp_path.glob("*.pkl")))
    payload.unlink()

    second = module.LRUPromptCache(max_size=10, cache_dir=tmp_path, fingerprint="fp-A")

    assert len(second) == 0
    assert _sidecars(tmp_path) == []


def test_rehydrate_respects_max_size(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Adopting more entries than max_size should trim back down on startup."""
    module = _load_prompt_cache_module(monkeypatch)

    first = module.LRUPromptCache(max_size=10, cache_dir=tmp_path, fingerprint="fp-A")
    first.insert_cache([1], [_FakeCacheLayer("one")])
    first.insert_cache([2], [_FakeCacheLayer("two")])
    first.insert_cache([3], [_FakeCacheLayer("three")])

    second = module.LRUPromptCache(max_size=2, cache_dir=tmp_path, fingerprint="fp-A")

    assert len(second) == 2
    # Two payloads survive; the oldest was trimmed along with its sidecar.
    assert len(list(tmp_path.glob("*.pkl"))) == 2
    assert len(_sidecars(tmp_path)) == 2


def test_eviction_removes_sidecar(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Evicting an entry should delete both its payload and its sidecar."""
    module = _load_prompt_cache_module(monkeypatch)
    cache = module.LRUPromptCache(max_size=1, cache_dir=tmp_path, fingerprint="fp-A")

    cache.insert_cache([1], [_FakeCacheLayer("old")])
    cache.insert_cache([2], [_FakeCacheLayer("new")])

    # Only the surviving entry's payload + sidecar remain.
    assert len(list(tmp_path.glob("*.pkl"))) == 1
    assert len(_sidecars(tmp_path)) == 1


def test_build_cache_fingerprint_is_config_sensitive(
    monkeypatch: Any,
) -> None:
    """The fingerprint should change when any cache-affecting setting changes."""
    module = _load_prompt_cache_module(monkeypatch)

    base = module.build_cache_fingerprint(
        model_path="m",
        kv_bits=None,
        kv_group_size=64,
        quantized_kv_start=0,
    )
    same = module.build_cache_fingerprint(
        model_path="m",
        kv_bits=None,
        kv_group_size=64,
        quantized_kv_start=0,
    )
    other_model = module.build_cache_fingerprint(
        model_path="other",
        kv_bits=None,
        kv_group_size=64,
        quantized_kv_start=0,
    )
    other_kv = module.build_cache_fingerprint(
        model_path="m",
        kv_bits=8,
        kv_group_size=64,
        quantized_kv_start=0,
    )

    assert base == same
    assert base != other_model
    assert base != other_kv
