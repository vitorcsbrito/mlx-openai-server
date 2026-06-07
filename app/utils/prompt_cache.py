# modified from https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any
import uuid

from loguru import logger
from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

from . import dill

# Bumped whenever the on-disk sidecar schema or payload layout changes. Sidecars
# written under a different version are ignored (and cleaned up) on rehydrate.
CACHE_FORMAT_VERSION = 1


def build_cache_fingerprint(
    *,
    model_path: str,
    kv_bits: int | None,
    kv_group_size: int,
    quantized_kv_start: int,
) -> str:
    """Return a stable identity for the model + KV-cache configuration.

    A persisted KV cache is only valid for the exact weights, tokenizer, and
    quantization settings that produced it, under a compatible MLX
    serialization layout. The fingerprint folds all of these in so that, on
    restart, only payloads built by an identical configuration are adopted.

    Parameters
    ----------
    model_path : str
        Path or identifier of the loaded model (weights + tokenizer identity).
    kv_bits : int | None
        KV-cache quantization bit width, or ``None`` when disabled.
    kv_group_size : int
        KV-cache quantization group size.
    quantized_kv_start : int
        Step at which quantized KV caching begins.

    Returns
    -------
    str
        A short hex digest uniquely identifying the configuration.
    """
    try:
        import mlx.core as mx  # noqa: PLC0415

        mlx_version = getattr(mx, "__version__", "unknown")
    except (ImportError, RuntimeError):
        mlx_version = "unknown"

    parts = [
        str(model_path),
        str(kv_bits),
        str(kv_group_size),
        str(quantized_kv_start),
        mlx_version,
        str(CACHE_FORMAT_VERSION),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass
class PromptTrieResult:
    """Result of searching the trie for a token sequence.

    Parameters
    ----------
    exact : list[int] | None
        Exact matching token sequence, if found.
    shorter : list[int] | None
        Shorter prefix match, if found.
    longer : list[int] | None
        Longer sequence containing the query as a prefix, if found.
    common_prefix : int
        Length of common prefix with matching cache entries.
    """

    exact: list[int] | None
    shorter: list[int] | None
    longer: list[int] | None
    common_prefix: int


class PromptTrie:
    """Prefix trie for storing prompt caches keyed by token sequences."""

    def __init__(self) -> None:
        self._trie: dict[int, Any] = {}

    def add(self, tokens: list[int], value: Any) -> Any:
        """Insert a value and return the previous value if any."""
        current = self._trie
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]
        prev = current.get("__value__", None)
        current["__value__"] = value
        return prev

    def get(self, tokens: list[int]) -> Any:
        """Exact lookup by token sequence."""
        current = self._trie
        for tok in tokens:
            current = current[tok]
        return current["__value__"]

    def pop(self, tokens: list[int]) -> Any:
        """Remove and return the value at the given token sequence."""
        path = [self._trie]
        for tok in tokens:
            path.append(path[-1][tok])
        value = path[-1].pop("__value__")
        for i in range(len(tokens), 0, -1):
            node = path[i]
            parent = path[i - 1]
            tok = tokens[i - 1]
            if len(node) > 0:
                break
            del parent[tok]
        return value

    def pop_prefixes(self, tokens: list[int]) -> list[tuple[int, Any]]:
        """Remove all prefix entries along the path to *tokens*."""
        values = []
        current = self._trie
        for i, tok in enumerate(tokens):
            if "__value__" in current:
                values.append((i, current.pop("__value__")))
            current = current[tok]
        return values

    def search(self, tokens: list[int]) -> PromptTrieResult:
        """Search for exact, shorter, or longer matches."""
        if not self._trie:
            return PromptTrieResult(None, None, None, 0)

        current = self._trie

        if not tokens and "__value__" in current:
            return PromptTrieResult([], None, None, 0)

        # Walk the tokens as far as we can
        last_index = -1
        index = 0
        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "__value__" in current:
                last_index = index
            index += 1

        # Got an exact match
        if last_index == len(tokens) - 1 >= 0:
            return PromptTrieResult(tokens, None, None, 0)

        # Check if we found a prefix at any point
        shorter = None
        if last_index > 0:
            shorter = tokens[: last_index + 1]

        # Check for sequences that are longer (DFS with pruning)
        longer = None
        common_prefix = index
        if index > 0:
            best = None
            stack = [(current, [])]
            while stack:
                current, extra = stack.pop()
                if "__value__" in current:
                    if best is None or len(extra) < len(best):
                        best = extra
                elif best is None or len(extra) < len(best):
                    stack.extend((current[tok], [*extra, tok]) for tok in current)
            if best is not None:
                longer = tokens[:index] + best

        return PromptTrieResult(None, shorter, longer, common_prefix)

    def find_shorter(
        self,
        tokens: list[int],
        predicate: Any,
    ) -> list[int] | None:
        """Find the longest strict-prefix value accepted by ``predicate``."""
        current = self._trie
        best: list[int] | None = None
        prefix: list[int] = []
        for tok in tokens[:-1]:
            if tok not in current:
                break
            current = current[tok]
            prefix.append(tok)
            entry = current.get("__value__")
            if entry is not None and predicate(entry):
                best = prefix[:]
        return best


class LRUPromptCache:
    """Disk-backed LRU cache for MLX prompt KV caches.

    The cache stores token sequences in a trie so it can efficiently find exact
    matches, shorter prefixes, and longer cached sequences that can be trimmed
    down to a requested prefix. Entries are evicted using an LRU policy with
    optional byte-based trimming. Only metadata is kept in memory; prompt-cache
    matrices are serialized to disk on insert and loaded only on cache hits.

    When a caller-supplied ``cache_dir`` already contains payloads from a prior
    run, the in-memory index is rebuilt from each payload's JSON sidecar at
    construction time, so warm prefixes survive a restart. Adoption is gated by
    ``fingerprint``: only entries produced by an identical model and KV-cache
    configuration are reused; mismatched, stale, or corrupt entries are
    discarded (and their files cleaned up) on startup.

    Parameters
    ----------
    max_size : int, optional
        Maximum number of cache entries to retain, by default 10.
    max_bytes : int, optional
        Maximum total bytes to retain across cached prompt caches, by default a
        practically unbounded value.
    cache_dir : str | Path | None, optional
        Directory used to store serialized prompt caches. A process-local
        temporary directory is created (and removed on close) when omitted; a
        caller-supplied directory persists across restarts and is rehydrated.
    fingerprint : str, optional
        Identity of the model and KV-cache configuration that produces these
        payloads. Persisted into each sidecar and required to match for a
        payload to be adopted on restart. Empty by default (no cross-restart
        identity check). See :func:`build_cache_fingerprint`.
    """

    @dataclass
    class CacheEntry:
        """Stored prompt cache metadata."""

        file_path: Path
        nbytes: int
        cache_type: str
        trimmable: bool
        source: str = "nonbatch"

    class CacheOrder:
        """Track cache recency with priority-based eviction."""

        def __init__(self, ordering: list[str] | None = None) -> None:
            if ordering is None:
                ordering = ["assistant", "user", "system"]
            self._ordering = ordering
            self._lrus: dict[str, deque[tuple[int, ...]]] = {k: deque() for k in ordering}

        def __len__(self) -> int:
            return sum(len(lru) for lru in self._lrus.values())

        def push(self, tokens: tuple[int, ...], cache_type: str = "assistant") -> None:
            self._lrus[cache_type].append(tokens)

        def remove(self, tokens: tuple[int, ...]) -> None:
            for cache_type in self._ordering:
                try:
                    self._lrus[cache_type].remove(tokens)
                    break
                except ValueError:
                    pass

        def pop(self) -> tuple[int, ...]:
            """Pop the least-recently-used entry, favouring lower-priority types."""
            i = 0
            while i + 1 < len(self._ordering):
                lru_a = self._lrus[self._ordering[i]]
                lru_b = self._lrus[self._ordering[i + 1]]
                if lru_a and len(lru_a) >= len(lru_b):
                    return lru_a.popleft()
                i += 1
            # Fall through to the last queue
            return self._lrus[self._ordering[-1]].popleft()

        @property
        def ordering(self) -> list[str]:
            """Return the priority ordering."""
            return self._ordering

        def count_by_type(self, cache_type: str) -> int:
            """Return the number of entries of the given type."""
            return len(self._lrus[cache_type])

        def pop_from_type(self, cache_type: str) -> tuple[int, ...]:
            """Pop the oldest entry for the given cache type."""
            return self._lrus[cache_type].popleft()

    def __init__(
        self,
        max_size: int = 10,
        max_bytes: int = 1 << 63,
        cache_dir: str | Path | None = None,
        fingerprint: str = "",
    ) -> None:
        self.max_size = max_size
        self.max_bytes = max_bytes
        # Identity of the model + KV-cache configuration that produces these
        # payloads. Persisted into each sidecar; on restart, only entries whose
        # fingerprint matches the running configuration are adopted, so a cache
        # built for one model/quantization is never loaded into another.
        self._fingerprint = fingerprint
        if cache_dir is None:
            self.cache_dir = Path(tempfile.mkdtemp(prefix="mlx-openai-prompt-cache-"))
            self._owns_cache_dir = True
        else:
            self.cache_dir = Path(cache_dir).expanduser()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._owns_cache_dir = False
        self._trie = PromptTrie()
        self._lru = self.CacheOrder()
        self._n_bytes = 0
        self._n_bytes_by_type: dict[str, int] = dict.fromkeys(self._lru.ordering, 0)

        # A caller-supplied directory may already hold payloads from a previous
        # run; rebuild the in-memory index from their sidecars. Owned temp dirs
        # are always fresh, so there is nothing to adopt.
        if not self._owns_cache_dir:
            self._rehydrate_from_disk()

    def __len__(self) -> int:
        return len(self._lru)

    @property
    def nbytes(self) -> int:
        return self._n_bytes

    def _cache_file_path(self) -> Path:
        """Return a unique path for a serialized cache entry."""
        return self.cache_dir / f"{uuid.uuid4().hex}.pkl"

    @staticmethod
    def _sidecar_path(payload_path: Path) -> Path:
        """Return the metadata-sidecar path for a payload ``{uuid}.pkl``."""
        return payload_path.with_name(f"{payload_path.stem}.meta.json")

    def _write_cache_to_disk(self, prompt_cache: list[Any]) -> Path:
        """Serialize a prompt cache to disk and return its final path.

        Parameters
        ----------
        prompt_cache : list[Any]
            MLX prompt-cache object list to serialize.

        Returns
        -------
        Path
            Path to the completed cache payload.

        Raises
        ------
        OSError
            If the cache directory or file cannot be written.
        pickle.PickleError
            If the prompt cache cannot be serialized.
        """
        file_path = self._cache_file_path()
        temp_path = file_path.with_suffix(".tmp")
        with temp_path.open("wb") as f:
            dill.dump(prompt_cache, f)
        temp_path.replace(file_path)
        return file_path

    def _write_sidecar(self, entry: CacheEntry, tokens_ids: list[int]) -> None:
        """Write the JSON metadata sidecar for a payload.

        The sidecar carries the trie key (``tokens``) and entry metadata so the
        in-memory index can be rebuilt on restart without deserializing the
        payload. It is written last (after the payload), via temp-then-replace,
        so a crash can only ever leave a payload without a sidecar — a harmless
        orphan that rehydrate discards — never the reverse.

        Parameters
        ----------
        entry : CacheEntry
            The freshly written cache entry whose payload is on disk.
        tokens_ids : list[int]
            Token sequence that keys this entry in the trie.
        """
        meta = {
            "version": CACHE_FORMAT_VERSION,
            "fingerprint": self._fingerprint,
            "tokens": list(tokens_ids),
            "nbytes": entry.nbytes,
            "cache_type": entry.cache_type,
            "trimmable": entry.trimmable,
            "source": entry.source,
        }
        sidecar_path = self._sidecar_path(entry.file_path)
        temp_path = sidecar_path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f)
        temp_path.replace(sidecar_path)

    def _load_cache_from_disk(self, entry: CacheEntry) -> list[Any]:
        """Deserialize a prompt cache entry from disk."""
        with entry.file_path.open("rb") as f:
            cache = dill.load(f)
        if not isinstance(cache, list):
            msg = f"Prompt cache payload at {entry.file_path} is not a list"
            raise TypeError(msg)
        return cache

    def _delete_entry_file(self, entry: CacheEntry) -> None:
        """Delete a serialized cache file and its sidecar, logging failures."""
        for path in (entry.file_path, self._sidecar_path(entry.file_path)):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(f"Failed to delete prompt cache file {path}: {exc!s}")

    def _remove_entry_accounting(self, entry: CacheEntry) -> None:
        """Subtract an entry from in-memory byte accounting."""
        self._n_bytes -= entry.nbytes
        self._n_bytes_by_type[entry.cache_type] -= entry.nbytes

    def _evict_tokens(self, tokens: tuple[int, ...]) -> None:
        """Evict one token sequence and its on-disk payload."""
        evicted_entry = self._trie.pop(list(tokens))
        self._remove_entry_accounting(evicted_entry)
        self._delete_entry_file(evicted_entry)

    def _invalidate_tokens(self, tokens: list[int], entry: CacheEntry) -> None:
        """Remove a corrupt or missing cache entry from all indexes."""
        try:
            self._trie.pop(tokens)
        except (KeyError, IndexError):
            pass
        self._lru.remove(tuple(tokens))
        self._remove_entry_accounting(entry)
        self._delete_entry_file(entry)

    def _try_load_cache(self, tokens: list[int], entry: CacheEntry) -> list[Any] | None:
        """Load an entry and invalidate it if the on-disk payload is unusable."""
        try:
            return self._load_cache_from_disk(entry)
        except (
            OSError,
            EOFError,
            TypeError,
            ValueError,
            AttributeError,
            pickle.PickleError,
        ) as exc:
            logger.warning(f"Failed to load prompt cache from {entry.file_path}: {exc!s}")
            self._invalidate_tokens(tokens, entry)
            self._release_evicted_memory()
            return None

    def _release_evicted_memory(self) -> None:
        """Ask Python and MLX to release memory after cache entries are evicted."""
        gc.collect()
        try:
            import mlx.core as mx  # noqa: PLC0415

            mx.clear_cache()
        except (ImportError, RuntimeError) as exc:
            logger.debug(f"Could not clear MLX cache after prompt-cache eviction: {exc!s}")

    def _discard_orphan(self, payload_path: Path, sidecar_path: Path) -> None:
        """Best-effort delete of an unusable payload/sidecar pair."""
        for path in (payload_path, sidecar_path):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug(f"Could not remove orphaned prompt cache file {path}: {exc!s}")

    def _rehydrate_from_disk(self) -> None:
        """Rebuild the trie, LRU, and byte accounting from on-disk sidecars.

        Scans ``cache_dir`` for ``*.meta.json`` sidecars and adopts every entry
        whose schema ``version`` and ``fingerprint`` match the running
        configuration and whose payload still exists. Only metadata is loaded
        into memory; payloads stay on disk and are deserialized lazily on the
        first matching ``fetch_nearest_cache`` hit, exactly as in steady state.

        True LRU recency does not survive a restart, so sidecars are adopted in
        mtime order (oldest first) as a best-effort proxy. Mismatched, corrupt,
        or half-written entries are discarded. After loading, ``max_size`` and
        ``max_bytes`` are re-applied.
        """
        valid_types = set(self._lru.ordering)
        adopted = skipped = 0
        for sidecar_path in sorted(
            self.cache_dir.glob("*.meta.json"),
            key=lambda p: p.stat().st_mtime,
        ):
            payload_path = sidecar_path.with_name(f"{sidecar_path.stem.removesuffix('.meta')}.pkl")
            try:
                meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug(f"Discarding unreadable prompt cache sidecar {sidecar_path}: {exc!s}")
                self._discard_orphan(payload_path, sidecar_path)
                skipped += 1
                continue

            if (
                meta.get("version") != CACHE_FORMAT_VERSION
                or meta.get("fingerprint") != self._fingerprint
                or meta.get("cache_type") not in valid_types
                or not payload_path.exists()
            ):
                self._discard_orphan(payload_path, sidecar_path)
                skipped += 1
                continue

            tokens = meta["tokens"]
            cache_type = meta["cache_type"]
            entry = self.CacheEntry(
                payload_path,
                int(meta["nbytes"]),
                cache_type,
                bool(meta["trimmable"]),
                meta.get("source", "nonbatch"),
            )
            self._trie.add(tokens, entry)
            self._lru.push(tuple(tokens), cache_type)
            self._n_bytes += entry.nbytes
            self._n_bytes_by_type[cache_type] += entry.nbytes
            adopted += 1

        if adopted or skipped:
            # Honour configured limits now that the whole working set is indexed.
            self.trim_to(n_sequences=self.max_size, n_bytes=self.max_bytes)
            logger.info(
                f"Rehydrated prompt cache from {self.cache_dir}: "
                f"{adopted} adopted, {skipped} skipped, {self.nbytes / 1e9:.2f} GB indexed"
            )

    def fetch_nearest_cache(
        self,
        tokens_ids: list[int],
        *,
        allowed_sources: set[str] | None = None,
    ) -> tuple[list[Any] | None, list[int]]:
        """Fetch the nearest matching cache for the given token sequence.

        Parameters
        ----------
        tokens_ids : list[int]
            Token sequence to look up.
        allowed_sources : set[str] | None, optional
            Restrict hits to caches produced by these generation paths. This
            prevents MLX cache objects created on one worker thread from being
            reused on another worker thread.

        Returns
        -------
        tuple[list[Any] | None, list[int]]
            Tuple of (prompt_cache, remaining_tokens). If no cache found,
            returns (None, original_tokens).
        """
        result = self._trie.search(tokens_ids)
        if result.exact is not None:
            cache_entry = self._trie.get(result.exact)
            if self._source_allowed(cache_entry, allowed_sources):
                cache = self._try_load_cache(result.exact, cache_entry)
                if cache is not None:
                    return cache, []
            shorter = self._find_allowed_shorter(tokens_ids, allowed_sources)
            if shorter is not None:
                cache_entry = self._trie.get(shorter)
                cache = self._try_load_cache(shorter, cache_entry)
                if cache is not None:
                    return cache, tokens_ids[len(shorter) :]
            return None, tokens_ids

        short_length = len(result.shorter) if result.shorter is not None else 0
        if result.longer is not None and result.common_prefix > short_length:
            cache_entry = self._trie.get(result.longer)
            if cache_entry.trimmable and self._source_allowed(cache_entry, allowed_sources):
                cache = self._try_load_cache(result.longer, cache_entry)
                if cache is not None:
                    prefix = min(len(tokens_ids) - 1, result.common_prefix)
                    num_to_trim = len(result.longer) - prefix
                    trim_prompt_cache(cache, num_to_trim)
                    return cache, tokens_ids[prefix:]

        if short_length > 0:
            shorter_tokens = result.shorter
            cache_entry = self._trie.get(result.shorter)
            if not self._source_allowed(cache_entry, allowed_sources):
                shorter_tokens = self._find_allowed_shorter(tokens_ids, allowed_sources)
                if shorter_tokens is None:
                    return None, tokens_ids
                cache_entry = self._trie.get(shorter_tokens)
                short_length = len(shorter_tokens)
            cache = self._try_load_cache(shorter_tokens, cache_entry)
            if cache is not None:
                return cache, tokens_ids[short_length:]

        return None, tokens_ids

    def _source_allowed(
        self,
        entry: CacheEntry,
        allowed_sources: set[str] | None,
    ) -> bool:
        """Return whether an entry may be used by the current generation path."""
        return allowed_sources is None or entry.source in allowed_sources

    def _find_allowed_shorter(
        self,
        tokens_ids: list[int],
        allowed_sources: set[str] | None,
    ) -> list[int] | None:
        """Find the longest allowed strict-prefix cache entry."""
        return self._trie.find_shorter(
            tokens_ids,
            lambda entry: self._source_allowed(entry, allowed_sources),
        )

    def insert_cache(
        self,
        tokens_ids: list[int],
        prompt_cache: list[Any],
        *,
        cache_type: str = "assistant",
        source: str = "nonbatch",
    ) -> None:
        """Insert or update a cache entry.

        Parameters
        ----------
        tokens_ids : list[int]
            Token sequence identifying this cache entry.
        prompt_cache : list[Any]
            The prompt cache data to store.
        cache_type : str, optional
            Priority category for eviction ordering, by default ``"assistant"``.
        source : str, optional
            Generation path that produced the cache, by default ``"nonbatch"``.
        """
        tokens_tuple = tuple(tokens_ids)

        # Make the cache entry
        entry = self.CacheEntry(
            self._write_cache_to_disk(prompt_cache),
            sum(getattr(c, "nbytes", 0) for c in prompt_cache),
            cache_type,
            can_trim_prompt_cache(prompt_cache),
            source,
        )
        # Persist the metadata sidecar so this entry can be rebuilt after a
        # restart without loading the payload (see _rehydrate_from_disk).
        self._write_sidecar(entry, tokens_ids)

        # Insert into the trie and update the byte counter and lru position
        self._n_bytes += entry.nbytes
        self._n_bytes_by_type[cache_type] += entry.nbytes
        prev = self._trie.add(tokens_ids, entry)
        if prev is not None:
            self._remove_entry_accounting(prev)
            self._delete_entry_file(prev)
            self._lru.remove(tokens_tuple)
        self._lru.push(tokens_tuple, cache_type)

        # If it is a trimmable cache remove all prefixes cause they just take
        # space
        if entry.trimmable:
            for prefix_len, removed_entry in self._trie.pop_prefixes(tokens_ids):
                self._remove_entry_accounting(removed_entry)
                self._delete_entry_file(removed_entry)
                self._lru.remove(tuple(tokens_ids[:prefix_len]))

        # Ensure we match the constraints
        if len(self._lru) > self.max_size:
            evicted = self._lru.pop()
            self._evict_tokens(evicted)

        while self._n_bytes > self.max_bytes:
            evicted = self._lru.pop()
            self._evict_tokens(evicted)

        self._release_evicted_memory()

    def trim_to(self, *, n_sequences: int | None = None, n_bytes: int | None = None) -> None:
        """Trim the cache down to sequence and/or byte limits."""
        max_sequences = max(0, n_sequences) if n_sequences is not None else 1 << 63
        max_bytes = max(0, n_bytes) if n_bytes is not None else 1 << 63

        while len(self._lru) > max_sequences:
            evicted = self._lru.pop()
            self._evict_tokens(evicted)

        while self._n_bytes > max_bytes:
            evicted = self._lru.pop()
            self._evict_tokens(evicted)

        self._release_evicted_memory()

    def clear(self) -> None:
        """Remove all tracked cache entries and serialized payloads."""
        for cache_type in self._lru.ordering:
            while self._lru.count_by_type(cache_type):
                self._evict_tokens(self._lru.pop_from_type(cache_type))
        self._release_evicted_memory()

    def _reset_in_memory_state(self) -> None:
        """Drop the in-memory index without touching on-disk payloads."""
        self._trie = PromptTrie()
        self._lru = self.CacheOrder()
        self._n_bytes = 0
        self._n_bytes_by_type = dict.fromkeys(self._lru.ordering, 0)

    def close(self) -> None:
        """Release the cache, preserving a caller-supplied directory's payloads.

        For a process-local temporary directory (created when ``cache_dir`` was
        omitted), the serialized payloads and the directory itself are removed.
        For a caller-supplied directory the on-disk payloads and sidecars are
        left in place so they can be rehydrated on the next run; only the
        in-memory index is dropped. This is what makes a configured
        ``prompt_cache_dir`` survive a restart rather than being wiped on
        shutdown.
        """
        if self._owns_cache_dir:
            self.clear()
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        else:
            self._reset_in_memory_state()
        self._release_evicted_memory()

    def stats_by_type(self) -> dict[str, dict[str, int]]:
        """Return per-type sequence count and byte usage."""
        result = {}
        for cache_type in self._lru.ordering:
            result[cache_type] = {
                "n_sequences": self._lru.count_by_type(cache_type),
                "n_bytes": self._n_bytes_by_type[cache_type],
            }
        return result

    def log_cache_stats(self) -> None:
        """Log the current cache size, bytes, and per-type stats."""
        logger.info(
            "KV Caches: {} seq, {:.2f} GB",
            len(self),
            self.nbytes / 1e9,
        )


if __name__ == "__main__":
    from app.models.mlx_lm import MLX_LM

    model_path = "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"
    model = MLX_LM(model_path)
    prompt_cache = LRUPromptCache()

    import time

    start_time = time.time()
    first_token = True

    prompt_1 = "Hello, how are you? I'm fine, thank you."
    input_prompt = model.create_input_prompt([{"role": "user", "content": prompt_1}], {})
    input_ids = model.encode_prompt(input_prompt)

    cache, rest_input_ids = prompt_cache.fetch_nearest_cache(input_ids)
    if cache is None:
        cache = model.create_prompt_cache()
    # Use full input_ids for cache_key, not rest_input_ids
    cache_key = input_ids[:]

    response_1 = model(rest_input_ids, cache, stream=True)
    for chunk in response_1:
        if chunk:
            if first_token:
                print("TIME TO FIRST TOKEN", time.time() - start_time)
                first_token = False
            cache_key.append(chunk.token)

    prompt_cache.insert_cache(cache_key, cache)

    start_time = time.time()
    first_token = True
    prompt_2 = "Hello, how are you? I'm fine, thank you."
    input_prompt_2 = model.create_input_prompt([{"role": "user", "content": prompt_2}], {})
    input_ids_2 = model.encode_prompt(input_prompt_2)
    cache, rest_input_ids_2 = prompt_cache.fetch_nearest_cache(input_ids_2)

    if cache is None:
        cache = model.create_prompt_cache()
    # Use full input_ids for cache_key, not rest_input_ids
    cache_key_2 = input_ids_2[:]

    start_time = time.time()
    response_2 = model(rest_input_ids_2, cache, stream=True)
    raw_text = ""
    for chunk in response_2:
        if chunk:
            if first_token:
                print("TIME TO FIRST TOKEN", time.time() - start_time)
                first_token = False
            raw_text += chunk.text
            cache_key_2.append(chunk.token)

    print("RAW TEXT", raw_text)

    prompt_cache.insert_cache(cache_key_2, cache)
