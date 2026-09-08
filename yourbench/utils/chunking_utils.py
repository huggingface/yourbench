import random
from typing import Any, Callable, Optional
from dataclasses import dataclass

import tiktoken


CHUNK_MODE_PERCENT = "percentage"
CHUNK_MODE_COUNT = "count"
CHUNK_MODE_ALL = "all"


@dataclass
class ChunkSamplingConfig:
    mode: str = CHUNK_MODE_ALL
    value: float = 1.0
    random_seed: int = 42
    strategy: str = "random"


def split_into_token_chunks(
    text: str,
    chunk_tokens: int = 1024,
    overlap: int = 100,
    encoding_name: str = "cl100k_base",
    preprocess: Optional[Callable[[str], str]] = None,
) -> list[str]:
    """
    Splits text into token-based chunks, with optional preprocessing.

    Args:
        text (str): The input text.
        chunk_tokens (int): Max tokens per chunk.
        overlap (int): Number of overlapping tokens.
        encoding_name (str): tiktoken encoding name.
        preprocess (Optional[Callable[[str], str]]): Optional preprocessing function.

    Returns:
        list[str]: List of decoded text chunks.
    """
    if chunk_tokens <= 0 or not 0 <= overlap < chunk_tokens:
        raise ValueError("Require chunk_tokens > 0 and 0 <= overlap < chunk_tokens")
    if preprocess:
        text = preprocess(text)

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text, disallowed_special=())
    stride = chunk_tokens - overlap
    return [enc.decode(tokens[i : i + chunk_tokens]) for i in range(0, len(tokens), stride)]


def get_sampling_cfg(cfg: Any) -> ChunkSamplingConfig:
    """Extract and return the chunk sampling config as a ChunkSamplingConfig dataclass"""
    cs = cfg.chunk_sampling
    # Map schema fields to local ChunkSamplingConfig
    return ChunkSamplingConfig(
        mode=CHUNK_MODE_COUNT if cs.enable else CHUNK_MODE_ALL,
        value=cs.num_samples if cs.enable else 1.0,
        strategy=getattr(cs, "strategy", "random"),
        random_seed=cs.random_seed if hasattr(cs, "random_seed") else 42,
    )


def safe_sample(lst: list[Any], k: int, rng: random.Random | None = None) -> list[Any]:
    """Sample k elements from lst, or return lst if k >= len(lst)"""
    return (rng or random.Random()).sample(lst, k) if k < len(lst) else lst


def sample_single_hop_chunks(
    chunks_list: list[dict[str, Any]], chunk_sampling: ChunkSamplingConfig
) -> list[dict[str, Any]]:
    if not chunks_list:
        return []

    rng = random.Random(chunk_sampling.random_seed)
    mode = chunk_sampling.mode.lower()
    value = chunk_sampling.value
    total = len(chunks_list)

    if mode == CHUNK_MODE_PERCENT:
        k = int(total * value)
        return chunks_list[:k] if chunk_sampling.strategy == "first" else safe_sample(chunks_list, k, rng)
    elif mode == CHUNK_MODE_COUNT:
        k = min(int(value), total)
        return chunks_list[:k] if chunk_sampling.strategy == "first" else safe_sample(chunks_list, k, rng)
    else:
        return chunks_list


def sample_multihop_groups(
    mh_chunks: list[dict[str, Any]], chunk_sampling_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    if not chunk_sampling_cfg:
        return mh_chunks
    mode = chunk_sampling_cfg.get("mode", CHUNK_MODE_ALL).lower()
    value = chunk_sampling_cfg.get("value", 1.0)
    rng = random.Random(chunk_sampling_cfg.get("random_seed", 42))
    total = len(mh_chunks)
    if total < 2:
        return mh_chunks
    if mode == CHUNK_MODE_PERCENT:
        k = int(total * value)
        return safe_sample(mh_chunks, k, rng)
    elif mode == CHUNK_MODE_COUNT:
        k = min(int(value), total)
        return safe_sample(mh_chunks, k, rng)
    return mh_chunks
