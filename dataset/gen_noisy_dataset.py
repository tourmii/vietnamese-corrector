import argparse
import csv
import logging
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterator, Sequence

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(ROOT_DIR, "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from gen_error import ALL_ERROR_TYPES, generate_error  # noqa: E402

# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------

_TEXTS: list[str] | None = None
_ERROR_TYPES: list[str] | None = None


def _init_worker(texts: list[str], error_types: list[str]) -> None:
    global _TEXTS, _ERROR_TYPES
    _TEXTS = texts
    _ERROR_TYPES = error_types


# ---------------------------------------------------------------------------
# Block: one worker owns a disjoint slice of sampled texts and applies
# ALL error types to each → no cross-worker duplicates possible.
# ---------------------------------------------------------------------------

def _generate_block(args: tuple[int, list[str]]) -> list[tuple[str, str, str]]:
    """Apply every error type to every text in this block.

    args:
        block_seed  – unique seed so random params differ across blocks.
        block_texts – disjoint slice of the sampled texts.
    """
    if _ERROR_TYPES is None:
        raise RuntimeError("Worker not initialized.")

    block_seed, block_texts = args
    rng = random.Random(block_seed)

    rows: list[tuple[str, str, str]] = []
    for text in block_texts:
        for error_type in _ERROR_TYPES:
            # Give each (text, error_type) pair a unique seed so random
            # kwargs (intensity, error_rate, …) are reproducible but varied.
            pair_seed = rng.getrandbits(64)
            noisy, used_type = generate_error(text, error_type=error_type, seed=pair_seed)
            rows.append((noisy, text, used_type))
    return rows


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_texts(path: str) -> list[str]:
    texts: list[str] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if row:
                text = row[0].strip()
                if text:
                    texts.append(text)
    if not texts:
        raise ValueError(f"No texts found in {path!r}.")
    return texts


def sample_texts(texts: list[str], n: int, seed: int) -> list[str]:
    """Return `n` unique texts sampled without replacement."""
    if n >= len(texts):
        logging.warning(
            "Requested %d texts but corpus only has %d — using all.", n, len(texts)
        )
        return texts[:]
    rng = random.Random(seed)
    return rng.sample(texts, n)


# ---------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------

def _make_blocks(
    sampled_texts: list[str],
    error_types: list[str],
    seed: int,
    num_workers: int,
) -> list[tuple[int, list[str]]]:
    """Split sampled_texts into num_workers disjoint slices."""
    n = len(sampled_texts)
    w = min(num_workers, n)
    chunk = max(1, (n + w - 1) // w)
    rng = random.Random(seed)
    blocks = []
    for i in range(w):
        slice_ = sampled_texts[i * chunk : (i + 1) * chunk]
        if not slice_:
            break
        block_seed = rng.getrandbits(64)
        blocks.append((block_seed, slice_))
    return blocks


def iter_rows_parallel(
    sampled_texts: list[str],
    error_types: list[str],
    seed: int,
    num_workers: int,
) -> Iterator[tuple[str, str, str]]:
    blocks = _make_blocks(sampled_texts, error_types, seed, num_workers)
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(sampled_texts, error_types),
    ) as pool:
        futs = [pool.submit(_generate_block, b) for b in blocks]
        for fut in as_completed(futs):
            yield from fut.result()


def iter_rows_single(
    sampled_texts: list[str],
    error_types: list[str],
    seed: int,
) -> Iterator[tuple[str, str, str]]:
    rng = random.Random(seed)
    for text in sampled_texts:
        for error_type in error_types:
            pair_seed = rng.getrandbits(64)
            noisy, used_type = generate_error(text, error_type=error_type, seed=pair_seed)
            yield noisy, text, used_type


# ---------------------------------------------------------------------------
# Split writer
# ---------------------------------------------------------------------------

def generate_split(
    input_path: str,
    output_path: str,
    total_rows: int,
    seed: int,
    chunk_size: int,
    log_every: int,
    num_workers: int,
) -> int:
    """Sample texts, apply all error types, write CSV. Returns actual row count."""
    texts = read_texts(input_path)
    error_types = list(ALL_ERROR_TYPES)
    n_types = len(error_types)

    # Round down to nearest multiple of n_types for perfect balance
    actual_total = (total_rows // n_types) * n_types
    n_texts = actual_total // n_types

    sampled = sample_texts(texts, n_texts, seed)
    actual_total = len(sampled) * n_types  # may differ if corpus is smaller

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    logging.info(
        "Split: input=%s | corpus=%d | sampled=%d texts × %d types = %d rows | workers=%d",
        input_path, len(texts), len(sampled), n_types, actual_total, num_workers,
    )

    row_iter = (
        iter_rows_parallel(sampled, error_types, seed, num_workers)
        if num_workers > 1
        else iter_rows_single(sampled, error_types, seed)
    )

    written = 0
    buffer: list[tuple[str, str, str]] = []

    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["noisy", "gt", "type"])

        for idx, row in enumerate(row_iter, 1):
            buffer.append(row)
            if len(buffer) >= chunk_size:
                writer.writerows(buffer)
                fh.flush()
                written += len(buffer)
                buffer.clear()
                logging.info("  flushed %d / %d rows → %s", written, actual_total, output_path)
            elif log_every > 0 and idx % log_every == 0:
                logging.info("  generated %d / %d rows for %s", idx, actual_total, output_path)

        if buffer:
            writer.writerows(buffer)
            fh.flush()
            written += len(buffer)
            buffer.clear()

    logging.info("Done: %d rows → %s", written, output_path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate balanced noisy Vietnamese text datasets.\n"
            "Each sampled text receives one noisy version per error type,\n"
            "so all error types have exactly equal representation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train-input",  default=os.path.join("dataset", "train_dataset.csv"))
    parser.add_argument("--test-input",   default=os.path.join("dataset", "test_dataset.csv"))
    parser.add_argument("--train-output", default=os.path.join("dataset", "train_noisy.csv"))
    parser.add_argument("--test-output",  default=os.path.join("dataset", "test_noisy.csv"))
    parser.add_argument(
        "--train-rows", type=int, default=10_000_000,
        help="Target total rows for the train split (rounded to multiple of error types).",
    )
    parser.add_argument(
        "--test-rows", type=int, default=500_000,
        help="Target total rows for the test split.",
    )
    parser.add_argument("--seed",       type=int, default=1337)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--log-every",  type=int, default=500_000)
    parser.add_argument(
        "--num-workers", type=int, default=max(1, mp.cpu_count() - 1),
        help="Worker processes (default: cpu_count - 1).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    train_rows = generate_split(
        args.train_input, args.train_output,
        total_rows=args.train_rows,
        seed=args.seed,
        chunk_size=args.chunk_size,
        log_every=args.log_every,
        num_workers=args.num_workers,
    )
    test_rows = generate_split(
        args.test_input, args.test_output,
        total_rows=args.test_rows,
        seed=args.seed ^ 0xDEADBEEF,
        chunk_size=args.chunk_size,
        log_every=args.log_every,
        num_workers=args.num_workers,
    )

    n_types = len(ALL_ERROR_TYPES)
    print(
        f"\nSummary\n"
        f"  error types : {n_types}  {ALL_ERROR_TYPES}\n"
        f"  train       : {train_rows:>10,} rows  ({train_rows // n_types:,} texts × {n_types} types)\n"
        f"  test        : {test_rows:>10,} rows  ({test_rows  // n_types:,} texts × {n_types} types)\n"
    )


if __name__ == "__main__":
    main()