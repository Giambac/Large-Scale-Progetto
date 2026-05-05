"""
data_loader.py — Dataset download, split, and hash verification.

PRE-01: Frozen held-out split with SHA-256 hash lock.
D-01: Amazon Reviews 2023 Arts, Crafts & Sewing, 15K records.
D-03: 80/20 random split, seeded.
D-04: verify_held_out_hash crashes on mismatch.
D-11: item_ids are sequential integers 0 to N-1.
"""
import hashlib
import json
import os
import random
from typing import Optional


# Constants — do not change after held_out.jsonl is created
DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"
DATASET_CONFIG = "raw_review_Arts_Crafts_and_Sewing"
# D-02 specifies 'review_text' as the field name; RESEARCH.md confirms the actual HuggingFace field is 'text' — HF_TEXT_FIELD = "text" is correct per RESEARCH.md correction.
HF_TEXT_FIELD = "text"          # HuggingFace field name (not 'review_text')
TARGET_SAMPLE_SIZE = 15_000
TRAIN_RATIO = 0.80
DEFAULT_SEED = 42


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hex digest of a file using chunked reads."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_held_out_hash(held_out_path: str, hash_path: str) -> None:
    """
    Assert that the SHA-256 of held_out_path matches the stored hash in hash_path.
    Crashes immediately with AssertionError if the hash does not match.
    Called at session start — any mismatch means the held-out set was contaminated.
    """
    with open(hash_path) as f:
        expected = f.read().strip()
    actual = compute_sha256(held_out_path)
    assert actual == expected, (
        f"Held-out split hash mismatch!\n"
        f"  Expected: {expected}\n"
        f"  Actual:   {actual}\n"
        f"  File may have been modified: {held_out_path}"
    )


def split_dataset(
    records: list[dict],
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], list[dict]]:
    """
    Randomly split records into (train, held_out) at TRAIN_RATIO.
    Returns exact integer counts: floor(N * TRAIN_RATIO) train, remainder held-out.
    Split is reproducible given the same seed.
    """
    assert len(records) > 0, "Cannot split an empty list"
    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(len(records) * TRAIN_RATIO)
    train_indices = set(indices[:n_train])
    train = [records[i] for i in range(len(records)) if i in train_indices]
    held_out = [records[i] for i in range(len(records)) if i not in train_indices]
    assert len(train) + len(held_out) == len(records)
    return train, held_out


def download_and_save_dataset(
    output_dir: str = "dataset",
    raw_path: Optional[str] = None,
    train_path: Optional[str] = None,
    held_out_path: Optional[str] = None,
    hash_path: Optional[str] = None,
    seed: int = DEFAULT_SEED,
) -> None:
    """
    One-time setup: download 15K reviews, split 80/20, write SHA-256 hash.

    IMPORTANT: If held_out_path and hash_path already exist, this function
    asserts and crashes rather than overwriting them. Overwriting the held-out
    split after experiments have run would contaminate evaluation.

    Files written:
      {output_dir}/arts_crafts_15k.jsonl  — raw 15K records
      {output_dir}/train.jsonl            — 80% split (12000 records)
      {output_dir}/held_out.jsonl         — 20% split (3000 records) — NEVER REGENERATE
      {output_dir}/held_out.sha256        — SHA-256 of held_out.jsonl — NEVER REGENERATE
    """
    from datasets import load_dataset  # only imported here — not needed at import time

    os.makedirs(output_dir, exist_ok=True)

    raw_path = raw_path or os.path.join(output_dir, "arts_crafts_15k.jsonl")
    train_path = train_path or os.path.join(output_dir, "train.jsonl")
    held_out_path = held_out_path or os.path.join(output_dir, "held_out.jsonl")
    hash_path = hash_path or os.path.join(output_dir, "held_out.sha256")

    # Refuse to overwrite the held-out split once it exists — D-04 invariant
    assert not os.path.exists(held_out_path), (
        f"Held-out file already exists: {held_out_path}\n"
        "Refusing to overwrite. Delete manually ONLY if no experiments have run."
    )
    assert not os.path.exists(hash_path), (
        f"Hash file already exists: {hash_path}\n"
        "Refusing to overwrite. Delete manually ONLY if no experiments have run."
    )

    # Stream dataset — never download all 9M rows
    print(f"Streaming {DATASET_ID} ({DATASET_CONFIG})...")
    ds = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    records = []
    for example in ds:
        text = example.get(HF_TEXT_FIELD, "").strip()
        if text:
            records.append({"item_id": len(records), "text": text})
        if len(records) >= TARGET_SAMPLE_SIZE:
            break

    assert len(records) == TARGET_SAMPLE_SIZE, (
        f"Expected {TARGET_SAMPLE_SIZE} records, got {len(records)}. "
        "Dataset may have fewer non-empty text records than expected."
    )

    # Write raw sample
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {raw_path}")

    # Split
    train, held_out = split_dataset(records, seed=seed)
    assert len(train) == 12_000, f"Expected 12000 train records, got {len(train)}"
    assert len(held_out) == 3_000, f"Expected 3000 held-out records, got {len(held_out)}"

    # Write train split
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train)} train records to {train_path}")

    # Write held-out split
    with open(held_out_path, "w", encoding="utf-8") as f:
        for r in held_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(held_out)} held-out records to {held_out_path}")

    # Write SHA-256 hash — this seals the held-out set
    digest = compute_sha256(held_out_path)
    with open(hash_path, "w") as f:
        f.write(digest)
    print(f"Wrote SHA-256 hash to {hash_path}: {digest[:16]}...")

    # Verify immediately after writing
    verify_held_out_hash(held_out_path, hash_path)
    print("Hash verification passed. Held-out split is sealed.")


if __name__ == "__main__":
    download_and_save_dataset()
