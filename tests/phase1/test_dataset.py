"""Tests for data_loader.py — PRE-01: dataset split and hash lock."""
import hashlib
import os
import tempfile
import pytest

# Will fail with ImportError until src/data_loader.py exists
from src.data_loader import compute_sha256, verify_held_out_hash, split_dataset


def test_compute_sha256_deterministic(tmp_path):
    """Same file content always produces the same SHA-256 hex string."""
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello world")
    h1 = compute_sha256(str(f))
    h2 = compute_sha256(str(f))
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex is always 64 chars


def test_hash_mismatch_crashes(tmp_path):
    """verify_held_out_hash raises AssertionError when hash file does not match file content."""
    data_file = tmp_path / "held_out.jsonl"
    hash_file = tmp_path / "held_out.sha256"
    data_file.write_bytes(b"original content")
    # Write a deliberately wrong hash
    hash_file.write_text("0" * 64)
    with pytest.raises(AssertionError, match="hash mismatch"):
        verify_held_out_hash(str(data_file), str(hash_file))


def test_verify_held_out_hash_passes_when_correct(tmp_path):
    """verify_held_out_hash passes silently when hash matches."""
    data_file = tmp_path / "held_out.jsonl"
    hash_file = tmp_path / "held_out.sha256"
    content = b"some review content"
    data_file.write_bytes(content)
    h = hashlib.sha256(content).hexdigest()
    hash_file.write_text(h)
    # Must not raise
    verify_held_out_hash(str(data_file), str(hash_file))


def test_split_ratio(tmp_path):
    """split_dataset produces an 80/20 split from a list of records."""
    records = [{"item_id": i, "text": f"review {i}"} for i in range(100)]
    train, held_out = split_dataset(records, seed=42)
    assert len(train) == 80, f"Expected 80 train items, got {len(train)}"
    assert len(held_out) == 20, f"Expected 20 held-out items, got {len(held_out)}"


def test_split_reproducible():
    """Same seed produces the same split every time."""
    records = [{"item_id": i, "text": f"review {i}"} for i in range(50)]
    train1, held1 = split_dataset(records, seed=99)
    train2, held2 = split_dataset(records, seed=99)
    assert [r["item_id"] for r in train1] == [r["item_id"] for r in train2]
    assert [r["item_id"] for r in held1] == [r["item_id"] for r in held2]


def test_split_no_overlap():
    """Train and held-out sets share no item_ids."""
    records = [{"item_id": i, "text": f"text {i}"} for i in range(100)]
    train, held_out = split_dataset(records, seed=0)
    train_ids = {r["item_id"] for r in train}
    held_ids = {r["item_id"] for r in held_out}
    assert train_ids.isdisjoint(held_ids)
