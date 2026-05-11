"""
embedding_store.py — Read-only embedding cache (FOUND-01).

D-12: Embeddings live in EmbeddingStore, NOT in ClusteringState.
D-11: item_id is a sequential integer; EmbeddingStore[item_id] = embedding vector.

The store is computed once at session setup and never mutated.
All clustering operations read from this store by item_id.
"""
from __future__ import annotations

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer


# Model locked for this project — do not change without re-computing embeddings
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768       # output dimension of all-mpnet-base-v2
BATCH_SIZE = 32           # safe for CPU inference on a laptop; increase if GPU available


class EmbeddingStore:
    """
    Read-only store for pre-computed sentence embeddings.

    Internal array shape: (N, EMBEDDING_DIM) float32.
    Item IDs are row indices (0-based sequential integers per D-11).

    Usage:
        # One-time setup:
        store = EmbeddingStore.compute_and_save(texts, "embeddings/embeddings.npy")
        # Every subsequent session:
        store = EmbeddingStore.load("embeddings/embeddings.npy")
        vec = store.get(item_id)    # shape (768,)
        all_vecs = store.get_all()  # shape (N, 768)
    """

    def __init__(self, embeddings: np.ndarray) -> None:
        assert embeddings.ndim == 2, (
            f"Embeddings must be 2D (n_items, dim), got shape {embeddings.shape}"
        )
        assert embeddings.shape[1] == EMBEDDING_DIM, (
            f"Expected embedding dim {EMBEDDING_DIM}, got {embeddings.shape[1]}"
        )
        # Store as read-only view to enforce the no-mutation contract
        self._embeddings = embeddings
        self._embeddings.flags.writeable = False

    @classmethod
    def compute_and_save(
        cls,
        texts: list[str],
        save_path: str,
    ) -> "EmbeddingStore":
        """
        Encode all texts with all-mpnet-base-v2 and save to save_path as .npy.

        This is a one-time operation — expect 10-30 minutes on CPU for 12K texts.
        After completion, use EmbeddingStore.load() for all subsequent sessions.
        """
        assert len(texts) > 0, "texts must be non-empty"
        assert save_path.endswith(".npy"), f"save_path must end in .npy, got {save_path}"

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        print(f"Loading model {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"Encoding {len(texts)} texts (batch_size={BATCH_SIZE})...")
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,  # returns float32 ndarray
        )

        assert embeddings.shape == (len(texts), EMBEDDING_DIM), (
            f"Expected shape ({len(texts)}, {EMBEDDING_DIM}), got {embeddings.shape}"
        )

        np.save(save_path, embeddings)
        print(f"Saved embeddings to {save_path} (shape: {embeddings.shape})")

        return cls(embeddings)

    @classmethod
    def load(cls, save_path: str) -> "EmbeddingStore":
        """Load pre-computed embeddings from a .npy file."""
        assert os.path.exists(save_path), f"Embeddings file not found: {save_path}"
        embeddings = np.load(save_path)
        return cls(embeddings)

    def get(self, item_id: int) -> np.ndarray:
        """Return the embedding vector for a single item. Shape: (EMBEDDING_DIM,)."""
        assert 0 <= item_id < len(self._embeddings), (
            f"item_id {item_id} out of range [0, {len(self._embeddings)})"
        )
        return self._embeddings[item_id]

    def get_all(self) -> np.ndarray:
        """Return all embeddings. Shape: (N, EMBEDDING_DIM). Array is read-only."""
        return self._embeddings

    def __len__(self) -> int:
        return len(self._embeddings)


def load_texts_from_jsonl(jsonl_path: str) -> list[str]:
    """
    Load text strings from a JSONL file produced by data_loader.py.
    Expects records with 'text' field (the canonical internal field name).
    """
    assert os.path.exists(jsonl_path), f"JSONL file not found: {jsonl_path}"
    texts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                assert "text" in record, f"Record missing 'text' field: {record}"
                texts.append(record["text"])
    assert len(texts) > 0, f"No texts loaded from {jsonl_path}"
    return texts


if __name__ == "__main__":
    """One-time embedding computation. Run once after data_loader.py completes."""
    train_path = "dataset/train.jsonl"
    embed_path = "embeddings/embeddings.npy"

    assert not os.path.exists(embed_path), (
        f"Embeddings already exist at {embed_path}. "
        "Delete the file manually if you need to recompute."
    )

    print(f"Loading training texts from {train_path}...")
    texts = load_texts_from_jsonl(train_path)
    print(f"Loaded {len(texts)} texts")

    store = EmbeddingStore.compute_and_save(texts, embed_path)
    print(f"Done. Store has {len(store)} items of dim {store.get(0).shape[0]}")
