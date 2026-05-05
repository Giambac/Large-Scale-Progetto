"""Tests for embedding_store.py — FOUND-01: EmbeddingStore."""
import numpy as np
import pytest

from src.embedding_store import EmbeddingStore


def test_embedding_store_shape(tmp_path, mock_embeddings):
    """EmbeddingStore saved from (5, 768) array loads with same shape."""
    save_path = str(tmp_path / "test_embeddings.npy")
    np.save(save_path, mock_embeddings)
    store = EmbeddingStore.load(save_path)
    assert store.get_all().shape == (5, 768)


def test_embedding_store_get_single_item(tmp_path, mock_embeddings):
    """get(item_id) returns a 1D vector of length 768."""
    save_path = str(tmp_path / "test_embeddings.npy")
    np.save(save_path, mock_embeddings)
    store = EmbeddingStore.load(save_path)
    vec = store.get(0)
    assert vec.shape == (768,)


def test_embedding_store_round_trip(tmp_path, mock_embeddings):
    """save then load reproduces the original array values."""
    save_path = str(tmp_path / "test_embeddings.npy")
    np.save(save_path, mock_embeddings)
    store = EmbeddingStore.load(save_path)
    np.testing.assert_array_almost_equal(store.get_all(), mock_embeddings)


def test_embedding_store_rejects_1d_array():
    """EmbeddingStore constructor raises AssertionError for 1D input (per fail-loudly)."""
    with pytest.raises(AssertionError):
        EmbeddingStore(np.zeros(768))


def test_embedding_store_get_all_is_readonly(tmp_path, mock_embeddings):
    """get_all() returns a read-only view: mutation must raise ValueError."""
    save_path = str(tmp_path / "emb.npy")
    np.save(save_path, mock_embeddings)
    store = EmbeddingStore.load(save_path)
    arr = store.get_all()
    assert not arr.flags.writeable, "get_all() must return a read-only view"
    with pytest.raises(ValueError):
        arr[0, 0] = 9999.0
