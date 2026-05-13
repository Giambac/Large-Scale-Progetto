"""
embedding_store.py — Contenitore read-only per gli embeddings pre-calcolati.

Questo file gestisce la tabella 12.000 × 768 degli embeddings: una riga per recensione, 768 numeri per riga. Questi numeri catturano il significato
semantico di ogni recensione — due recensioni simili hanno vettori numericamente vicini, due recensioni diverse hanno vettori lontani.

La tabella viene calcolata una volta sola (operazione di 10-30 minuti su CPU) e salvata in un file .npy da ~37MB. Tutte le sessioni successive la leggono
da quel file senza ricalcolarla.

Regola fondamentale: la tabella è in sola lettura dopo il caricamento.
Se qualcuno prova a modificare un valore, il programma crasha subito.
Questo garantisce che gli embeddings rimangano identici per tutto il progetto — se cambiassero, i risultati degli esperimenti non sarebbero più confrontabili.
"""
from __future__ import annotations

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer


# Modello bloccato per questo progetto — non cambiare senza ricalcolare gli embeddings.
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768       # dimensione dell'output di all-mpnet-base-v2
BATCH_SIZE = 32           # sicuro per inferenza su CPU; aumentare se disponibile una GPU

"""
class EmbeddingStore:
    Contenitore read-only per gli embeddings pre-calcolati.

    Internamente tiene una matrice (N, 768) in sola lettura.
    Gli item_id sono indici di riga interi sequenziali (0, 1, 2, ...).

    Utilizzo tipico:
        -> Una volta sola al setup iniziale:
            store = EmbeddingStore.compute_and_save(texts, "embeddings/embeddings.npy")

        -> Ad ogni sessione successiva:
            store = EmbeddingStore.load("embeddings/embeddings.npy")
            vec = store.get(item_id)    # vettore di una singola recensione, shape (768,)
            all_vecs = store.get_all()  # tutta la matrice, shape (N, 768)
"""
class EmbeddingStore:
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

    """
    def compute_and_save( )
        Codifica tutti i testi con all-mpnet-base-v2 e salva la matrice su file.

        Operazione lenta (10-30 minuti su CPU per 12K testi) — va eseguita una volta sola. Tutte le sessioni successive usano EmbeddingStore.load().
    """
    @classmethod
    def compute_and_save(
        cls,
        texts: list[str],
        save_path: str,
    ) -> "EmbeddingStore":
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
        """Carica gli embeddings pre-calcolati da un file .npy."""
        assert os.path.exists(save_path), f"Embeddings file not found: {save_path}"
        embeddings = np.load(save_path)
        return cls(embeddings)

    def get(self, item_id: int) -> np.ndarray:
        """Restituisce il vettore embedding di una singola recensione. Shape: (768,)."""
        assert 0 <= item_id < len(self._embeddings), (
            f"item_id {item_id} out of range [0, {len(self._embeddings)})"
        )
        return self._embeddings[item_id]

    def get_all(self) -> np.ndarray:
        """Restituisce tutta la matrice degli embeddings. Shape: (N, 768). Read-only."""
        return self._embeddings

    def __len__(self) -> int:
        return len(self._embeddings)

"""
def load_texts_from_jsonl( )
    Carica i testi delle recensioni da un file JSONL prodotto da data_loader.py.

    Legge ogni riga del file e restituisce la lista dei testi nel campo "text".
    Crasha se il file non esiste o se una riga non ha il campo "text".
"""
def load_texts_from_jsonl(jsonl_path: str) -> list[str]:
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
    """Calcolo embedding una tantum. Eseguire dopo che data_loader.py ha completato."""
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
