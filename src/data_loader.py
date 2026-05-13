"""
data_loader.py — Scarica i dati, li divide e protegge il held-out.

Questo file fa tre cose, tutte eseguite una volta sola all'inizio del progetto:

    1. Scarica 15.000 recensioni Amazon (categoria Arts & Crafts) da HuggingFace in streaming — senza scaricare tutti i 9 milioni di righe del dataset.

    2. Divide le 15.000 recensioni in due parti: 12.000 per il training (usate per clustering e conversazione) e 3.000 held-out (chiuse in
    cassaforte per la valutazione finale nella Fase 6).

    3. Calcola l'impronta digitale SHA-256 del file held-out e la salva. Ogni volta che il sistema parte, l'hash viene ricalcolato e confrontato — 
    se il file è stato toccato, il programma crasha immediatamente.
"""
import hashlib
import json
import os
import random
from typing import Optional


# Costanti — non modificare dopo che held_out.jsonl è stato creato
DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"
DATASET_CONFIG = "raw_review_Arts_Crafts_and_Sewing"
HF_TEXT_FIELD = "text"          # nome del campo testo su HuggingFace
TARGET_SAMPLE_SIZE = 15_000
TRAIN_RATIO = 0.80
DEFAULT_SEED = 42

"""
def compute_sha256( )
    Calcola l'impronta digitale SHA-256 di un file.

    Legge il file a blocchi da 64KB per non caricare tutto in memoria.
    Restituisce una stringa esadecimale di 64 caratteri.
    Se il contenuto del file cambia anche di un solo carattere,
    l'impronta cambia completamente.
"""
def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

"""
def verify_held_out_hash( )
    Verifica che il file held-out non sia stato modificato dal momento in cui è stato creato.

    Legge l'hash salvato in hash_path, ricalcola l'hash del file held-out, e crasha immediatamente se non coincidono. 
    Viene chiamata ogni volta che il sistema parte — è la guardia che impedisce contaminazioni accidentali del set di valutazione.
"""
def verify_held_out_hash(held_out_path: str, hash_path: str) -> None:
    with open(hash_path) as f:
        expected = f.read().strip()
    actual = compute_sha256(held_out_path)
    assert actual == expected, (
        f"Held-out split hash mismatch!\n"
        f"  Expected: {expected}\n"
        f"  Actual:   {actual}\n"
        f"  File may have been modified: {held_out_path}"
    )

"""
def split_dataset( )
    Divide le recensioni in training e held-out in modo riproducibile.

    Con seed=42 la divisione è sempre identica — se la rifai domani ottieni gli stessi due gruppi.

    Restituisce (train, held_out) con:
        - train : floor(N * 0.80) record
        - held_out : i restanti record
"""
def split_dataset(
    records: list[dict],
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], list[dict]]:
    assert len(records) > 0, "Cannot split an empty list"
    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(len(records) * TRAIN_RATIO)
    train = [records[i] for i in indices[:n_train]]
    held_out = [records[i] for i in indices[n_train:]]
    assert len(train) + len(held_out) == len(records)
    return train, held_out

"""
    Setup iniziale: scarica 15K recensioni, divide 80/20, salva l'hash SHA-256.

    Viene eseguita una volta sola. Se held_out.jsonl o held_out.sha256 esistono già, crasha immediatamente invece di sovrascriverli — sovrascrivere il
    held-out dopo aver fatto esperimenti contaminerebbe la valutazione finale.

    File scritti:
        dataset/arts_crafts_15k.jsonl       — tutte le 15K recensioni grezze
        dataset/train.jsonl                 — 12.000 recensioni per il training
        dataset/held_out.jsonl              — 3.000 recensioni bloccate — NON RIGENERARE
        dataset/held_out.sha256             — hash SHA-256 del held-out — NON RIGENERARE
"""
def download_and_save_dataset(
    output_dir: str = "dataset",
    raw_path: Optional[str] = None,
    train_path: Optional[str] = None,
    held_out_path: Optional[str] = None,
    hash_path: Optional[str] = None,
    seed: int = DEFAULT_SEED,
) -> None:
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

    # Re-index train records so item_id == row index (0..N_train-1).
    # This ensures EmbeddingStore row index == item_id throughout (D-11).
    # held_out keeps original item_ids since it is never used with EmbeddingStore.
    train = [{"item_id": i, "text": r["text"]} for i, r in enumerate(train)]

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
