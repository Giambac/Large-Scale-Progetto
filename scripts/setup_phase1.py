"""
setup_phase1.py — End-to-end Phase 1 setup script.

Runs the full pipeline from pre-checked artifacts to an initial ClusteringState:
    1. Verify held-out split hash (crashes if tampered)
    2. Load pre-computed embeddings from embeddings/embeddings.npy
    3. Load training texts from dataset/train.jsonl
    4. Build initial ClusteringState via HDBSCAN + LLM naming
    5. Append turn 0 to audit_log.jsonl

Prerequisites (must exist before running this script):
    - dataset/held_out.jsonl and dataset/held_out.sha256 (from data_loader.py)
    - dataset/train.jsonl (from data_loader.py)
    - embeddings/embeddings.npy (from embedding_store.py)
    - ANTHROPIC_API_KEY environment variable set

Usage:
    python scripts/setup_phase1.py
    python scripts/setup_phase1.py --audit-log custom_audit.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 end-to-end setup")
    parser.add_argument(
        "--held-out",
        default="dataset/held_out.jsonl",
        help="Path to held-out split JSONL",
    )
    parser.add_argument(
        "--hash-file",
        default="dataset/held_out.sha256",
        help="Path to SHA-256 hash file for held-out split",
    )
    parser.add_argument(
        "--embeddings",
        default="embeddings/embeddings.npy",
        help="Path to pre-computed embeddings .npy file",
    )
    parser.add_argument(
        "--train",
        default="dataset/train.jsonl",
        help="Path to training split JSONL",
    )
    parser.add_argument(
        "--audit-log",
        default="audit_log.jsonl",
        help="Path to AuditLog JSONL (appended, not overwritten)",
    )
    args = parser.parse_args()

    # Validate prerequisites
    for path, label in [
        (args.held_out, "held-out split"),
        (args.hash_file, "SHA-256 hash file"),
        (args.embeddings, "embeddings file"),
        (args.train, "training split"),
    ]:
        assert os.path.exists(path), (
            f"Required file missing: {path} ({label})\n"
            "Run src/data_loader.py and src/embedding_store.py first."
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    assert api_key, (
        "ANTHROPIC_API_KEY environment variable is not set.\n"
        "Export it before running: export ANTHROPIC_API_KEY=sk-ant-..."
    )

    # Step 1: Verify held-out hash (crashes on mismatch)
    print("[1/5] Verifying held-out split integrity...")
    from src.data_loader import verify_held_out_hash
    verify_held_out_hash(args.held_out, args.hash_file)
    print("      Hash verification passed.")

    # Step 2: Load embeddings
    print("[2/5] Loading embeddings...")
    from src.embedding_store import EmbeddingStore
    store = EmbeddingStore.load(args.embeddings)
    print(f"      Loaded {len(store)} embeddings, dim={store.get(0).shape[0]}")

    # Step 3: Load training texts + records
    print("[3/5] Loading training texts...")
    records = []
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    assert len(records) == len(store), (
        f"Record count {len(records)} != embedding count {len(store)}. "
        "Did you re-generate the dataset without re-computing embeddings?"
    )
    print(f"      Loaded {len(records)} records.")

    # Step 4: Build initial ClusteringState
    print("[4/5] Building initial ClusteringState (HDBSCAN + LLM naming)...")
    print("      This may take 1-5 minutes depending on the number of clusters.")
    import anthropic
    from src.cluster_naming import AnthropicClusterNamer
    from src.clustering import build_initial_clustering_state

    # Build client and namer outside any try block — per CLAUDE.md, only external
    # API calls may have try/except. The try/except for anthropic.APIError belongs
    # inside cluster_naming.py's name_cluster(), wrapping only client.messages.create().
    # In this CLI entry point, a top-level try/except Exception is acceptable only
    # at the outermost main() boundary; internal pipeline calls are outside try.
    client = anthropic.Anthropic(api_key=api_key)
    namer = AnthropicClusterNamer(client)
    # build_initial_clustering_state is a pure pipeline function — called OUTSIDE try.
    # If the LLM call inside it raises anthropic.APIError, that error propagates up
    # and will be caught by the outermost exception handler at the CLI entry point.
    state = build_initial_clustering_state(
        embeddings=store.get_all(),
        records=records,
        namer=namer,
    )

    assert len(state.clusters) > 0, (
        "HDBSCAN produced 0 clusters — all points classified as noise. "
        "Increase min_cluster_size or check the embeddings."
    )
    print(f"      Produced {len(state.clusters)} clusters at turn_index={state.turn_index}:")
    for cluster in state.clusters:
        print(f"        Cluster {cluster.id}: '{cluster.name}' ({len(cluster.item_ids)} items)")

    # Step 5: Write turn 0 to AuditLog
    print(f"[5/5] Appending turn 0 to AuditLog: {args.audit_log}")
    from src.serialization import append_to_audit_log
    append_to_audit_log(state, args.audit_log)
    print(f"      Done. AuditLog written to {args.audit_log}")

    print("\nPhase 1 setup complete.")
    print(f"  Clusters: {len(state.clusters)}")
    print(f"  Items assigned: {len(state.assignments)}")
    print(f"  AuditLog: {args.audit_log}")


if __name__ == "__main__":
    # Top-level CLI exception handler — the only permitted try/except in this file.
    # All internal pipeline calls inside main() are outside any try block.
    try:
        main()
    except Exception as e:
        print(f"Phase 1 setup failed: {e}", file=sys.stderr)
        raise
