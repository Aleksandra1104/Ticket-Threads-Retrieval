from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ticket resolution retrieval from pair CSV."
    )
    parser.add_argument("--input", required=True, help="Path to train_pairs.csv")
    parser.add_argument("--model", required=True, help="Model path or HF name")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="ticket_eval_report.json")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {"sentence1", "sentence2", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        for row in reader:
            s1 = (row.get("sentence1") or "").strip()
            s2 = (row.get("sentence2") or "").strip()
            label = float(row.get("label", 0.0))

            if not s1 or not s2:
                continue

            rows.append(
                {
                    "sentence1": s1,
                    "sentence2": s2,
                    "label": label,
                    "ticket_id": row.get("ticket_id", ""),
                    "issue_category": row.get("issue_category", ""),
                }
            )

    return rows


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def reciprocal_rank(top_indices: np.ndarray, positive_indices: set[int]) -> float:
    for rank, idx in enumerate(top_indices, start=1):
        if idx in positive_indices:
            return 1.0 / rank
    return 0.0


def evaluate(rows: List[Dict[str, str]], model: SentenceTransformer, top_k: int):
    queries = sorted(set(row["sentence1"] for row in rows))
    corpus = [row["sentence2"] for row in rows]

    query_embeddings = model.encode(
        queries,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    corpus_embeddings = model.encode(
        corpus,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    query_embeddings = normalize(query_embeddings)
    corpus_embeddings = normalize(corpus_embeddings)

    scores = query_embeddings @ corpus_embeddings.T

    recall_hits = {k: 0 for k in range(1, top_k + 1)}
    rr_total = 0.0
    examples = []

    query_to_positive_indices = {}

    for q in queries:
        query_category = None

        for row in rows:
            if row["sentence1"] == q:
                query_category = row["issue_category"]
                break

        positive_indices = {
            i
            for i, row in enumerate(rows)
            if row["issue_category"] == query_category and row["label"] == 1.0
        }

        query_to_positive_indices[q] = positive_indices

    for i, query in enumerate(queries):
        ranked = np.argsort(scores[i])[::-1]
        positives = query_to_positive_indices[query]

        rr = reciprocal_rank(ranked, positives)
        rr_total += rr

        for k in range(1, top_k + 1):
            if any(idx in positives for idx in ranked[:k]):
                recall_hits[k] += 1

        best_idx = int(ranked[0])
        examples.append(
            {
                "query": query,
                "top1_resolution": corpus[best_idx],
                "top1_score": float(scores[i][best_idx]),
                "reciprocal_rank": rr,
            }
        )

    n = len(queries)

    metrics = {
        f"recall@{k}": recall_hits[k] / n
        for k in range(1, top_k + 1)
    }
    metrics["mrr"] = rr_total / n

    random.shuffle(examples)

    return {
        "num_queries": n,
        "metrics": metrics,
        "examples": examples[:10],
    }


def main() -> int:
    args = parse_args()

    rows = load_rows(Path(args.input))
    model = SentenceTransformer(args.model)

    report = evaluate(rows, model, args.top_k)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved report to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())