from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Set

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CATEGORY_TO_FAMILY: Dict[str, str] = {
    "account_locked": "IAM",
    "password_reset": "IAM",
    "mfa_issue": "IAM",
    "permission_issue": "IAM",
    "onboarding_offboarding": "IAM",
    "vpn_issue": "Networking",
    "wifi_connectivity": "Networking",
    "internet_access": "Networking",
    "voip_telephony": "Networking",
    "workstation_failure": "Hardware",
    "peripheral_issue": "Hardware",
    "printer_issue": "Hardware",
    "mobile_device_issue": "Hardware",
    "email_issue": "Software",
    "software_install": "Software",
    "application_crash": "Software",
    "browser_issue": "Software",
    "shared_drive_issue": "Storage",
    "data_recovery": "Storage",
    "disk_space_full": "Storage",
    "phishing_report": "Security",
    "malware_infection": "Security",
    "encryption_issue": "Security",
    "server_unavailable": "Backend",
    "database_connection": "Backend",
    "api_failure": "Backend",
    "other": "Other",
}


@dataclass
class PairRow:
    sentence1: str
    sentence2: str
    label: float
    ticket_id: str
    issue_category: str
    family: str


@dataclass
class QueryItem:
    query: str
    ticket_id: str
    issue_category: str
    family: str
    gold_resolution: str


@dataclass
class CorpusItem:
    text: str
    ticket_id: str
    issue_category: str
    family: str


def category_family(category: str) -> str:
    return CATEGORY_TO_FAMILY.get((category or "").strip(), "Unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ticket-resolution retrieval from pair CSV."
    )
    parser.add_argument("--input", required=True, help="Path to pair CSV")
    parser.add_argument("--model", required=True, help="Model path or HF name")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum K for Recall@K")
    parser.add_argument(
        "--relevance-mode",
        choices=["exact", "category", "family"],
        default="exact",
        help=(
            "How to define a relevant retrieval:\n"
            "exact    = same ticket_id\n"
            "category = same issue category\n"
            "family   = same issue family"
        ),
    )
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        help="Exclude the exact gold memory for the query ticket from the searchable corpus.",
    )
    parser.add_argument(
        "--output",
        default="ticket_eval_report.json",
        help="Path to save JSON report",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling example rows",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[PairRow]:
    rows: List[PairRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {"sentence1", "sentence2", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        for row in reader:
            sentence1 = (row.get("sentence1") or "").strip()
            sentence2 = (row.get("sentence2") or "").strip()
            if not sentence1 or not sentence2:
                continue

            label = float(row.get("label", 0.0))
            issue_category = (
                (row.get("ground_truth_primary_issue_category") or "").strip()
                or (row.get("issue_category") or "").strip()
            )

            rows.append(
                PairRow(
                    sentence1=sentence1,
                    sentence2=sentence2,
                    label=label,
                    ticket_id=(row.get("ticket_id") or "").strip(),
                    issue_category=issue_category,
                    family=category_family(issue_category),
                )
            )

    if not rows:
        raise ValueError("No usable rows were loaded from the input CSV.")

    return rows


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def build_queries_and_corpus(rows: List[PairRow]) -> tuple[List[QueryItem], List[CorpusItem]]:
    positive_rows = [row for row in rows if row.label == 1.0]
    if not positive_rows:
        raise ValueError("No positive rows found. Retrieval corpus cannot be built.")

    queries: List[QueryItem] = []
    corpus: List[CorpusItem] = []

    # corpus = only valid historical memories
    for row in positive_rows:
        corpus.append(
            CorpusItem(
                text=row.sentence2,
                ticket_id=row.ticket_id,
                issue_category=row.issue_category,
                family=row.family,
            )
        )

    # queries = one per ticket_id using its positive row
    seen_ticket_ids: Set[str] = set()
    for row in positive_rows:
        if row.ticket_id in seen_ticket_ids:
            continue
        seen_ticket_ids.add(row.ticket_id)

        queries.append(
            QueryItem(
                query=row.sentence1,
                ticket_id=row.ticket_id,
                issue_category=row.issue_category,
                family=row.family,
                gold_resolution=row.sentence2,
            )
        )

    if not queries:
        raise ValueError("No queries were created from positive rows.")

    return queries, corpus


def relevant_indices_for_query(
    query: QueryItem,
    corpus: List[CorpusItem],
    relevance_mode: str,
    exclude_self: bool,
) -> Set[int]:
    relevant: Set[int] = set()

    for i, item in enumerate(corpus):
        if exclude_self and item.ticket_id == query.ticket_id:
            continue

        if relevance_mode == "exact":
            if item.ticket_id == query.ticket_id:
                relevant.add(i)
        elif relevance_mode == "category":
            if item.issue_category == query.issue_category:
                relevant.add(i)
        elif relevance_mode == "family":
            if item.family == query.family:
                relevant.add(i)
        else:
            raise ValueError(f"Unsupported relevance mode: {relevance_mode}")

    return relevant


def reciprocal_rank(top_indices: np.ndarray, positive_indices: Set[int]) -> float:
    for rank, idx in enumerate(top_indices, start=1):
        if int(idx) in positive_indices:
            return 1.0 / rank
    return 0.0


def evaluate(
    queries: List[QueryItem],
    corpus: List[CorpusItem],
    model: SentenceTransformer,
    top_k: int,
    relevance_mode: str,
    exclude_self: bool,
    rng: random.Random,
) -> Dict:
    query_texts = [q.query for q in queries]
    corpus_texts = [c.text for c in corpus]

    query_embeddings = model.encode(
        query_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    corpus_embeddings = model.encode(
        corpus_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    query_embeddings = normalize(query_embeddings)
    corpus_embeddings = normalize(corpus_embeddings)

    scores = query_embeddings @ corpus_embeddings.T

    recall_hits = {k: 0 for k in range(1, top_k + 1)}
    rr_total = 0.0
    first_positive_ranks: List[int] = []
    top1_exact_category_hits = 0
    top1_family_hits = 0
    skipped_queries = 0
    examples: List[Dict] = []

    for i, query in enumerate(queries):
        positives = relevant_indices_for_query(
            query=query,
            corpus=corpus,
            relevance_mode=relevance_mode,
            exclude_self=exclude_self,
        )

        if not positives:
            skipped_queries += 1
            continue

        ranked = np.argsort(scores[i])[::-1]

        rr = reciprocal_rank(ranked, positives)
        rr_total += rr

        first_positive_rank: Optional[int] = None
        for rank, idx in enumerate(ranked, start=1):
            if int(idx) in positives:
                first_positive_rank = rank
                first_positive_ranks.append(rank)
                break

        for k in range(1, top_k + 1):
            if any(int(idx) in positives for idx in ranked[:k]):
                recall_hits[k] += 1

        best_idx = int(ranked[0])
        top1_item = corpus[best_idx]

        if top1_item.issue_category == query.issue_category:
            top1_exact_category_hits += 1
        if top1_item.family == query.family:
            top1_family_hits += 1

        examples.append(
            {
                "query": query.query,
                "query_ticket_id": query.ticket_id,
                "query_issue_category": query.issue_category,
                "query_family": query.family,
                "gold_resolution": query.gold_resolution,
                "top1_resolution": top1_item.text,
                "top1_score": float(scores[i][best_idx]),
                "top1_ticket_id": top1_item.ticket_id,
                "top1_issue_category": top1_item.issue_category,
                "top1_family": top1_item.family,
                "top1_exact_category_match": top1_item.issue_category == query.issue_category,
                "top1_family_match": top1_item.family == query.family,
                "reciprocal_rank": rr,
                "first_positive_rank": first_positive_rank,
            }
        )

    evaluated_n = len(queries) - skipped_queries
    if evaluated_n <= 0:
        raise ValueError("No queries had any valid positives under the selected evaluation mode.")

    metrics = {
        f"recall@{k}": recall_hits[k] / evaluated_n
        for k in range(1, top_k + 1)
    }
    metrics["mrr"] = rr_total / evaluated_n
    metrics["top1_exact_category_accuracy"] = top1_exact_category_hits / evaluated_n
    metrics["top1_family_accuracy"] = top1_family_hits / evaluated_n
    metrics["mean_first_positive_rank"] = (
        float(sum(first_positive_ranks) / len(first_positive_ranks))
        if first_positive_ranks
        else 0.0
    )
    metrics["median_first_positive_rank"] = (
        float(median(first_positive_ranks))
        if first_positive_ranks
        else 0.0
    )

    rng.shuffle(examples)

    return {
        "num_queries_total": len(queries),
        "num_queries_evaluated": evaluated_n,
        "num_queries_skipped": skipped_queries,
        "num_corpus_items": len(corpus),
        "relevance_mode": relevance_mode,
        "exclude_self": exclude_self,
        "metrics": metrics,
        "examples": examples[:10],
    }


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    rows = load_rows(Path(args.input))
    queries, corpus = build_queries_and_corpus(rows)
    model = SentenceTransformer(args.model)

    report = evaluate(
        queries=queries,
        corpus=corpus,
        model=model,
        top_k=args.top_k,
        relevance_mode=args.relevance_mode,
        exclude_self=args.exclude_self,
        rng=rng,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))

    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
