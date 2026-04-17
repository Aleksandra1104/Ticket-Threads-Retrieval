from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==============================
# Taxonomy
# ==============================
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


def category_family(category: str) -> str:
    return CATEGORY_TO_FAMILY.get((category or "").strip(), "Unknown")


# ==============================
# Domain objects
# ==============================
@dataclass
class GroundTruthTicket:
    ticket_id: str
    status: str
    title: str
    ground_truth_issue_titles: List[str]
    ground_truth_primary_issue_category: str
    ground_truth_secondary_issue_category: str
    ground_truth_resolution_state: str
    ground_truth_resolution_summary: str
    ground_truth_should_extract: bool
    metadata: Dict[str, Any]


@dataclass
class ExtractedPair:
    ticket_id: str
    query: str
    positive: str
    issue_category: str
    confidence: float
    metadata: Dict[str, Any]


# ==============================
# CLI
# ==============================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate extracted ticket pairs against original synthetic tickets."
    )
    parser.add_argument(
        "--tickets",
        required=True,
        help="Path to original ticket JSONL with nested ground_truth object",
    )
    parser.add_argument(
        "--extracted",
        required=True,
        help="Path to extracted_pairs.jsonl",
    )
    parser.add_argument(
        "--skipped",
        help="Optional path to skipped_tickets.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where evaluation outputs will be written",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for semantic similarity",
    )
    parser.add_argument(
        "--only-resolved",
        action="store_true",
        help="Evaluate only tickets with ground_truth.resolution_state == 'resolved'",
    )
    return parser.parse_args()


# ==============================
# File IO
# ==============================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at line {line_no} in {path}")
            rows.append(obj)
    return rows


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ==============================
# Basic math helpers
# ==============================
def safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


# ==============================
# Text helpers
# ==============================
def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: Any) -> List[str]:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def token_f1(a: str, b: str) -> float:
    a_tokens = tokenize(a)
    b_tokens = tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0

    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)

    overlap = 0
    for token in a_counts:
        overlap += min(a_counts[token], b_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def embedding_similarity(a: str, b: str, embedding_model: SentenceTransformer) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    vecs = embedding_model.encode([a, b], normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))


def best_issue_text_match(
    pred_issue: str,
    gt_titles: List[str],
    embedding_model: SentenceTransformer,
) -> Dict[str, Any]:
    """
    Compare predicted issue text against all ground-truth title variants
    and keep the best score for each metric.
    """
    pred_issue = (pred_issue or "").strip()
    titles = [str(t).strip() for t in gt_titles if str(t).strip()]

    if not pred_issue or not titles:
        return {
            "best_title_match": "",
            "token_f1": 0.0,
            "sequence_similarity": 0.0,
            "embedding_similarity": 0.0,
        }

    best_title_by_embedding = ""
    best_token_f1 = 0.0
    best_seq_sim = 0.0
    best_emb_sim = 0.0

    pred_vec = embedding_model.encode([pred_issue], normalize_embeddings=True)[0]

    for title in titles:
        tok = token_f1(pred_issue, title)
        seq = string_similarity(pred_issue, title)

        title_vec = embedding_model.encode([title], normalize_embeddings=True)[0]
        emb = float(np.dot(pred_vec, title_vec))

        if tok > best_token_f1:
            best_token_f1 = tok
        if seq > best_seq_sim:
            best_seq_sim = seq
        if emb > best_emb_sim:
            best_emb_sim = emb
            best_title_by_embedding = title

    return {
        "best_title_match": best_title_by_embedding,
        "token_f1": best_token_f1,
        "sequence_similarity": best_seq_sim,
        "embedding_similarity": best_emb_sim,
    }


# ==============================
# Loading domain objects
# ==============================
def load_ground_truth_tickets(path: Path) -> Dict[str, GroundTruthTicket]:
    rows = read_jsonl(path)
    tickets: Dict[str, GroundTruthTicket] = {}

    for row in rows:
        ticket_id = str(row.get("ticket_id") or "").strip()
        if not ticket_id:
            continue

        gt = row.get("ground_truth")
        if not isinstance(gt, dict):
            gt = {}

        gt_meta = gt.get("metadata")
        if not isinstance(gt_meta, dict):
            gt_meta = {}

        title = str(row.get("title") or "").strip()

        raw_titles = row.get("ground_truth_issue_titles")
        if isinstance(raw_titles, list):
            ground_truth_issue_titles = [
                str(x).strip() for x in raw_titles if str(x).strip()
            ]
        else:
            ground_truth_issue_titles = [title] if title else []

        tickets[ticket_id] = GroundTruthTicket(
            ticket_id=ticket_id,
            status=str(row.get("status") or ""),
            title=title,
            ground_truth_issue_titles=ground_truth_issue_titles,
            ground_truth_primary_issue_category=str(
                gt.get("primary_issue_category") or ""
            ).strip(),
            ground_truth_secondary_issue_category=str(
                gt.get("secondary_issue_category") or ""
            ).strip(),
            ground_truth_resolution_state=str(
                gt.get("resolution_state") or ""
            ).strip(),
            ground_truth_resolution_summary=str(
                gt.get("resolution_summary") or ""
            ).strip(),
            ground_truth_should_extract=bool(gt.get("should_extract")),
            metadata=gt_meta,
        )
    return tickets


def load_extracted_pairs(path: Path) -> Dict[str, ExtractedPair]:
    rows = read_jsonl(path)
    extracted: Dict[str, ExtractedPair] = {}

    for row in rows:
        ticket_id = str(row.get("ticket_id") or "").strip()
        if not ticket_id:
            continue

        confidence_raw = row.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0

        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        extracted[ticket_id] = ExtractedPair(
            ticket_id=ticket_id,
            query=str(row.get("query") or "").strip(),
            positive=str(row.get("positive") or "").strip(),
            issue_category=str(row.get("issue_category") or "").strip(),
            confidence=confidence,
            metadata=metadata,
        )
    return extracted


def load_skipped_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    return read_jsonl(path)


# ==============================
# Evaluation eligibility
# ==============================
def should_ticket_be_extractable(ticket: GroundTruthTicket, only_resolved: bool) -> bool:
    resolution_state = ticket.ground_truth_resolution_state.strip().lower()

    if only_resolved:
        return resolution_state == "resolved"

    if resolution_state:
        return resolution_state == "resolved"

    return ticket.ground_truth_should_extract


# ==============================
# Evaluation
# ==============================
def evaluate(
    tickets: Dict[str, GroundTruthTicket],
    extracted: Dict[str, ExtractedPair],
    skipped: List[Dict[str, Any]],
    only_resolved: bool,
    embedding_model: SentenceTransformer,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    eligible_ticket_ids: List[str] = []
    all_ticket_ids: List[str] = list(tickets.keys())

    for ticket_id, ticket in tickets.items():
        if should_ticket_be_extractable(ticket, only_resolved=only_resolved):
            eligible_ticket_ids.append(ticket_id)

    eligible_set = set(eligible_ticket_ids)
    extracted_set = set(extracted.keys())
    all_set = set(all_ticket_ids)

    true_positives = len(eligible_set & extracted_set)
    false_positives = len(extracted_set - eligible_set)
    false_negatives = len(eligible_set - extracted_set)
    true_negatives = len((all_set - eligible_set) - extracted_set)

    extraction_precision = safe_div(true_positives, true_positives + false_positives)
    extraction_recall = safe_div(true_positives, true_positives + false_negatives)
    extraction_f1 = f1_score(extraction_precision, extraction_recall)

    exact_category_total = 0
    exact_category_correct = 0

    family_total = 0
    family_correct = 0

    confidence_values: List[float] = []

    issue_token_f1_values: List[float] = []
    issue_sequence_similarity_values: List[float] = []
    issue_embedding_similarity_values: List[float] = []

    resolution_token_f1_values: List[float] = []
    resolution_sequence_similarity_values: List[float] = []
    resolution_embedding_similarity_values: List[float] = []

    category_confusion: Dict[str, Counter] = defaultdict(Counter)
    family_confusion: Dict[str, Counter] = defaultdict(Counter)

    predicted_category_counts = Counter()
    ground_truth_category_counts = Counter()
    predicted_family_counts = Counter()
    ground_truth_family_counts = Counter()

    per_ticket_rows: List[Dict[str, Any]] = []

    for ticket_id in sorted(eligible_set & extracted_set):
        gt_ticket = tickets[ticket_id]
        pred = extracted[ticket_id]

        # --- Category evaluation
        gt_category = gt_ticket.ground_truth_primary_issue_category
        pred_category = pred.issue_category

        gt_family = category_family(gt_category) if gt_category else ""
        pred_family = category_family(pred_category) if pred_category else ""

        category_match = gt_category == pred_category if gt_category else False
        family_match = gt_family == pred_family if gt_family else False

        if gt_category:
            exact_category_total += 1
            ground_truth_category_counts[gt_category] += 1
            predicted_category_counts[pred_category or "<empty>"] += 1
            category_confusion[gt_category][pred_category or "<empty>"] += 1
            if category_match:
                exact_category_correct += 1

        if gt_family:
            family_total += 1
            ground_truth_family_counts[gt_family] += 1
            predicted_family_counts[pred_family or "<empty>"] += 1
            family_confusion[gt_family][pred_family or "<empty>"] += 1
            if family_match:
                family_correct += 1

        confidence_values.append(pred.confidence)

        # --- Issue text evaluation against title variants
        pred_issue = pred.query.strip()
        gt_issue_titles = gt_ticket.ground_truth_issue_titles

        issue_match = best_issue_text_match(
            pred_issue=pred_issue,
            gt_titles=gt_issue_titles,
            embedding_model=embedding_model,
        )

        issue_token_f1_values.append(issue_match["token_f1"])
        issue_sequence_similarity_values.append(issue_match["sequence_similarity"])
        issue_embedding_similarity_values.append(issue_match["embedding_similarity"])

        # --- Resolution text evaluation against ground-truth resolution summary
        gt_resolution = gt_ticket.ground_truth_resolution_summary.strip()
        pred_resolution = pred.positive.strip()

        res_tok_f1 = token_f1(pred_resolution, gt_resolution) if gt_resolution else 0.0
        res_seq_sim = string_similarity(pred_resolution, gt_resolution) if gt_resolution else 0.0
        res_emb_sim = embedding_similarity(pred_resolution, gt_resolution, embedding_model) if gt_resolution else 0.0

        if gt_resolution:
            resolution_token_f1_values.append(res_tok_f1)
            resolution_sequence_similarity_values.append(res_seq_sim)
            resolution_embedding_similarity_values.append(res_emb_sim)

        per_ticket_rows.append(
            {
                "ticket_id": ticket_id,
                "title": gt_ticket.title,
                "ground_truth_issue_titles": gt_issue_titles,
                "best_matching_ground_truth_title": issue_match["best_title_match"],
                "predicted_issue_text": pred_issue,
                "issue_token_f1": round(issue_match["token_f1"], 4),
                "issue_sequence_similarity": round(issue_match["sequence_similarity"], 4),
                "issue_embedding_similarity": round(issue_match["embedding_similarity"], 4),

                "ground_truth_resolution_summary": gt_resolution,
                "predicted_resolution_summary": pred_resolution,
                "resolution_token_f1": round(res_tok_f1, 4),
                "resolution_sequence_similarity": round(res_seq_sim, 4),
                "resolution_embedding_similarity": round(res_emb_sim, 4),

                "ground_truth_resolution_state": gt_ticket.ground_truth_resolution_state,
                "ground_truth_primary_issue_category": gt_category,
                "predicted_issue_category": pred_category,
                "category_match": category_match,
                "ground_truth_category_family": gt_family,
                "predicted_category_family": pred_family,
                "family_match": family_match,
                "confidence": pred.confidence,
                "query": pred.query,
                "positive": pred.positive,
            }
        )

    skipped_reason_counts = Counter()
    for row in skipped:
        reason = str(row.get("reason") or "unknown")
        skipped_reason_counts[reason] += 1

    report: Dict[str, Any] = {
        "dataset": {
            "total_tickets": len(all_ticket_ids),
            "eligible_tickets": len(eligible_ticket_ids),
            "extracted_tickets": len(extracted),
            "evaluation_scope": "resolved_only" if only_resolved else "default_extractable",
        },
        "extraction_metrics": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "precision": round(extraction_precision, 4),
            "recall": round(extraction_recall, 4),
            "f1": round(extraction_f1, 4),
        },
        "category_metrics": {
            "evaluated_predictions": exact_category_total,
            "correct_predictions": exact_category_correct,
            "accuracy": round(safe_div(exact_category_correct, exact_category_total), 4),
        },
        "family_metrics": {
            "evaluated_predictions": family_total,
            "correct_predictions": family_correct,
            "accuracy": round(safe_div(family_correct, family_total), 4),
        },
        "text_similarity_metrics": {
            "issue": {
                "avg_token_f1": round(safe_mean(issue_token_f1_values), 4),
                "avg_sequence_similarity": round(safe_mean(issue_sequence_similarity_values), 4),
                "avg_embedding_similarity": round(safe_mean(issue_embedding_similarity_values), 4),
            },
            "resolution": {
                "avg_token_f1": round(safe_mean(resolution_token_f1_values), 4),
                "avg_sequence_similarity": round(safe_mean(resolution_sequence_similarity_values), 4),
                "avg_embedding_similarity": round(safe_mean(resolution_embedding_similarity_values), 4),
            },
        },
        "confidence_metrics": {
            "avg_confidence": round(safe_mean(confidence_values), 4),
            "count": len(confidence_values),
        },
        "category_distributions": {
            "ground_truth_categories": dict(sorted(ground_truth_category_counts.items())),
            "predicted_categories": dict(sorted(predicted_category_counts.items())),
            "ground_truth_families": dict(sorted(ground_truth_family_counts.items())),
            "predicted_families": dict(sorted(predicted_family_counts.items())),
        },
        "confusion_matrix": {
            "fine_grained": {
                gt_cat: dict(pred_counter)
                for gt_cat, pred_counter in sorted(category_confusion.items())
            },
            "family_level": {
                gt_family: dict(pred_counter)
                for gt_family, pred_counter in sorted(family_confusion.items())
            },
        },
        "skipped_reason_counts": dict(skipped_reason_counts),
        "examples": {
            "category_errors": [
                row for row in per_ticket_rows if not row["category_match"]
            ][:20],
            "family_errors": [
                row for row in per_ticket_rows if not row["family_match"]
            ][:20],
            "worst_issue_matches": sorted(
                per_ticket_rows,
                key=lambda row: row.get("issue_embedding_similarity", 0.0)
            )[:10],
            "worst_resolution_matches": sorted(
                [row for row in per_ticket_rows if row.get("ground_truth_resolution_summary")],
                key=lambda row: row.get("resolution_embedding_similarity", 0.0)
            )[:10],
        },
        "per_ticket_results_count": len(per_ticket_rows),
    }

    return report, per_ticket_rows


# ==============================
# Markdown report
# ==============================
def make_markdown_report(report: Dict[str, Any]) -> str:
    dataset = report["dataset"]
    extraction = report["extraction_metrics"]
    category = report["category_metrics"]
    family = report["family_metrics"]
    text_metrics = report["text_similarity_metrics"]
    confidence = report["confidence_metrics"]
    skipped = report["skipped_reason_counts"]
    distributions = report["category_distributions"]
    confusion = report["confusion_matrix"]
    examples = report["examples"]

    lines: List[str] = []

    lines.append("# Extraction Evaluation Report")
    lines.append("")

    lines.append("## Dataset")
    lines.append(f"- Total tickets: {dataset['total_tickets']}")
    lines.append(f"- Eligible tickets: {dataset['eligible_tickets']}")
    lines.append(f"- Extracted tickets: {dataset['extracted_tickets']}")
    lines.append(f"- Evaluation scope: {dataset['evaluation_scope']}")
    lines.append("")

    lines.append("## Extraction Metrics")
    lines.append(f"- Precision: {extraction['precision']}")
    lines.append(f"- Recall: {extraction['recall']}")
    lines.append(f"- F1: {extraction['f1']}")
    lines.append(f"- True positives: {extraction['true_positives']}")
    lines.append(f"- False positives: {extraction['false_positives']}")
    lines.append(f"- False negatives: {extraction['false_negatives']}")
    lines.append(f"- True negatives: {extraction['true_negatives']}")
    lines.append("")

    lines.append("## Category Metrics")
    lines.append(f"- Fine-grained evaluated predictions: {category['evaluated_predictions']}")
    lines.append(f"- Fine-grained correct predictions: {category['correct_predictions']}")
    lines.append(f"- Fine-grained accuracy: {category['accuracy']}")
    lines.append(f"- Family-level evaluated predictions: {family['evaluated_predictions']}")
    lines.append(f"- Family-level correct predictions: {family['correct_predictions']}")
    lines.append(f"- Family-level accuracy: {family['accuracy']}")
    lines.append("")

    lines.append("## Text Similarity Metrics")
    lines.append("### Issue Text (predicted query vs best-matching ground-truth title)")
    lines.append(f"- Average token F1: {text_metrics['issue']['avg_token_f1']}")
    lines.append(f"- Average sequence similarity: {text_metrics['issue']['avg_sequence_similarity']}")
    lines.append(f"- Average embedding similarity: {text_metrics['issue']['avg_embedding_similarity']}")
    lines.append("")
    lines.append("### Resolution Text (predicted positive vs ground-truth resolution summary)")
    lines.append(f"- Average token F1: {text_metrics['resolution']['avg_token_f1']}")
    lines.append(f"- Average sequence similarity: {text_metrics['resolution']['avg_sequence_similarity']}")
    lines.append(f"- Average embedding similarity: {text_metrics['resolution']['avg_embedding_similarity']}")
    lines.append("")

    lines.append("## Confidence")
    lines.append(f"- Average confidence: {confidence['avg_confidence']}")
    lines.append(f"- Count: {confidence['count']}")
    lines.append("")

    lines.append("## Skipped Ticket Reasons")
    if skipped:
        for reason, count in sorted(skipped.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Category Distributions")
    lines.append("### Ground Truth Categories")
    for k, v in distributions["ground_truth_categories"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### Predicted Categories")
    for k, v in distributions["predicted_categories"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### Ground Truth Families")
    for k, v in distributions["ground_truth_families"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### Predicted Families")
    for k, v in distributions["predicted_families"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Fine-Grained Category Confusion Matrix")
    fine_confusion = confusion["fine_grained"]
    if fine_confusion:
        for gt_category, pred_counts in fine_confusion.items():
            pretty_preds = ", ".join(f"{pred}:{count}" for pred, count in sorted(pred_counts.items()))
            lines.append(f"- {gt_category} -> {pretty_preds}")
    else:
        lines.append("- No category comparisons available")
    lines.append("")

    lines.append("## Family-Level Confusion Matrix")
    family_confusion = confusion["family_level"]
    if family_confusion:
        for gt_family, pred_counts in family_confusion.items():
            pretty_preds = ", ".join(f"{pred}:{count}" for pred, count in sorted(pred_counts.items()))
            lines.append(f"- {gt_family} -> {pretty_preds}")
    else:
        lines.append("- No family comparisons available")
    lines.append("")

    lines.append("## Example Fine-Grained Category Errors")
    if examples["category_errors"]:
        for row in examples["category_errors"][:10]:
            lines.append(
                f"- {row['ticket_id']}: gt={row['ground_truth_primary_issue_category']}, "
                f"pred={row['predicted_issue_category']}, "
                f"gt_family={row['ground_truth_category_family']}, "
                f"pred_family={row['predicted_category_family']}, "
                f"conf={row['confidence']}"
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Example Family Errors")
    if examples["family_errors"]:
        for row in examples["family_errors"][:10]:
            lines.append(
                f"- {row['ticket_id']}: gt_family={row['ground_truth_category_family']}, "
                f"pred_family={row['predicted_category_family']}, "
                f"gt={row['ground_truth_primary_issue_category']}, "
                f"pred={row['predicted_issue_category']}"
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Worst Issue Matches")
    if examples["worst_issue_matches"]:
        for row in examples["worst_issue_matches"][:5]:
            lines.append(
                f"- {row['ticket_id']}: "
                f"issue_emb={row['issue_embedding_similarity']}, "
                f"pred='{row['predicted_issue_text']}', "
                f"best_gt='{row['best_matching_ground_truth_title']}'"
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Worst Resolution Matches")
    if examples["worst_resolution_matches"]:
        for row in examples["worst_resolution_matches"][:5]:
            lines.append(
                f"- {row['ticket_id']}: "
                f"res_emb={row['resolution_embedding_similarity']}, "
                f"pred='{row['predicted_resolution_summary']}', "
                f"gt='{row['ground_truth_resolution_summary']}'"
            )
    else:
        lines.append("- None")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


# ==============================
# Main
# ==============================
def main() -> int:
    args = parse_args()

    tickets_path = Path(args.tickets)
    extracted_path = Path(args.extracted)
    skipped_path = Path(args.skipped) if args.skipped else None
    output_dir = Path(args.output_dir)

    ensure_dir(output_dir)

    if not tickets_path.exists():
        print(f"Tickets file not found: {tickets_path}")
        return 1
    if not extracted_path.exists():
        print(f"Extracted pairs file not found: {extracted_path}")
        return 1
    if skipped_path and not skipped_path.exists():
        print(f"Skipped file not found: {skipped_path}")
        return 1

    print(f"Loading embedding model: {args.embedding_model}")
    embedding_model = SentenceTransformer(args.embedding_model)

    tickets = load_ground_truth_tickets(tickets_path)
    extracted = load_extracted_pairs(extracted_path)
    skipped = load_skipped_rows(skipped_path)

    report, per_ticket_rows = evaluate(
        tickets=tickets,
        extracted=extracted,
        skipped=skipped,
        only_resolved=args.only_resolved,
        embedding_model=embedding_model,
    )

    json_report_path = output_dir / "evaluation_report.json"
    md_report_path = output_dir / "evaluation_report.md"
    per_ticket_path = output_dir / "per_ticket_results.jsonl"

    write_json(json_report_path, report)
    write_markdown(md_report_path, make_markdown_report(report))
    write_jsonl(per_ticket_path, per_ticket_rows)

    print(f"Wrote: {json_report_path}")
    print(f"Wrote: {md_report_path}")
    print(f"Wrote: {per_ticket_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
