from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ------------------------------
# Data structures
# ------------------------------
@dataclass
class GroundTruthTicket:
    ticket_id: str
    status: str
    title: str
    ground_truth: Dict[str, Any]


@dataclass
class ExtractedPair:
    ticket_id: str
    query: str
    positive: str
    issue_category: str
    confidence: float
    metadata: Dict[str, Any]


# ------------------------------
# CLI
# ------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate extracted ticket pairs against synthetic ground-truth tickets."
    )
    parser.add_argument(
        "--tickets",
        required=True,
        help="Path to original synthetic tickets JSONL (with ground_truth)",
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
        "--only-resolved",
        action="store_true",
        help="Evaluate only tickets with ground_truth.resolution_state == 'resolved'",
    )
    return parser.parse_args()


# ------------------------------
# File IO
# ------------------------------
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


# ------------------------------
# Normalization helpers
# ------------------------------
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
    if not text:
        return []
    return text.split()


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
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


# ------------------------------
# Loading domain objects
# ------------------------------
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

        tickets[ticket_id] = GroundTruthTicket(
            ticket_id=ticket_id,
            status=str(row.get("status") or ""),
            title=str(row.get("title") or ""),
            ground_truth=gt,
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
            query=str(row.get("query") or ""),
            positive=str(row.get("positive") or ""),
            issue_category=str(row.get("issue_category") or ""),
            confidence=confidence,
            metadata=metadata,
        )
    return extracted


def load_skipped_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path:
        return []
    if not path.exists():
        return []
    return read_jsonl(path)


# ------------------------------
# Evaluation logic
# ------------------------------
def should_ticket_be_extractable(gt: Dict[str, Any], only_resolved: bool) -> bool:
    resolution_state = str(gt.get("resolution_state") or "").strip().lower()
    should_extract = bool(gt.get("should_extract"))

    if only_resolved:
        return resolution_state == "resolved"

    if resolution_state:
        return resolution_state == "resolved"
    return should_extract


def evaluate(
    tickets: Dict[str, GroundTruthTicket],
    extracted: Dict[str, ExtractedPair],
    skipped: List[Dict[str, Any]],
    only_resolved: bool,
) -> Dict[str, Any]:
    eligible_ticket_ids: List[str] = []
    all_ticket_ids: List[str] = list(tickets.keys())

    for ticket_id, ticket in tickets.items():
        if should_ticket_be_extractable(ticket.ground_truth, only_resolved=only_resolved):
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

    category_total = 0
    category_correct = 0
    confidence_values: List[float] = []
    token_f1_values: List[float] = []
    sequence_similarity_values: List[float] = []
    category_confusion: Dict[str, Counter] = defaultdict(Counter)
    per_ticket_rows: List[Dict[str, Any]] = []

    for ticket_id in sorted(eligible_set & extracted_set):
        gt_ticket = tickets[ticket_id]
        pred = extracted[ticket_id]

        gt_category = str(gt_ticket.ground_truth.get("primary_issue_category") or "").strip()
        gt_resolution = str(gt_ticket.ground_truth.get("resolution_summary") or "").strip()

        pred_category = pred.issue_category.strip()
        pred_resolution = pred.positive.strip()

        category_match = gt_category == pred_category
        if gt_category:
            category_total += 1
            if category_match:
                category_correct += 1
            category_confusion[gt_category][pred_category or "<empty>"] += 1

        res_token_f1 = token_f1(pred_resolution, gt_resolution)
        res_similarity = string_similarity(pred_resolution, gt_resolution)

        token_f1_values.append(res_token_f1)
        sequence_similarity_values.append(res_similarity)
        confidence_values.append(pred.confidence)

        per_ticket_rows.append(
            {
                "ticket_id": ticket_id,
                "title": gt_ticket.title,
                "ground_truth_resolution_state": gt_ticket.ground_truth.get("resolution_state"),
                "ground_truth_primary_issue_category": gt_category,
                "predicted_issue_category": pred_category,
                "category_match": category_match,
                "confidence": pred.confidence,
                "ground_truth_resolution_summary": gt_resolution,
                "predicted_resolution_summary": pred_resolution,
                "resolution_token_f1": round(res_token_f1, 4),
                "resolution_sequence_similarity": round(res_similarity, 4),
            }
        )

    skipped_reason_counts = Counter()
    for row in skipped:
        reason = str(row.get("reason") or "unknown")
        skipped_reason_counts[reason] += 1

    category_accuracy = safe_div(category_correct, category_total)

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
            "evaluated_predictions": category_total,
            "correct_predictions": category_correct,
            "accuracy": round(category_accuracy, 4),
        },
        "resolution_metrics": {
            "avg_token_f1": round(safe_mean(token_f1_values), 4),
            "avg_sequence_similarity": round(safe_mean(sequence_similarity_values), 4),
        },
        "confidence_metrics": {
            "avg_confidence": round(safe_mean(confidence_values), 4),
            "count": len(confidence_values),
        },
        "confusion_matrix": {
            gt_cat: dict(pred_counter)
            for gt_cat, pred_counter in sorted(category_confusion.items())
        },
        "skipped_reason_counts": dict(skipped_reason_counts),
        "examples": {
            "best_resolution_matches": sorted(
                per_ticket_rows,
                key=lambda row: row["resolution_token_f1"],
                reverse=True,
            )[:10],
            "worst_resolution_matches": sorted(
                per_ticket_rows,
                key=lambda row: row["resolution_token_f1"],
            )[:10],
            "category_errors": [
                row for row in per_ticket_rows if not row["category_match"]
            ][:20],
        },
        "per_ticket_results_count": len(per_ticket_rows),
    }

    return report, per_ticket_rows


# ------------------------------
# Markdown formatting
# ------------------------------
def make_markdown_report(report: Dict[str, Any]) -> str:
    dataset = report["dataset"]
    extraction = report["extraction_metrics"]
    category = report["category_metrics"]
    resolution = report["resolution_metrics"]
    confidence = report["confidence_metrics"]
    skipped = report["skipped_reason_counts"]
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
    lines.append(f"- Evaluated predictions: {category['evaluated_predictions']}")
    lines.append(f"- Correct predictions: {category['correct_predictions']}")
    lines.append(f"- Accuracy: {category['accuracy']}")
    lines.append("")

    lines.append("## Resolution Similarity Metrics")
    lines.append(f"- Average token F1: {resolution['avg_token_f1']}")
    lines.append(f"- Average sequence similarity: {resolution['avg_sequence_similarity']}")
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

    lines.append("## Category Confusion Matrix")
    if confusion:
        for gt_category, pred_counts in confusion.items():
            pretty_preds = ", ".join(f"{pred}:{count}" for pred, count in sorted(pred_counts.items()))
            lines.append(f"- {gt_category} -> {pretty_preds}")
    else:
        lines.append("- No category comparisons available")
    lines.append("")

    lines.append("## Example Category Errors")
    if examples["category_errors"]:
        for row in examples["category_errors"][:10]:
            lines.append(
                f"- {row['ticket_id']}: gt={row['ground_truth_primary_issue_category']}, "
                f"pred={row['predicted_issue_category']}, conf={row['confidence']}"
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Worst Resolution Matches")
    if examples["worst_resolution_matches"]:
        for row in examples["worst_resolution_matches"][:5]:
            lines.append(
                f"- {row['ticket_id']}: token_f1={row['resolution_token_f1']}, "
                f"seq_sim={row['resolution_sequence_similarity']}"
            )
    else:
        lines.append("- None")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


# ------------------------------
# Main
# ------------------------------
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

    tickets = load_ground_truth_tickets(tickets_path)
    extracted = load_extracted_pairs(extracted_path)
    skipped = load_skipped_rows(skipped_path)

    report, per_ticket_rows = evaluate(
        tickets=tickets,
        extracted=extracted,
        skipped=skipped,
        only_resolved=args.only_resolved,
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