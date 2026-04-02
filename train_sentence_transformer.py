#!/usr/bin/env python3
"""
Train a SentenceTransformer model from generated ticket retrieval data.

Supported inputs
----------------
1) Pair classification data from `train_pairs.csv`
   Columns:
     - sentence1
     - sentence2
     - label   (float, typically 1.0 or 0.0)

2) Triplet data from `triplets.jsonl`
   Fields:
     - anchor
     - positive
     - negative

Examples
--------
Train from pair data:
  python train_sentence_transformer.py ^
    --input out_pairs_test_20/train_pairs.csv ^
    --output-dir models/ticket-pairs

Train from triplet data:
  python train_sentence_transformer.py ^
    --input out_pairs_test_20/triplets.jsonl ^
    --task triplet ^
    --output-dir models/ticket-triplets
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple, TypeVar

from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, TripletEvaluator
from torch.utils.data import DataLoader


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_SEED = 42
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SentenceTransformer model from generated ticket data.")
    parser.add_argument("--input", required=True, help="Path to train_pairs.csv or triplets.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory where the trained model will be saved")
    parser.add_argument(
        "--task",
        choices=["auto", "pairs", "triplet"],
        default="auto",
        help="Training objective to use. 'auto' infers from the input filename suffix.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base SentenceTransformer model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Optimizer learning rate")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help="Fraction of examples reserved for validation; use 0 to disable validation",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    return parser.parse_args()


def infer_task(input_path: Path, requested_task: str) -> str:
    if requested_task != "auto":
        return requested_task
    if input_path.name.endswith("train_pairs.csv"):
        return "pairs"
    if input_path.name.endswith("triplets.jsonl"):
        return "triplet"
    if input_path.suffix.lower() == ".csv":
        return "pairs"
    if input_path.suffix.lower() == ".jsonl":
        return "triplet"
    raise ValueError("Could not infer task from input path. Pass --task pairs or --task triplet.")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_examples(items: Sequence[T], validation_split: float, rng: random.Random) -> Tuple[List[T], List[T]]:
    shuffled = list(items)
    rng.shuffle(shuffled)

    if not shuffled or validation_split <= 0:
        return shuffled, []

    validation_count = int(len(shuffled) * validation_split)
    if validation_count == 0 and len(shuffled) > 1:
        validation_count = 1
    if validation_count >= len(shuffled):
        validation_count = len(shuffled) - 1

    if validation_count <= 0:
        return shuffled, []

    validation = shuffled[:validation_count]
    train = shuffled[validation_count:]
    return train, validation


def load_pair_examples(path: Path) -> List[InputExample]:
    examples: List[InputExample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sentence1", "sentence2", "label"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")

        for row_no, row in enumerate(reader, start=2):
            sentence1 = str(row.get("sentence1") or "").strip()
            sentence2 = str(row.get("sentence2") or "").strip()
            label_text = str(row.get("label") or "").strip()
            if not sentence1 or not sentence2:
                continue
            try:
                label = float(label_text)
            except ValueError as exc:
                raise ValueError(f"Invalid label at CSV row {row_no}: {label_text}") from exc
            examples.append(InputExample(texts=[sentence1, sentence2], label=label))

    return examples


def load_triplet_examples(path: Path) -> List[InputExample]:
    examples: List[InputExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc

            anchor = str(obj.get("anchor") or "").strip()
            positive = str(obj.get("positive") or "").strip()
            negative = str(obj.get("negative") or "").strip()
            if not anchor or not positive or not negative:
                continue
            examples.append(InputExample(texts=[anchor, positive, negative]))

    return examples


def build_pair_evaluator(examples: Sequence[InputExample], name: str) -> EmbeddingSimilarityEvaluator | None:
    if not examples:
        return None
    return EmbeddingSimilarityEvaluator.from_input_examples(list(examples), name=name)


def build_triplet_evaluator(examples: Sequence[InputExample], name: str) -> TripletEvaluator | None:
    if not examples:
        return None
    anchors = [example.texts[0] for example in examples]
    positives = [example.texts[1] for example in examples]
    negatives = [example.texts[2] for example in examples]
    return TripletEvaluator(anchors=anchors, positives=positives, negatives=negatives, name=name)


def train_pairs(
    model: SentenceTransformer,
    examples: Sequence[InputExample],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_split: float,
    rng: random.Random,
) -> None:
    train_examples, validation_examples = split_examples(examples, validation_split, rng)
    if not train_examples:
        raise ValueError("No usable pair examples were found in the input file.")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    evaluator = build_pair_evaluator(validation_examples, name="pairs-validation")
    warmup_steps = math.ceil(len(train_dataloader) * epochs * DEFAULT_WARMUP_RATIO)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        save_best_model=bool(evaluator),
        show_progress_bar=True,
    )


def train_triplets(
    model: SentenceTransformer,
    examples: Sequence[InputExample],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_split: float,
    rng: random.Random,
) -> None:
    train_examples, validation_examples = split_examples(examples, validation_split, rng)
    if not train_examples:
        raise ValueError("No usable triplet examples were found in the input file.")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.TripletLoss(model=model)
    evaluator = build_triplet_evaluator(validation_examples, name="triplets-validation")
    warmup_steps = math.ceil(len(train_dataloader) * epochs * DEFAULT_WARMUP_RATIO)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        save_best_model=bool(evaluator),
        show_progress_bar=True,
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1
    if args.epochs < 1:
        print("--epochs must be at least 1", file=sys.stderr)
        return 1
    if args.batch_size < 1:
        print("--batch-size must be at least 1", file=sys.stderr)
        return 1
    if not 0 <= args.validation_split < 1:
        print("--validation-split must be between 0 and 1", file=sys.stderr)
        return 1

    task = infer_task(input_path, args.task)
    rng = random.Random(args.seed)
    model = SentenceTransformer(args.model_name)

    print(f"Loading {task} data from {input_path}...")
    if task == "pairs":
        examples = load_pair_examples(input_path)
        print(f"Loaded pair examples: {len(examples)}")
        train_pairs(
            model=model,
            examples=examples,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split,
            rng=rng,
        )
    else:
        examples = load_triplet_examples(input_path)
        print(f"Loaded triplet examples: {len(examples)}")
        train_triplets(
            model=model,
            examples=examples,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split,
            rng=rng,
        )

    metadata = {
        "input": str(input_path),
        "task": task,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_split": args.validation_split,
        "seed": args.seed,
    }
    metadata_path = output_dir / "training_run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model to {output_dir}")
    print(f"Wrote training metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
