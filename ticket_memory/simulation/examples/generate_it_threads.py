from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ticket_memory.simulation.core.engine import SimulationEngine
from ticket_memory.simulation.domains.it_support.rules import build_it_support_domain
from ticket_memory.simulation.exporters.raw_threads import export_raw_threads
from ticket_memory.simulation.exporters.retrieval_pairs import export_retrieval_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic IT support conversation threads.")
    parser.add_argument("--count", type=int, default=200, help="Number of tickets to generate")
    parser.add_argument("--output", default="simulated_tdx_tickets.jsonl", help="Raw thread JSONL output path")
    parser.add_argument("--pairs-output", help="Optional retrieval-pair JSONL output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    engine = SimulationEngine(build_it_support_domain(), rng)
    results = engine.generate_tickets(args.count)
    output_path = Path(args.output)
    export_raw_threads(results, output_path)
    if args.pairs_output:
        export_retrieval_pairs(results, Path(args.pairs_output))

    resolved = sum(1 for result in results if result.resolution_state == "resolved")
    partial = sum(1 for result in results if result.resolution_state == "partial")
    unresolved = sum(1 for result in results if result.resolution_state == "unresolved")
    mixed = sum(1 for result in results if result.scenario.secondary_issue_variant is not None)

    print(f"Wrote {len(results)} tickets to {output_path}")
    print(f"Resolved tickets:          {resolved}")
    print(f"Partially resolved tickets:{partial}")
    print(f"Unresolved tickets:        {unresolved}")
    print(f"Mixed-issue tickets:       {mixed}")
    print(f"IT issue families:         {len({result.scenario.issue_variant.family for result in results})}")


if __name__ == "__main__":
    main()

