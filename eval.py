"""
eval.py
Evaluates the conflict detection system against a labeled ground truth.

Usage:
    python eval.py
    python eval.py --csv rules.csv --ground-truth ground_truth.csv
    python eval.py --rescan          # force re-check all pairs in DB

Metrics reported:
    - Precision   : of pairs flagged as conflicts, how many are correct?
    - Recall      : of actual conflicts, how many did we catch?
    - F1 Score    : harmonic mean of precision and recall
    - Accuracy    : overall correct predictions across all labeled pairs
    - Confusion matrix breakdown
    - Per-conflict-type breakdown (direct / exception_negates / near_miss)
"""

import argparse
import csv
import sys
import os
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv(".env", override=True)

from kg_builder import get_driver, setup_schema, ingest_rules
from conflict import check_all_conflicts


# ──────────────────────────────────────────────
# Load ground truth
# ──────────────────────────────────────────────

def load_ground_truth(path: str) -> dict[str, dict]:
    """
    Returns a dict keyed by canonical pair key "RA-RB" (sorted).
    Value: {"is_conflict": bool, "conflict_type": str, "notes": str}
    """
    gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = "-".join(sorted([row["rule_id_a"].strip(), row["rule_id_b"].strip()]))
            gt[key] = {
                "is_conflict":    row["is_conflict"].strip().lower() == "true",
                "conflict_type":  row.get("conflict_type", "").strip(),
                "notes":          row.get("notes", "").strip(),
            }
    return gt


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate(predicted_pairs: list, ground_truth: dict) -> dict:
    """
    Compare system output against ground truth.

    predicted_pairs : list of ConflictPair objects returned by check_all_conflicts
    ground_truth    : dict from load_ground_truth()
    """
    predicted_keys = {
        "-".join(sorted([p.rule_id_a, p.rule_id_b]))
        for p in predicted_pairs
    }

    # Only evaluate pairs that appear in ground truth
    labeled_pairs = set(ground_truth.keys())

    tp, fp, fn, tn = 0, 0, 0, 0

    results = []
    type_breakdown = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})

    for key, label in ground_truth.items():
        predicted   = key in predicted_keys
        actual      = label["is_conflict"]
        ctype       = label["conflict_type"]

        if predicted and actual:
            outcome = "TP"
            tp += 1
            type_breakdown[ctype]["tp"] += 1
        elif predicted and not actual:
            outcome = "FP"
            fp += 1
            type_breakdown[ctype]["fp"] += 1
        elif not predicted and actual:
            outcome = "FN"
            fn += 1
            type_breakdown[ctype]["fn"] += 1
        else:
            outcome = "TN"
            tn += 1

        results.append({
            "pair":          key,
            "actual":        actual,
            "predicted":     predicted,
            "outcome":       outcome,
            "conflict_type": ctype,
            "notes":         label["notes"],
        })

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(ground_truth) if ground_truth else 0.0

    return {
        "metrics": {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "accuracy":  round(accuracy, 4),
        },
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "type_breakdown": dict(type_breakdown),
        "results": results,
    }


# ──────────────────────────────────────────────
# Pretty print
# ──────────────────────────────────────────────

def print_report(eval_result: dict, total_rules: int):
    m  = eval_result["metrics"]
    c  = eval_result["confusion"]
    tb = eval_result["type_breakdown"]
    rs = eval_result["results"]

    print("\n" + "=" * 60)
    print("  CONFLICT DETECTION — EVALUATION REPORT")
    print("=" * 60)
    print(f"  Rules in DB     : {total_rules}")
    print(f"  Labeled pairs   : {len(rs)}")
    print()
    print(f"  Precision       : {m['precision']:.2%}")
    print(f"  Recall          : {m['recall']:.2%}")
    print(f"  F1 Score        : {m['f1']:.2%}")
    print(f"  Accuracy        : {m['accuracy']:.2%}")
    print()
    print("  Confusion Matrix:")
    print(f"    True Positives  (caught real conflicts)     : {c['tp']}")
    print(f"    False Negatives (missed real conflicts)     : {c['fn']}")
    print(f"    False Positives (wrongly flagged as conflict): {c['fp']}")
    print(f"    True Negatives  (correctly ignored)         : {c['tn']}")
    print()

    if tb:
        print("  Breakdown by conflict type:")
        for ctype, counts in sorted(tb.items()):
            caught = counts.get("tp", 0)
            missed = counts.get("fn", 0)
            false_ = counts.get("fp", 0)
            print(f"    {ctype:<25} caught={caught}  missed={missed}  false_positive={false_}")
        print()

    # List misses
    misses = [r for r in rs if r["outcome"] in ("FN", "FP")]
    if misses:
        print("  Errors:")
        for r in misses:
            tag = "❌ MISSED" if r["outcome"] == "FN" else "⚠️  FALSE POS"
            print(f"    {tag}  {r['pair']:<20}  [{r['conflict_type']}]  {r['notes']}")
    else:
        print("  ✅ No errors — perfect score!")

    print("=" * 60 + "\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate conflict detection accuracy.")
    parser.add_argument("--csv",          default="rules.csv",         help="Path to rules CSV")
    parser.add_argument("--ground-truth", default="ground_truth.csv",  help="Path to ground truth CSV")
    parser.add_argument("--rescan",       action="store_true",          help="Force re-check all pairs")
    parser.add_argument("--skip-ingest",  action="store_true",          help="Skip CSV ingest (rules already in DB)")
    args = parser.parse_args()

    # Validate files
    for path in [args.csv, args.ground_truth]:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            sys.exit(1)

    # Setup DB
    print("Connecting to Neo4j…")
    driver = get_driver()
    setup_schema(driver)

    if not args.skip_ingest:
        print(f"Ingesting rules from {args.csv}…")
        count = ingest_rules(args.csv, driver)
        print(f"  → {count} rules loaded.")
    
    driver.close()

    # Run conflict detection
    print(f"Running conflict detection (rescan={args.rescan})…")
    response = check_all_conflicts(rescan=args.rescan)
    print(f"  → {len(response.conflict_pairs)} conflict pair(s) detected.")

    # Load ground truth and evaluate
    print(f"Loading ground truth from {args.ground_truth}…")
    gt = load_ground_truth(args.ground_truth)

    result = evaluate(response.conflict_pairs, gt)
    print_report(result, total_rules=response.total_rules)


if __name__ == "__main__":
    main()