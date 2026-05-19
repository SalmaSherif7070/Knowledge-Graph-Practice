"""
scripts/eval.py
Evaluates the conflict detection system against labeled ground truth.

Usage (run from repo root):
    python -m scripts.eval
    python -m scripts.eval --csv data/rules.csv --ground-truth data/ground_truth.csv
    python -m scripts.eval --rescan
"""

import argparse
import csv
import sys
import os
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv(".env", override=True)

from app.graph.neo4j_client import get_driver, setup_schema, ingest_rules
from app.conflict.detector import check_all_conflicts


def load_ground_truth(path: str) -> dict[str, dict]:
    gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = "-".join(sorted([row["rule_id_a"].strip(), row["rule_id_b"].strip()]))
            gt[key] = {
                "is_conflict":   row["is_conflict"].strip().lower() == "true",
                "conflict_type": row.get("conflict_type", "").strip(),
                "notes":         row.get("notes", "").strip(),
            }
    return gt


def evaluate(predicted_pairs: list, ground_truth: dict) -> dict:
    predicted_keys = {"-".join(sorted([p.rule_id_a, p.rule_id_b])) for p in predicted_pairs}
    tp = fp = fn = tn = 0
    results = []
    type_breakdown = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})

    for key, label in ground_truth.items():
        predicted = key in predicted_keys
        actual    = label["is_conflict"]
        ctype     = label["conflict_type"]

        if predicted and actual:       outcome = "TP"; tp += 1; type_breakdown[ctype]["tp"] += 1
        elif predicted and not actual: outcome = "FP"; fp += 1; type_breakdown[ctype]["fp"] += 1
        elif not predicted and actual: outcome = "FN"; fn += 1; type_breakdown[ctype]["fn"] += 1
        else:                          outcome = "TN"; tn += 1

        results.append({"pair": key, "actual": actual, "predicted": predicted,
                         "outcome": outcome, "conflict_type": ctype, "notes": label["notes"]})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(ground_truth) if ground_truth else 0.0

    return {
        "metrics":        {"precision": round(precision, 4), "recall": round(recall, 4),
                           "f1": round(f1, 4), "accuracy": round(accuracy, 4)},
        "confusion":      {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "type_breakdown": dict(type_breakdown),
        "results":        results,
    }


def print_report(eval_result: dict, total_rules: int):
    m, c, tb, rs = (eval_result[k] for k in ("metrics", "confusion", "type_breakdown", "results"))
    print("\n" + "=" * 60)
    print("  CONFLICT DETECTION — EVALUATION REPORT")
    print("=" * 60)
    print(f"  Rules in DB   : {total_rules}")
    print(f"  Labeled pairs : {len(rs)}\n")
    print(f"  Precision     : {m['precision']:.2%}")
    print(f"  Recall        : {m['recall']:.2%}")
    print(f"  F1 Score      : {m['f1']:.2%}")
    print(f"  Accuracy      : {m['accuracy']:.2%}\n")
    print("  Confusion Matrix:")
    print(f"    TP (caught real conflicts)      : {c['tp']}")
    print(f"    FN (missed real conflicts)      : {c['fn']}")
    print(f"    FP (wrongly flagged)            : {c['fp']}")
    print(f"    TN (correctly ignored)          : {c['tn']}\n")
    if tb:
        print("  Breakdown by conflict type:")
        for ctype, counts in sorted(tb.items()):
            print(f"    {ctype:<25} caught={counts.get('tp',0)}  missed={counts.get('fn',0)}  fp={counts.get('fp',0)}")
        print()
    misses = [r for r in rs if r["outcome"] in ("FN", "FP")]
    if misses:
        print("  Errors:")
        for r in misses:
            tag = "❌ MISSED" if r["outcome"] == "FN" else "⚠️  FALSE POS"
            print(f"    {tag}  {r['pair']:<20}  [{r['conflict_type']}]  {r['notes']}")
    else:
        print("  ✅ Perfect score!")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate conflict detection accuracy.")
    parser.add_argument("--csv",          default="data/rules.csv")
    parser.add_argument("--ground-truth", default="data/ground_truth.csv")
    parser.add_argument("--rescan",       action="store_true")
    parser.add_argument("--skip-ingest",  action="store_true")
    args = parser.parse_args()

    for path in [args.csv, args.ground_truth]:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            sys.exit(1)

    driver = get_driver()
    setup_schema(driver)
    if not args.skip_ingest:
        count = ingest_rules(args.csv, driver)
        print(f"→ {count} rules loaded.")
    driver.close()

    response = check_all_conflicts(rescan=args.rescan)
    gt = load_ground_truth(args.ground_truth)
    print_report(evaluate(response.conflict_pairs, gt), total_rules=response.total_rules)


if __name__ == "__main__":
    main()
