"""
conflict.py
Orchestrates the two conflict-detection workflows:
  1. check_all_conflicts  — scan every rule pair in the DB
  2. check_new_rule       — check one rule against all DB rules
"""

import os
from itertools import combinations
from typing import Optional

from embeddings import cosine_similarity
from llm import check_conflict_with_llm
from kg_builder import (
    get_all_rules,
    get_driver,
    save_conflict,
    find_similar_rules,
    add_rule_to_graph,
    get_all_conflicts,
    get_conflicts_for_rule,
)
from models import ConflictPair, CheckAllConflictsResponse, NewRuleConflictResponse

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
TOP_K_CANDIDATES     = int(os.getenv("TOP_K_CANDIDATES", "20"))


# ──────────────────────────────────────────────
# Endpoint 1: Check all rules against each other
# ──────────────────────────────────────────────

def check_all_conflicts(rescan: bool = False) -> CheckAllConflictsResponse:
    """
    1. Load all Rule nodes + their embeddings from Neo4j.
    2. For every pair, compute cosine similarity.
    3. A pair is a candidate if:
         - similarity >= SIMILARITY_THRESHOLD, OR
         - both rules share the same non-empty category
         (same-category check catches opposite-phrased conflicts with low similarity)
    4. Send candidate pairs to Gemini for conflict analysis.
    5. Persist confirmed conflicts as CONFLICTS_WITH edges.
    6. Return all conflict pairs.

    If rescan=False (default), skip pairs that already have a
    CONFLICTS_WITH relationship in the graph.
    """
    driver = get_driver()

    rules = get_all_rules(driver)
    if len(rules) < 2:
        driver.close()
        return CheckAllConflictsResponse(
            total_rules=len(rules),
            conflict_pairs=[],
            message="Not enough rules in the database to check for conflicts.",
        )

    emb_map      = {r["rule_id"]: r["embedding"] for r in rules if r.get("embedding")}
    category_map = {r["rule_id"]: (r.get("category") or "").strip().lower() for r in rules}

    existing_conflicts: set[str] = set()
    if not rescan:
        for c in get_all_conflicts(driver):
            key = "-".join(sorted([c["rule_id_a"], c["rule_id_b"]]))
            existing_conflicts.add(key)

    confirmed_conflicts: list[ConflictPair] = []

    pairs = list(combinations(rules, 2))
    print(f"Checking {len(pairs)} total pairs across {len(rules)} rules…")

    for rule_a, rule_b in pairs:
        id_a, id_b = rule_a["rule_id"], rule_b["rule_id"]
        pair_key   = "-".join(sorted([id_a, id_b]))

        if pair_key in existing_conflicts:
            continue

        emb_a = emb_map.get(id_a)
        emb_b = emb_map.get(id_b)
        if emb_a is None or emb_b is None:
            continue

        sim = cosine_similarity(emb_a, emb_b)

        cat_a = category_map.get(id_a, "")
        cat_b = category_map.get(id_b, "")
        same_category = bool(cat_a and cat_a == cat_b)

        if sim < SIMILARITY_THRESHOLD and not same_category:
            continue  # too different in both dimensions — skip

        print(f"  → Checking {id_a} vs {id_b} (sim={sim:.4f}, same_category={same_category})")

        conflicts_found, explanation = check_conflict_with_llm(
            id_a, rule_a["rule_text"],
            id_b, rule_b["rule_text"],
        )

        if conflicts_found:
            save_conflict(id_a, id_b, explanation, sim, driver)
            confirmed_conflicts.append(ConflictPair(
                rule_id_a=id_a,
                rule_text_a=rule_a["rule_text"],
                rule_id_b=id_b,
                rule_text_b=rule_b["rule_text"],
                conflict_explanation=explanation,
                similarity_score=round(sim, 4),
            ))

    # Include previously stored conflicts when rescan=False
    if not rescan:
        existing_keys = {
            "-".join(sorted([c.rule_id_a, c.rule_id_b]))
            for c in confirmed_conflicts
        }
        for stored in get_all_conflicts(driver):
            pair_key = "-".join(sorted([stored["rule_id_a"], stored["rule_id_b"]]))
            if pair_key not in existing_keys:
                confirmed_conflicts.append(ConflictPair(
                    rule_id_a=stored["rule_id_a"],
                    rule_text_a=stored["rule_text_a"],
                    rule_id_b=stored["rule_id_b"],
                    rule_text_b=stored["rule_text_b"],
                    conflict_explanation=stored["conflict_explanation"],
                    similarity_score=round(stored["similarity_score"] or 0.0, 4),
                ))

    driver.close()

    return CheckAllConflictsResponse(
        total_rules=len(rules),
        conflict_pairs=confirmed_conflicts,
        message=f"Found {len(confirmed_conflicts)} conflict pair(s) among {len(rules)} rules.",
    )


# ──────────────────────────────────────────────
# Endpoint 2: Check a new rule against the DB
# ──────────────────────────────────────────────

def check_new_rule(
    rule_id: str,
    rule_text: str,
    category: Optional[str],
    description: Optional[str],
    save_to_db: bool,
) -> NewRuleConflictResponse:
    """
    1. Compute embedding for the new rule.
    2. Use Neo4j vector index to find top-K most similar existing rules.
    3. Also include any rules sharing the same category (catches opposite-phrased conflicts).
    4. Send each candidate pair to Gemini for conflict analysis.
    5. Optionally persist the new rule + its conflicts to the graph.
    """
    driver = get_driver()

    new_rule = {
        "rule_id":     rule_id,
        "rule_text":   rule_text,
        "category":    category or "",
        "description": description or "",
    }

    if save_to_db:
        query_embedding = add_rule_to_graph(new_rule, driver)
    else:
        from embeddings import get_single_embedding
        query_embedding = get_single_embedding(rule_text)

    # Vector similarity candidates
    vector_candidates = find_similar_rules(
        query_embedding=query_embedding,
        top_k=TOP_K_CANDIDATES,
        exclude_rule_id=rule_id if save_to_db else None,
        driver=driver,
    )

    # Same-category candidates (deduped)
    seen_ids: set[str] = {c["rule_id"] for c in vector_candidates}
    category_candidates: list[dict] = []

    if category:
        norm_cat  = category.strip().lower()
        all_rules = get_all_rules(driver)
        for r in all_rules:
            if r["rule_id"] == rule_id:
                continue
            if (r.get("category") or "").strip().lower() == norm_cat and r["rule_id"] not in seen_ids:
                category_candidates.append({
                    "rule_id":   r["rule_id"],
                    "rule_text": r["rule_text"],
                    "score":     0.0,
                })
                seen_ids.add(r["rule_id"])

    # Filter vector candidates by threshold; keep all category candidates
    filtered_vector = [c for c in vector_candidates if c["score"] >= SIMILARITY_THRESHOLD]
    all_candidates  = filtered_vector + category_candidates

    print(
        f"Checking '{rule_id}' against {len(all_candidates)} candidate(s) "
        f"({len(filtered_vector)} by similarity, {len(category_candidates)} by category)…"
    )

    conflicts: list[ConflictPair] = []

    for candidate in all_candidates:
        conflicts_found, explanation = check_conflict_with_llm(
            rule_id, rule_text,
            candidate["rule_id"], candidate["rule_text"],
        )

        if conflicts_found:
            sim = candidate.get("score", 0.0)
            if save_to_db:
                save_conflict(rule_id, candidate["rule_id"], explanation, sim, driver)

            conflicts.append(ConflictPair(
                rule_id_a=rule_id,
                rule_text_a=rule_text,
                rule_id_b=candidate["rule_id"],
                rule_text_b=candidate["rule_text"],
                conflict_explanation=explanation,
                similarity_score=round(sim, 4),
            ))

    driver.close()

    status = "saved to database" if save_to_db else "not saved (pass save_to_db=true to persist)"
    return NewRuleConflictResponse(
        rule_id=rule_id,
        rule_text=rule_text,
        conflicts=conflicts,
        message=(
            f"Rule '{rule_id}' {status}. "
            f"Checked against {len(all_candidates)} candidate rule(s). "
            f"Found {len(conflicts)} conflict(s)."
        ),
    )