"""
conflict.py
Orchestrates the two conflict-detection workflows:
  1. check_all_conflicts  — scan every rule pair in the DB
  2. check_new_rule       — check one rule against all DB rules
"""

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

# Similarity threshold — pairs above this are sent to the LLM for conflict checking.
# Lower = more pairs checked (slower, more thorough).
# Higher = fewer pairs checked (faster, may miss subtle conflicts).
SIMILARITY_THRESHOLD = float(__import__("os").getenv("SIMILARITY_THRESHOLD", "0.60"))

# How many similar candidates to retrieve per rule when checking a new rule
TOP_K_CANDIDATES = int(__import__("os").getenv("TOP_K_CANDIDATES", "20"))


# ──────────────────────────────────────────────
# Endpoint 1: Check all rules against each other
# ──────────────────────────────────────────────

def check_all_conflicts(rescan: bool = False) -> CheckAllConflictsResponse:
    """
    1. Load all Rule nodes + their embeddings from Neo4j.
    2. For every pair, compute cosine similarity.
    3. Send candidate pairs (similarity >= threshold) to Gemini.
    4. Persist confirmed conflicts as CONFLICTS_WITH edges.
    5. Return all conflict pairs.

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

    # Build lookup: rule_id → embedding
    emb_map = {r["rule_id"]: r["embedding"] for r in rules if r.get("embedding")}

    # Optionally fetch already-known conflict pairs to skip re-checking
    existing_conflicts: set[str] = set()
    if not rescan:
        for c in get_all_conflicts(driver):
            key = "-".join(sorted([c["rule_id_a"], c["rule_id_b"]]))
            existing_conflicts.add(key)

    confirmed_conflicts: list[ConflictPair] = []

    pairs = list(combinations(rules, 2))
    print(f"Checking {len(pairs)} pairs across {len(rules)} rules…")

    for rule_a, rule_b in pairs:
        id_a, id_b = rule_a["rule_id"], rule_b["rule_id"]
        pair_key = "-".join(sorted([id_a, id_b]))

        if pair_key in existing_conflicts:
            # Already confirmed conflict — re-add from DB
            continue

        emb_a = emb_map.get(id_a)
        emb_b = emb_map.get(id_b)
        if emb_a is None or emb_b is None:
            continue

        sim = cosine_similarity(emb_a, emb_b)
        if sim < SIMILARITY_THRESHOLD:
            continue  # semantically too different to conflict

        # Ask Gemini
        conflicts, explanation = check_conflict_with_llm(
            id_a, rule_a["rule_text"],
            id_b, rule_b["rule_text"],
        )

        if conflicts:
            save_conflict(id_a, id_b, explanation, sim, driver)
            confirmed_conflicts.append(ConflictPair(
                rule_id_a=id_a,
                rule_text_a=rule_a["rule_text"],
                rule_id_b=id_b,
                rule_text_b=rule_b["rule_text"],
                conflict_explanation=explanation,
                similarity_score=round(sim, 4),
            ))

    # Also include previously stored conflicts (when rescan=False)
    if not rescan:
        for stored in get_all_conflicts(driver):
            pair_key = "-".join(sorted([stored["rule_id_a"], stored["rule_id_b"]]))
            already = any(
                "-".join(sorted([c.rule_id_a, c.rule_id_b])) == pair_key
                for c in confirmed_conflicts
            )
            if not already:
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
    3. Send each candidate pair to Gemini.
    4. Optionally persist the new rule + its conflicts to the graph.
    """
    driver = get_driver()

    # Add (or temporarily hold) the new rule
    new_rule = {
        "rule_id": rule_id,
        "rule_text": rule_text,
        "category": category or "",
        "description": description or "",
    }

    # Compute embedding (also saves to DB if save_to_db=True)
    if save_to_db:
        query_embedding = add_rule_to_graph(new_rule, driver)
    else:
        from embeddings import get_single_embedding
        query_embedding = get_single_embedding(rule_text)

    # Find similar rules from DB (exclude the rule itself if it was just saved)
    candidates = find_similar_rules(
        query_embedding=query_embedding,
        top_k=TOP_K_CANDIDATES,
        exclude_rule_id=rule_id if save_to_db else None,
        driver=driver,
    )

    conflicts: list[ConflictPair] = []

    for candidate in candidates:
        sim = candidate["score"]
        if sim < SIMILARITY_THRESHOLD:
            break  # results are ordered by score desc

        conflicts_found, explanation = check_conflict_with_llm(
            rule_id, rule_text,
            candidate["rule_id"], candidate["rule_text"],
        )

        if conflicts_found:
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
            f"Checked against {len(candidates)} candidate rules. "
            f"Found {len(conflicts)} conflict(s)."
        ),
    )