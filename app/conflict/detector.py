"""
app/conflict/detector.py
Conflict detection workflows.
"""

import json
from itertools import combinations
from typing import Optional

from app.core.config import get_settings
from app.embeddings.jina import cosine_similarity, get_single_embedding
from app.llm.base import call_llm
from app.graph.neo4j_client import (
    get_all_rules, get_driver, save_conflict, find_similar_rules,
    add_rule_to_graph, get_all_conflicts,
)
from app.models.schemas import ConflictPair, CheckAllConflictsResponse, NewRuleConflictResponse

# ── Prompts ───────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a strict rule conflict analyst.
Your job is to identify when two rules are in conflict.

Rules ARE in conflict if ANY of the following apply:
1. They cannot both be satisfied simultaneously.
2. Following one necessarily violates the other.
3. One rule creates an exception that directly undermines or negates the other rule's requirement.
4. They impose contradictory obligations on the same subject.

Do NOT dismiss a conflict just because one rule frames itself as an exception or emergency provision.
Respond ONLY in valid JSON."""

_PROMPT_TEMPLATE = """Analyze whether these two rules conflict:

Rule A (ID: {id_a}):
{text_a}

Rule B (ID: {id_b}):
{text_b}

Respond with:
{{
  "conflicts": true or false,
  "explanation": "brief explanation"
}}"""


def _check_conflict(id_a: str, text_a: str, id_b: str, text_b: str) -> tuple[bool, str]:
    raw = call_llm(
        _PROMPT_TEMPLATE.format(id_a=id_a, text_a=text_a, id_b=id_b, text_b=text_b),
        system_prompt=_SYSTEM_PROMPT,
    )
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start != -1 and end > start:
        clean = clean[start : end + 1]
    try:
        result = json.loads(clean)
        return bool(result.get("conflicts", False)), result.get("explanation", "")
    except json.JSONDecodeError:
        lower = raw.lower()
        conflicts = '"conflicts": true' in lower or (
            "conflict" in lower and '"conflicts": false' not in lower
        )
        return conflicts, raw


# ── Workflow 1: check all ─────────────────────────────────────

def check_all_conflicts(rescan: bool = False) -> CheckAllConflictsResponse:
    s = get_settings()
    driver = get_driver()
    rules = get_all_rules(driver)

    if len(rules) < 2:
        driver.close()
        return CheckAllConflictsResponse(
            total_rules=len(rules),
            conflict_pairs=[],
            message="Not enough rules to check for conflicts.",
        )

    emb_map      = {r["rule_id"]: r["embedding"] for r in rules if r.get("embedding")}
    category_map = {r["rule_id"]: (r.get("category") or "").strip().lower() for r in rules}

    existing: set[str] = set()
    if not rescan:
        for c in get_all_conflicts(driver):
            existing.add("-".join(sorted([c["rule_id_a"], c["rule_id_b"]])))

    confirmed: list[ConflictPair] = []
    pairs = list(combinations(rules, 2))
    print(f"Checking {len(pairs)} pairs across {len(rules)} rules…")

    for rule_a, rule_b in pairs:
        id_a, id_b = rule_a["rule_id"], rule_b["rule_id"]
        pair_key   = "-".join(sorted([id_a, id_b]))

        if pair_key in existing:
            continue

        emb_a, emb_b = emb_map.get(id_a), emb_map.get(id_b)
        if not emb_a or not emb_b:
            continue

        sim = cosine_similarity(emb_a, emb_b)
        same_cat = bool(
            (cat := category_map.get(id_a)) and cat == category_map.get(id_b)
        )

        if sim < s.similarity_threshold and not same_cat:
            continue

        print(f"  → {id_a} vs {id_b}  (sim={sim:.4f}, same_cat={same_cat})")
        conflict, explanation = _check_conflict(id_a, rule_a["rule_text"], id_b, rule_b["rule_text"])

        if conflict:
            save_conflict(id_a, id_b, explanation, sim, driver)
            confirmed.append(ConflictPair(
                rule_id_a=id_a, rule_text_a=rule_a["rule_text"],
                rule_id_b=id_b, rule_text_b=rule_b["rule_text"],
                conflict_explanation=explanation, similarity_score=round(sim, 4),
            ))

    # Merge in already-stored conflicts when not rescanning
    if not rescan:
        confirmed_keys = {"-".join(sorted([c.rule_id_a, c.rule_id_b])) for c in confirmed}
        for stored in get_all_conflicts(driver):
            key = "-".join(sorted([stored["rule_id_a"], stored["rule_id_b"]]))
            if key not in confirmed_keys:
                confirmed.append(ConflictPair(
                    rule_id_a=stored["rule_id_a"], rule_text_a=stored["rule_text_a"],
                    rule_id_b=stored["rule_id_b"], rule_text_b=stored["rule_text_b"],
                    conflict_explanation=stored["conflict_explanation"],
                    similarity_score=round(stored["similarity_score"] or 0.0, 4),
                ))

    driver.close()
    return CheckAllConflictsResponse(
        total_rules=len(rules),
        conflict_pairs=confirmed,
        message=f"Found {len(confirmed)} conflict pair(s) among {len(rules)} rules.",
    )


# ── Workflow 2: check new rule ────────────────────────────────

def check_new_rule(
    rule_id: str,
    rule_text: str,
    category: Optional[str],
    description: Optional[str],
    save_to_db: bool,
) -> NewRuleConflictResponse:
    s = get_settings()
    driver = get_driver()

    new_rule = {"rule_id": rule_id, "rule_text": rule_text,
                "category": category or "", "description": description or ""}

    query_embedding = (
        add_rule_to_graph(new_rule, driver)
        if save_to_db
        else get_single_embedding(rule_text)
    )

    vector_candidates = find_similar_rules(
        query_embedding=query_embedding,
        top_k=s.top_k_candidates,
        exclude_rule_id=rule_id if save_to_db else None,
        driver=driver,
    )

    seen_ids: set[str] = {c["rule_id"] for c in vector_candidates}
    category_candidates: list[dict] = []

    if category:
        norm_cat = category.strip().lower()
        for r in get_all_rules(driver):
            if r["rule_id"] == rule_id or r["rule_id"] in seen_ids:
                continue
            if (r.get("category") or "").strip().lower() == norm_cat:
                category_candidates.append({"rule_id": r["rule_id"], "rule_text": r["rule_text"], "score": 0.0})
                seen_ids.add(r["rule_id"])

    filtered = [c for c in vector_candidates if c["score"] >= s.similarity_threshold]
    candidates = filtered + category_candidates
    print(f"Checking '{rule_id}' against {len(candidates)} candidate(s)…")

    conflicts: list[ConflictPair] = []
    for c in candidates:
        found, explanation = _check_conflict(rule_id, rule_text, c["rule_id"], c["rule_text"])
        if found:
            sim = c.get("score", 0.0)
            if save_to_db:
                save_conflict(rule_id, c["rule_id"], explanation, sim, driver)
            conflicts.append(ConflictPair(
                rule_id_a=rule_id, rule_text_a=rule_text,
                rule_id_b=c["rule_id"], rule_text_b=c["rule_text"],
                conflict_explanation=explanation, similarity_score=round(sim, 4),
            ))

    driver.close()
    status = "saved to database" if save_to_db else "not saved (pass save_to_db=true to persist)"
    return NewRuleConflictResponse(
        rule_id=rule_id, rule_text=rule_text, conflicts=conflicts,
        message=(
            f"Rule '{rule_id}' {status}. "
            f"Checked against {len(candidates)} candidate(s). "
            f"Found {len(conflicts)} conflict(s)."
        ),
    )
