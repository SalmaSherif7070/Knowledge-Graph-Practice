"""
app/graph/neo4j_client.py
All Neo4j interactions: schema setup, rule CRUD, conflict edges.
"""

import csv
import os
from typing import Optional

from neo4j import GraphDatabase

from app.core.config import get_settings
from app.embeddings.jina import get_embeddings, get_single_embedding

_VECTOR_DIMS = 1024

_SETUP_QUERIES = [
    """CREATE CONSTRAINT unique_rule IF NOT EXISTS
       FOR (r:Rule) REQUIRE r.rule_id IS UNIQUE""",

    """CREATE FULLTEXT INDEX fullTextRuleText IF NOT EXISTS
       FOR (r:Rule) ON EACH [r.rule_text, r.category]""",

    f"""CREATE VECTOR INDEX rule_embeddings IF NOT EXISTS
        FOR (r:Rule) ON (r.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {_VECTOR_DIMS},
                `vector.similarity_function`: 'cosine'
            }}
        }}""",
]


# ── Driver ────────────────────────────────────────────────────

def get_driver():
    s = get_settings()
    if not s.neo4j_uri or not s.neo4j_password:
        raise ValueError("NEO4J_URI and NEO4J_PASSWORD must be set in .env")
    return GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_username, s.neo4j_password))


def _db() -> str:
    return get_settings().neo4j_database


# ── Schema ────────────────────────────────────────────────────

def setup_schema(driver) -> None:
    with driver.session(database=_db()) as session:
        for q in _SETUP_QUERIES:
            try:
                session.run(q)
            except Exception as exc:
                print(f"Schema note: {exc}")


# ── Ingest ────────────────────────────────────────────────────

def _load_csv(csv_path: str) -> list[dict]:
    rules = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rules.append({
                "rule_id":     row.get("rule_id", "").strip(),
                "rule_text":   row.get("rule_text", "").strip(),
                "category":    row.get("category", "").strip(),
                "description": row.get("description", "").strip(),
            })
    return [r for r in rules if r["rule_id"] and r["rule_text"]]


_UPSERT_RULE = """
MERGE (r:Rule {rule_id: $rule_id})
ON CREATE SET r.rule_text=$rule_text, r.category=$category,
              r.description=$description, r.embedding=$embedding
ON MATCH  SET r.rule_text=$rule_text, r.category=$category,
              r.description=$description, r.embedding=$embedding
"""


def ingest_rules(csv_path: str, driver=None) -> int:
    rules = _load_csv(csv_path)
    if not rules:
        return 0

    embeddings = get_embeddings([r["rule_text"] for r in rules])

    _close = driver is None
    if _close:
        driver = get_driver()

    with driver.session(database=_db()) as session:
        for rule, emb in zip(rules, embeddings):
            session.run(_UPSERT_RULE, {**rule, "embedding": emb})

    if _close:
        driver.close()

    return len(rules)


# ── Read ──────────────────────────────────────────────────────

def get_all_rules(driver) -> list[dict]:
    with driver.session(database=_db()) as session:
        result = session.run(
            "MATCH (r:Rule) RETURN r.rule_id AS rule_id, r.rule_text AS rule_text, "
            "r.category AS category, r.description AS description, r.embedding AS embedding"
        )
        return [dict(rec) for rec in result]


def get_rule_by_id(rule_id: str, driver) -> Optional[dict]:
    with driver.session(database=_db()) as session:
        rec = session.run(
            "MATCH (r:Rule {rule_id: $rule_id}) "
            "RETURN r.rule_id AS rule_id, r.rule_text AS rule_text, "
            "r.category AS category, r.embedding AS embedding",
            {"rule_id": rule_id},
        ).single()
        return dict(rec) if rec else None


def add_rule_to_graph(rule: dict, driver) -> list[float]:
    emb = get_single_embedding(rule["rule_text"])
    with driver.session(database=_db()) as session:
        session.run(_UPSERT_RULE, {
            "rule_id": rule["rule_id"], "rule_text": rule["rule_text"],
            "category": rule.get("category", ""), "description": rule.get("description", ""),
            "embedding": emb,
        })
    return emb


# ── Vector search ─────────────────────────────────────────────

def find_similar_rules(
    query_embedding: list[float],
    top_k: int = 20,
    exclude_rule_id: Optional[str] = None,
    driver=None,
) -> list[dict]:
    _close = driver is None
    if _close:
        driver = get_driver()

    with driver.session(database=_db()) as session:
        rows = [dict(rec) for rec in session.run(
            """
            CALL db.index.vector.queryNodes('rule_embeddings', $top_k, $embedding)
            YIELD node, score
            WHERE ($exclude_id IS NULL OR node.rule_id <> $exclude_id)
            RETURN node.rule_id AS rule_id, node.rule_text AS rule_text,
                   node.category AS category, score
            ORDER BY score DESC
            """,
            {
                "top_k": top_k + (1 if exclude_rule_id else 0),
                "embedding": query_embedding,
                "exclude_id": exclude_rule_id,
            },
        )]

    if _close:
        driver.close()

    return rows[:top_k]


# ── Conflicts ─────────────────────────────────────────────────

def save_conflict(rule_id_a: str, rule_id_b: str, explanation: str, score: float, driver) -> None:
    pair_key = "-".join(sorted([rule_id_a, rule_id_b]))
    with driver.session(database=_db()) as session:
        session.run(
            """
            MATCH (a:Rule {rule_id: $id_a}), (b:Rule {rule_id: $id_b})
            MERGE (a)-[r:CONFLICTS_WITH {pair_key: $pair_key}]->(b)
            ON CREATE SET r.explanation=$explanation, r.similarity_score=$score
            ON MATCH  SET r.explanation=$explanation, r.similarity_score=$score
            """,
            {"id_a": rule_id_a, "id_b": rule_id_b, "pair_key": pair_key,
             "explanation": explanation, "score": score},
        )


def get_all_conflicts(driver) -> list[dict]:
    with driver.session(database=_db()) as session:
        return [dict(rec) for rec in session.run(
            """
            MATCH (a:Rule)-[r:CONFLICTS_WITH]->(b:Rule)
            RETURN a.rule_id AS rule_id_a, a.rule_text AS rule_text_a,
                   b.rule_id AS rule_id_b, b.rule_text AS rule_text_b,
                   r.explanation AS conflict_explanation,
                   r.similarity_score AS similarity_score
            """
        )]


def get_conflicts_for_rule(rule_id: str, driver) -> list[dict]:
    with driver.session(database=_db()) as session:
        return [dict(rec) for rec in session.run(
            """
            MATCH (a:Rule {rule_id: $rule_id})-[r:CONFLICTS_WITH]->(b:Rule)
            RETURN b.rule_id AS rule_id_b, b.rule_text AS rule_text_b,
                   r.explanation AS conflict_explanation, r.similarity_score AS similarity_score
            UNION
            MATCH (a:Rule)-[r:CONFLICTS_WITH]->(b:Rule {rule_id: $rule_id})
            RETURN a.rule_id AS rule_id_b, a.rule_text AS rule_text_b,
                   r.explanation AS conflict_explanation, r.similarity_score AS similarity_score
            """,
            {"rule_id": rule_id},
        )]
