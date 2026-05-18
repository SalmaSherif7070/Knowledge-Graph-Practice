"""
kg_builder.py
Handles all Neo4j interactions for Rule nodes:
  - Load rules.csv into the graph
  - Store/retrieve embeddings
  - Persist CONFLICTS_WITH relationships
"""

import os
import csv
import json
from typing import Optional

from neo4j import GraphDatabase

from embeddings import get_embeddings, get_single_embedding

VECTOR_DIMS = 1024  # jina-embeddings-v3 outputs 1024-dim vectors by default


# ──────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────

def get_driver():
    uri      = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not password:
        raise ValueError("NEO4J_URI and NEO4J_PASSWORD must be set in .env")
    return GraphDatabase.driver(uri, auth=(username, password))


# ──────────────────────────────────────────────
# Schema setup
# ──────────────────────────────────────────────

SETUP_QUERIES = [
    # Uniqueness constraint on rule_id
    """CREATE CONSTRAINT unique_rule IF NOT EXISTS
       FOR (r:Rule) REQUIRE r.rule_id IS UNIQUE""",

    # Full-text index for keyword search
    """CREATE FULLTEXT INDEX fullTextRuleText IF NOT EXISTS
       FOR (r:Rule) ON EACH [r.rule_text, r.category]""",

    # Vector index for semantic similarity
    f"""CREATE VECTOR INDEX rule_embeddings IF NOT EXISTS
        FOR (r:Rule) ON (r.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {VECTOR_DIMS},
                `vector.similarity_function`: 'cosine'
            }}
        }}""",
]


def _db():
    return os.getenv("NEO4J_DATABASE", "neo4j")


def setup_schema(driver):
    with driver.session(database=_db()) as session:
        for q in SETUP_QUERIES:
            try:
                session.run(q)
            except Exception as e:
                # Constraint/index may already exist — that's fine
                print(f"Schema setup note: {e}")


# ──────────────────────────────────────────────
# Load rules.csv
# ──────────────────────────────────────────────

def load_rules_from_csv(csv_path: str) -> list[dict]:
    """
    Expected CSV columns (at minimum): rule_id, rule_text
    Optional columns: category, description
    """
    rules = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rules.append({
                "rule_id": row.get("rule_id", "").strip(),
                "rule_text": row.get("rule_text", "").strip(),
                "category": row.get("category", "").strip(),
                "description": row.get("description", "").strip(),
            })
    return [r for r in rules if r["rule_id"] and r["rule_text"]]


def ingest_rules(csv_path: str, driver=None) -> int:
    """
    Load rules.csv → compute Jina embeddings → upsert Rule nodes in Neo4j.
    Returns the count of rules ingested.
    """
    rules = load_rules_from_csv(csv_path)
    if not rules:
        return 0

    # Batch-compute embeddings
    texts = [r["rule_text"] for r in rules]
    embeddings = get_embeddings(texts)

    close_after = driver is None
    if driver is None:
        driver = get_driver()

    merge_query = """
    MERGE (r:Rule {rule_id: $rule_id})
    ON CREATE SET
        r.rule_text   = $rule_text,
        r.category    = $category,
        r.description = $description,
        r.embedding   = $embedding
    ON MATCH SET
        r.rule_text   = $rule_text,
        r.category    = $category,
        r.description = $description,
        r.embedding   = $embedding
    RETURN r.rule_id
    """

    with driver.session(database=_db()) as session:
        for rule, emb in zip(rules, embeddings):
            session.run(merge_query, {
                "rule_id":     rule["rule_id"],
                "rule_text":   rule["rule_text"],
                "category":    rule["category"],
                "description": rule["description"],
                "embedding":   emb,
            })

    if close_after:
        driver.close()

    return len(rules)


# ──────────────────────────────────────────────
# Retrieve rules
# ──────────────────────────────────────────────

def get_all_rules(driver) -> list[dict]:
    with driver.session(database=_db()) as session:
        result = session.run(
            "MATCH (r:Rule) RETURN r.rule_id AS rule_id, "
            "r.rule_text AS rule_text, r.category AS category, "
            "r.description AS description, r.embedding AS embedding"
        )
        return [dict(rec) for rec in result]


def get_rule_by_id(rule_id: str, driver) -> Optional[dict]:
    with driver.session(database=_db()) as session:
        result = session.run(
            "MATCH (r:Rule {rule_id: $rule_id}) "
            "RETURN r.rule_id AS rule_id, r.rule_text AS rule_text, "
            "r.category AS category, r.embedding AS embedding",
            {"rule_id": rule_id}
        )
        rec = result.single()
        return dict(rec) if rec else None


# ──────────────────────────────────────────────
# Vector similarity search (ANN in Neo4j)
# ──────────────────────────────────────────────

def find_similar_rules(
    query_embedding: list[float],
    top_k: int = 20,
    exclude_rule_id: Optional[str] = None,
    driver=None,
) -> list[dict]:
    """
    Use Neo4j vector index to find rules most similar to the query embedding.
    Returns list of {rule_id, rule_text, category, score}.
    """
    close_after = driver is None
    if driver is None:
        driver = get_driver()

    query = """
    CALL db.index.vector.queryNodes('rule_embeddings', $top_k, $embedding)
    YIELD node, score
    WHERE ($exclude_id IS NULL OR node.rule_id <> $exclude_id)
    RETURN node.rule_id  AS rule_id,
           node.rule_text AS rule_text,
           node.category  AS category,
           score
    ORDER BY score DESC
    """

    with driver.session(database=_db()) as session:
        result = session.run(query, {
            "top_k": top_k + (1 if exclude_rule_id else 0),  # fetch extra to compensate for exclusion
            "embedding": query_embedding,
            "exclude_id": exclude_rule_id,
        })
        rows = [dict(rec) for rec in result]

    if close_after:
        driver.close()

    return rows[:top_k]


# ──────────────────────────────────────────────
# Persist / query CONFLICTS_WITH relationships
# ──────────────────────────────────────────────

def save_conflict(rule_id_a: str, rule_id_b: str, explanation: str, score: float, driver):
    """Create a CONFLICTS_WITH relationship (bidirectional via two directed edges)."""
    query = """
    MATCH (a:Rule {rule_id: $id_a}), (b:Rule {rule_id: $id_b})
    MERGE (a)-[r:CONFLICTS_WITH {pair_key: $pair_key}]->(b)
    ON CREATE SET r.explanation = $explanation, r.similarity_score = $score
    ON MATCH  SET r.explanation = $explanation, r.similarity_score = $score
    """
    # Canonical pair key so we don't duplicate in both directions
    pair_key = "-".join(sorted([rule_id_a, rule_id_b]))
    with driver.session(database=_db()) as session:
        session.run(query, {
            "id_a": rule_id_a,
            "id_b": rule_id_b,
            "pair_key": pair_key,
            "explanation": explanation,
            "score": score,
        })


def get_all_conflicts(driver) -> list[dict]:
    """Return all stored CONFLICTS_WITH edges."""
    query = """
    MATCH (a:Rule)-[r:CONFLICTS_WITH]->(b:Rule)
    RETURN a.rule_id   AS rule_id_a,
           a.rule_text AS rule_text_a,
           b.rule_id   AS rule_id_b,
           b.rule_text AS rule_text_b,
           r.explanation     AS conflict_explanation,
           r.similarity_score AS similarity_score
    """
    with driver.session(database=_db()) as session:
        return [dict(rec) for rec in session.run(query)]


def get_conflicts_for_rule(rule_id: str, driver) -> list[dict]:
    query = """
    MATCH (a:Rule {rule_id: $rule_id})-[r:CONFLICTS_WITH]->(b:Rule)
    RETURN b.rule_id   AS rule_id_b,
           b.rule_text AS rule_text_b,
           r.explanation      AS conflict_explanation,
           r.similarity_score AS similarity_score
    UNION
    MATCH (a:Rule)-[r:CONFLICTS_WITH]->(b:Rule {rule_id: $rule_id})
    RETURN a.rule_id   AS rule_id_b,
           a.rule_text AS rule_text_b,
           r.explanation      AS conflict_explanation,
           r.similarity_score AS similarity_score
    """
    with driver.session(database=_db()) as session:
        return [dict(rec) for rec in session.run(query, {"rule_id": rule_id})]


def add_rule_to_graph(rule: dict, driver):
    """Upsert a single rule node (with its embedding)."""
    emb = get_single_embedding(rule["rule_text"])
    merge_query = """
    MERGE (r:Rule {rule_id: $rule_id})
    ON CREATE SET r.rule_text=$rule_text, r.category=$category,
                  r.description=$description, r.embedding=$embedding
    ON MATCH  SET r.rule_text=$rule_text, r.category=$category,
                  r.description=$description, r.embedding=$embedding
    """
    with driver.session(database=_db()) as session:
        session.run(merge_query, {
            "rule_id":     rule["rule_id"],
            "rule_text":   rule["rule_text"],
            "category":    rule.get("category", ""),
            "description": rule.get("description", ""),
            "embedding":   emb,
        })
    return emb