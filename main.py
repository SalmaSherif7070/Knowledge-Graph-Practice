"""
main.py — Rule Conflict Detection API

Endpoints:
  POST /rules/load          — Load rules.csv into the Neo4j knowledge graph
  POST /rules/check-all     — Find all conflicting rule pairs in the database
  POST /rules/check-new     — Check if a new rule conflicts with any stored rule
  GET  /rules               — List all rules in the graph
  GET  /health              — Health check
"""

from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
load_dotenv(".env", override=True)

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from models import (
    NewRuleRequest,
    NewRuleConflictResponse,
    CheckAllConflictsResponse,
)
from kg_builder import (
    get_driver,
    setup_schema,
    ingest_rules,
    get_all_rules,
)
from conflict import check_all_conflicts, check_new_rule


# ──────────────────────────────────────────────
# App lifecycle
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: ensure Neo4j schema (constraints + indexes) exists
    try:
        driver = get_driver()
        setup_schema(driver)
        driver.close()
        print("✅ Neo4j schema ready.")
    except Exception as e:
        print(f"⚠️  Neo4j schema setup failed: {e}")
    yield


app = FastAPI(
    title="Rule Conflict Detection API",
    description=(
        "Knowledge-graph-powered rule conflict detection using "
        "Neo4j, Jina embeddings, and Gemini LLM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


# ──────────────────────────────────────────────
# Load rules.csv into the graph
# ──────────────────────────────────────────────

@app.post("/rules/load", tags=["rules"], summary="Load rules.csv into Neo4j")
def load_rules(
    csv_path: str = Query(
        default="rules.csv",
        description="Path to the rules CSV file (relative to the working directory).",
    )
):
    """
    Reads `rules.csv`, computes Jina embeddings for each rule,
    and upserts them as `Rule` nodes in Neo4j.

    **CSV format** (columns):
    - `rule_id`    — unique identifier  (required)
    - `rule_text`  — the rule content   (required)
    - `category`   — rule category      (optional)
    - `description`— extra context      (optional)
    """
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"File not found: {csv_path}")

    try:
        driver = get_driver()
        count = ingest_rules(csv_path, driver)
        driver.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"Successfully loaded {count} rules into the knowledge graph."}

@app.delete("/rules/reset", tags=["rules"], summary="Delete all rules and conflicts from Neo4j")
def reset_graph():
    """
    Removes all Rule nodes and CONFLICTS_WITH relationships from the graph.
    Useful for a clean re-ingest.
    """
    try:
        driver = get_driver()
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            session.run("MATCH (r:Rule) DETACH DELETE r")
        driver.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "All Rule nodes and CONFLICTS_WITH relationships deleted."}

# ──────────────────────────────────────────────
# List all rules
# ──────────────────────────────────────────────

@app.get("/rules", tags=["rules"], summary="List all rules in the graph")
def list_rules():
    try:
        driver = get_driver()
        rules = get_all_rules(driver)
        driver.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Strip raw embedding vectors from the response (too large)
    return [
        {
            "rule_id": r["rule_id"],
            "rule_text": r["rule_text"],
            "category": r["category"],
        }
        for r in rules
    ]


# ──────────────────────────────────────────────
# Endpoint 1: Check all rules for conflicts
# ──────────────────────────────────────────────

@app.post(
    "/rules/check-all",
    response_model=CheckAllConflictsResponse,
    tags=["conflict-detection"],
    summary="Find all conflicting rule pairs in the database",
)
def check_all(
    rescan: bool = Query(
        default=False,
        description=(
            "If true, re-evaluate ALL pairs even if a CONFLICTS_WITH "
            "relationship already exists. Useful after rules are updated."
        ),
    )
):
    """
    Scans every rule in the Neo4j knowledge graph and identifies pairs
    that conflict with each other.

    **Algorithm:**
    1. Load all `Rule` nodes with their Jina embedding vectors.
    2. Compute cosine similarity for every pair.
    3. Pairs above the similarity threshold are sent to Gemini for
       conflict analysis.
    4. Confirmed conflicts are stored as `CONFLICTS_WITH` edges.
    5. All conflict pairs are returned.

    > **Note:** For large rule sets this can be slow (O(n²) pairs).
    > Set `SIMILARITY_THRESHOLD` in `.env` to control the cutoff
    > (default 0.60).  Only semantically similar rules are LLM-checked.
    """
    try:
        return check_all_conflicts(rescan=rescan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Endpoint 2: Check a new rule against the DB
# ──────────────────────────────────────────────

@app.post(
    "/rules/check-new",
    response_model=NewRuleConflictResponse,
    tags=["conflict-detection"],
    summary="Check if a new rule conflicts with any existing rule",
)
def check_new(body: NewRuleRequest):
    """
    Takes a new rule and checks whether it conflicts with any rule
    already stored in the knowledge graph.

    **Algorithm:**
    1. Compute a Jina embedding for the new rule.
    2. Use Neo4j's vector index (ANN search) to find the most similar
       existing rules.
    3. Send each candidate pair to Gemini for conflict analysis.
    4. If `save_to_db=true`, the new rule and any confirmed conflicts
       are persisted to the graph.

    **Body fields:**
    - `rule_id`    — unique identifier for the new rule
    - `rule_text`  — the full text of the rule
    - `category`   — (optional) rule category
    - `description`— (optional) additional context
    - `save_to_db` — (optional, default false) persist the rule to Neo4j
    """
    try:
        return check_new_rule(
            rule_id=body.rule_id,
            rule_text=body.rule_text,
            category=body.category,
            description=body.description,
            save_to_db=body.save_to_db,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))