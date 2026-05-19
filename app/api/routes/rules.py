"""
app/api/routes/rules.py
CRUD-style rule endpoints: load, list, reset.
"""

import os
from fastapi import APIRouter, HTTPException, Query

from app.graph.neo4j_client import get_driver, setup_schema, ingest_rules, get_all_rules
from app.core.config import get_settings

router = APIRouter()


@router.post("/load", summary="Load rules.csv into Neo4j")
def load_rules(
    csv_path: str = Query(default="data/rules.csv", description="Path to the rules CSV file."),
):
    """
    Reads a CSV file, computes Jina embeddings, and upserts Rule nodes into Neo4j.

    CSV columns: `rule_id` (required), `rule_text` (required), `category`, `description`.
    """
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"File not found: {csv_path}")
    try:
        driver = get_driver()
        count = ingest_rules(csv_path, driver)
        driver.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"message": f"Successfully loaded {count} rules into the knowledge graph."}


@router.get("", summary="List all rules in the graph")
def list_rules():
    """Returns all Rule nodes (embedding vectors excluded)."""
    try:
        driver = get_driver()
        rules = get_all_rules(driver)
        driver.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return [{"rule_id": r["rule_id"], "rule_text": r["rule_text"], "category": r["category"]} for r in rules]


@router.delete("/reset", summary="Delete all rules and conflicts from Neo4j")
def reset_graph():
    """Removes all Rule nodes and CONFLICTS_WITH relationships. Use for a clean re-ingest."""
    try:
        driver = get_driver()
        with driver.session(database=get_settings().neo4j_database) as session:
            session.run("MATCH (r:Rule) DETACH DELETE r")
        driver.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"message": "All Rule nodes and CONFLICTS_WITH relationships deleted."}
