"""
app/api/routes/conflicts.py
Conflict detection endpoints: check-all, check-new.
"""

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    NewRuleRequest,
    CheckAllConflictsResponse,
    NewRuleConflictResponse,
)
from app.conflict.detector import check_all_conflicts, check_new_rule

router = APIRouter()


@router.post(
    "/check-all",
    response_model=CheckAllConflictsResponse,
    summary="Find all conflicting rule pairs in the database",
)
def check_all(
    rescan: bool = Query(
        default=False,
        description="Re-evaluate ALL pairs even if a CONFLICTS_WITH edge already exists.",
    )
):
    """
    Scans every rule pair in the graph. Pairs above the similarity threshold
    (or in the same category) are sent to the LLM for conflict analysis.
    Confirmed conflicts are stored as CONFLICTS_WITH edges.
    """
    try:
        return check_all_conflicts(rescan=rescan)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/check-new",
    response_model=NewRuleConflictResponse,
    summary="Check if a new rule conflicts with any existing rule",
)
def check_new(body: NewRuleRequest):
    """
    Embeds the new rule, finds the most similar existing rules via Neo4j ANN search,
    and sends each candidate pair to the LLM. Set `save_to_db=true` to persist.
    """
    try:
        return check_new_rule(
            rule_id=body.rule_id,
            rule_text=body.rule_text,
            category=body.category,
            description=body.description,
            save_to_db=body.save_to_db,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
