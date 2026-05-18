from pydantic import BaseModel
from typing import Optional


class Rule(BaseModel):
    rule_id: str
    rule_text: str
    category: Optional[str] = None
    description: Optional[str] = None


class ConflictPair(BaseModel):
    rule_id_a: str
    rule_text_a: str
    rule_id_b: str
    rule_text_b: str
    conflict_explanation: str
    similarity_score: float


class CheckAllConflictsResponse(BaseModel):
    total_rules: int
    conflict_pairs: list[ConflictPair]
    message: str


class NewRuleRequest(BaseModel):
    rule_id: str
    rule_text: str
    category: Optional[str] = None
    description: Optional[str] = None
    save_to_db: bool = False  # if True, persist the rule to the graph


class NewRuleConflictResponse(BaseModel):
    rule_id: str
    rule_text: str
    conflicts: list[ConflictPair]
    message: str