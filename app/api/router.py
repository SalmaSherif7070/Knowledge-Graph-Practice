"""
app/api/router.py
Mounts all route modules under a single APIRouter.
"""

from fastapi import APIRouter
from app.api.routes import rules, conflicts

api_router = APIRouter()
api_router.include_router(rules.router,     prefix="/rules", tags=["rules"])
api_router.include_router(conflicts.router, prefix="/rules", tags=["conflict-detection"])
