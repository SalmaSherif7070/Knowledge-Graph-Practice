"""
main.py
Thin entrypoint — wires FastAPI app and registers all routes.

Run with:
    uvicorn main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(".env", override=True)

from fastapi import FastAPI
from app.api.router import api_router
from app.graph.neo4j_client import get_driver, setup_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        driver = get_driver()
        setup_schema(driver)
        driver.close()
        print("✅ Neo4j schema ready.")
    except Exception as exc:
        print(f"⚠️  Neo4j schema setup failed: {exc}")
    yield


app = FastAPI(
    title="Rule Conflict Detection API",
    description="Knowledge-graph-powered rule conflict detection using Neo4j, Jina, and LLM.",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(api_router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}
