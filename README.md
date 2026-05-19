# Rule Conflict Detection Agent

Detects conflicting rules using **Neo4j** (knowledge graph), **Jina** (embeddings), and an LLM (**Gemini** or **Groq**).

## Project Structure

```
knowledge-graph-practice/
├── app/
│   ├── api/
│   │   ├── router.py          # mounts all sub-routers
│   │   └── routes/
│   │       ├── rules.py       # /rules/* CRUD endpoints
│   │       └── conflicts.py   # /rules/check-all, /rules/check-new
│   ├── core/
│   │   └── config.py          # all env vars via pydantic-settings
│   ├── llm/
│   │   ├── base.py            # call_llm(prompt, provider="gemini"|"groq")
│   │   ├── gemini.py          # Gemini HTTP logic + multi-key fallback
│   │   └── groq.py            # Groq HTTP logic (OpenAI-compatible)
│   ├── graph/
│   │   └── neo4j_client.py    # all Neo4j interactions
│   ├── embeddings/
│   │   └── jina.py            # Jina AI embedding calls
│   ├── conflict/
│   │   └── detector.py        # conflict detection workflows
│   └── models/
│       └── schemas.py         # Pydantic request/response models
├── data/
│   ├── rules.csv
│   └── ground_truth.csv
├── scripts/
│   └── eval.py                # evaluation script
├── main.py                    # thin FastAPI entrypoint
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
```

## Run

```bash
uvicorn main:app --reload --port 8000
```

## LLM Providers

Set `LLM_PROVIDER=gemini` or `LLM_PROVIDER=groq` in `.env`.  
Or override per-call: `call_llm(prompt, provider="groq")`.

| Provider | Model env var    | Default                    |
|----------|-----------------|----------------------------|
| Gemini   | `GEMINI_MODEL`  | `gemini-2.0-flash`         |
| Groq     | `GROQ_MODEL`    | `llama-3.3-70b-versatile`  |

## Evaluation

```bash
python -m scripts.eval
python -m scripts.eval --rescan --skip-ingest
```

## Endpoints

| Method | Path                  | Description                        |
|--------|-----------------------|------------------------------------|
| GET    | /health               | Health check                       |
| POST   | /rules/load           | Load rules CSV into Neo4j          |
| GET    | /rules                | List all rules                     |
| DELETE | /rules/reset          | Wipe all rules and conflicts       |
| POST   | /rules/check-all      | Scan all rule pairs for conflicts  |
| POST   | /rules/check-new      | Check a new rule against the DB   |
