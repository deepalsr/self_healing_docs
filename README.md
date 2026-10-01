# Self-Healing Documentation

A pipeline that watches a repo's pushes, figures out exactly which functions changed, and opens a pull request updating the documentation that describes them — grounded in the real diff, verified through three layers of guardrails before anything is proposed, and escalated to a human review PR rather than guessed at when it can't be done safely.

Built as a hands-on production-engineering exercise: AST-aware diffing, retrieval-augmented generation with Pinecone, deterministic + LLM-judge guardrails, bounded retry with escalation, and a fully wired webhook-to-PR pipeline, containerized and covered by CI.

## How it works

1. **Trigger** — GitHub sends a push webhook. The signature is verified (HMAC-SHA256, constant-time comparison) before anything else runs. The request is acknowledged immediately; processing happens in the background so GitHub never times out or retries a slow request.
2. **Diff** — only the `.py` files actually touched by the push are fetched (old and new versions) and parsed with tree-sitter into function/class-level definitions. Added, deleted, and modified functions are identified by name; modifications are further split into `signature`, `body_only`, or `both`.
3. **Chunk** — each changed function becomes one chunk: a clean, current-state text block for embedding, plus structured metadata (old/new signature, modification type, file path) for generation. The existing documentation for that function, if any, is fetched fresh from the repo and attached as `doc_snippet` — via an explicit `<!-- doc-anchor -->` comment if present, a matching Markdown heading, or a paragraph mentioning the function by name, in that order of preference.
4. **Store** — chunks are embedded (Gemini `gemini-embedding-001`, 768 dimensions) and upserted into Pinecone, namespaced per repo. Deleted functions are explicitly removed from the index rather than left as stale vectors.
5. **Retrieve** — before generating, the pipeline checks Pinecone for an exact match on this function's own chunk ID (cheap, deterministic). Only if that fails does it fall back to a similarity search, used purely for style reference on undocumented functions.
6. **Generate + guard** — an LLM proposes a structured JSON patch (`old_text` → `new_text`), which must pass three layers before it's trusted:
   - **Layer 1 (deterministic):** `old_text` must appear verbatim and exactly once in the existing doc; no-op and empty patches are rejected.
   - **Layer 2 (deterministic):** the patch may only touch the function it was generated for.
   - **Layer 3 (LLM-as-judge):** a second, low-temperature call checks the patch only describes what actually changed.

   A rejection at any layer feeds its reason back into the next generation attempt (bounded at 2 attempts). If no patch passes, the pipeline does not guess — it escalates.
7. **Apply + PR** — an approved patch is re-verified against the *live* file content (which may have changed since generation) and applied. A normal PR is opened for approved patches; a draft PR carrying the rejection reason is opened for escalated ones. Nothing is ever silently dropped or auto-merged.

## Architecture

```
app/
  main.py                    FastAPI webhook receiver + full pipeline orchestration
  config.py                  Typed, fail-fast settings (pydantic-settings)
  logging_config.py          Structured stdout logging

  diffing/
    ast_parser.py            tree-sitter based function-level diffing
  chunking/
    chunker.py                Diff entries -> embeddable chunks + metadata
  embeddings/
    embedder.py                Gemini embeddings (768-dim, pinned explicitly)
  vectorstore/
    pinecone_client.py        Index management, upsert/delete routing
  retrieval/
    retriever.py                Two-tier retrieval: exact match, then similarity
  generation/
    llm_client.py              Provider-isolated Gemini wrapper, retries 5xx with backoff
    patch_generator.py          Prompt construction + structured patch parsing
    judge.py                    Layer 3 LLM-as-judge
    pipeline.py                 Orchestrates generate -> guard -> retry -> escalate
  guardrails.py                 Layer 1 (grounding) + Layer 2 (scope) checks
  github/
    repo_reader.py               Fetches file contents + doc snippets from GitHub
    doc_extractor.py             Anchor / heading / paragraph doc-section extraction
    patch_applier.py             Applies a verified patch to live file content
    pr_manager.py                 Branch, commit, PR creation (both outcome paths)
  models/
    schemas.py                   DocPatch / PatchChange (Pydantic)

tests/                          43 tests: pure unit tests, mocked orchestration
                                  tests, and real integration tests against
                                  Gemini/Pinecone/GitHub

Dockerfile, .dockerignore         Containerized deployment
.github/workflows/test.yml        CI: full suite on every push/PR to main
```

## Setup

```bash
python -m venv sheal_env
source sheal_env/bin/activate
pip install -r requirements.txt
```

Create `.env` (see `.env.example`):

```bash
WEBHOOK_SECRET=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
GEMINI_API_KEY=<from Google AI Studio>
PINECONE_API_KEY=<from Pinecone dashboard>
GITHUB_TOKEN=<fine-grained PAT, scoped to the target repo: Contents + Pull requests read/write>
```

Run locally:

```bash
uvicorn app.main:app --reload --port 8000
```

Expose it for a real GitHub webhook (local dev only):

```bash
ngrok http 8000
```

Register the webhook on the target repo: Settings → Webhooks → Add webhook, payload URL `<ngrok-url>/webhook/github`, content type `application/json`, secret matching `WEBHOOK_SECRET`, events: push only.

## Running with Docker

```bash
docker build -t self-healing-docs .
docker run -p 8000:8000 --env-file .env self-healing-docs
```

## Tests

```bash
python -m pytest tests/ -v
```

43 tests covering diffing, chunking, embeddings, Pinecone (upsert/delete/routing), both retrieval tiers, prompt construction, patch parsing, both deterministic guardrail layers, the LLM judge and retry/escalation loop (mocked), patch application, both PR outcome paths, and the full webhook-to-pipeline wiring. Several tests exist specifically as regression guards for real bugs found during development (see Known limitations and the git history for specifics — e.g. a missing-body bug in chunk embedding text, a Pinecone null-metadata constraint, a regex false-positive in doc extraction).

CI (`.github/workflows/test.yml`) runs the full suite on every push and PR to `main`, using GitHub's encrypted repository secrets — verified to pass on a clean `ubuntu-latest` runner with zero local cache.

## Known limitations

This is an honest list, not hidden scope:

- **Single hardcoded doc file** (`README.md` at repo root). Multi-file docs (one file per module) aren't supported yet.
- **`insert` appends to the end of the file** rather than inserting near related content. Functional but not elegant for undocumented functions.
- **Rename detection**: diffing is by function name, so a rename plus no other change is reported as one deletion and one unrelated addition, losing continuity. Documented, not yet solved (would need body-similarity matching across name changes).
- **`needs_human_review` currently opens a draft PR with no diff** rather than a GitHub Issue, since there's nothing to review yet — a known, slightly awkward artifact.
- **`BackgroundTasks`, not a durable task queue.** A server restart mid-job silently loses that job. A real deployment should use Celery/Redis, SQS, or similar.
- **No multi-tenant repo configuration** — one deployment, one implicit set of assumptions. Scoping per-repo settings is unbuilt.
- **CI runs the full suite, including real API calls, on every push.** A fast/slow (`@pytest.mark.integration`) split would reduce cost and latency on routine pushes.
- **Doc-section extraction** works well for API-reference-style docs (headings, anchors) and reasonably for narrative prose (paragraph fallback), but has no understanding of documentation structures beyond regex-based heuristics.

## What this project demonstrates

Built step by step, debugging real issues as they surfaced rather than against pre-written working code:

- A model/package deprecation mid-build (`text-embedding-004`, `gemini-2.0-flash`, `google-generativeai` all retired during development) and the habit of pinning explicitly rather than trusting defaults
- A real external API constraint discovered only through integration testing (Pinecone rejecting `null` metadata values)
- Timing-attack-safe secret comparison (`hmac.compare_digest`), and why it matters
- Idempotency and race-condition reasoning applied twice: once to webhook redelivery, once to doc staleness between patch generation and patch application
- Pure/impure separation ("push impurity to the edges") applied consistently from chunking through generation through the webhook orchestrator, to keep 40+ tests fast and mock-free wherever possible
- A live, unmocked end-to-end run — real push, real webhook delivery, real guardrail pipeline, real PR — verified by screenshot, not just by test suite