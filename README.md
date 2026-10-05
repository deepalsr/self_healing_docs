# Self-Healing Documentation

A pipeline that watches a repo's pushes, figures out exactly which functions changed, and opens a pull request updating the documentation that describes them — grounded in the real diff, verified through three layers of guardrails before anything is proposed, and escalated to a human-review PR rather than guessed at when it can't be done safely.

Built as a hands-on production-engineering exercise: AST-aware diffing, retrieval-augmented generation with Pinecone, deterministic + LLM-judge guardrails, bounded retry with escalation, multi-file doc resolution, and a fully wired webhook-to-PR pipeline — containerized, allowlisted, and covered by a cost-aware CI split.

## How it works

1. **Trigger** — GitHub sends a push webhook. The signature is verified (HMAC-SHA256, constant-time comparison) before anything else runs. The request is checked against a per-repo allowlist, then acknowledged immediately; processing happens in the background so GitHub never times out or retries a slow request.
2. **Diff** — only the `.py` files actually touched by the push are fetched (old and new versions) and parsed with tree-sitter into function/class-level definitions. Added, deleted, and modified functions are identified by name; modifications are further split into `signature`, `body_only`, or `both`.
3. **Resolve the doc target** — each changed source file is mapped to its own doc file by convention (`src/ingest.py` → `docs/ingest.md`), falling back to the repo's root `README.md` when no module-specific doc exists. No per-repo config file required for the common case.
4. **Chunk** — each changed function becomes one chunk: a clean, current-state text block for embedding, plus structured metadata (old/new signature, modification type, file path) for generation. The existing documentation for that function, if any, is fetched fresh from the resolved doc file and attached as `doc_snippet` — via an explicit `<!-- doc-anchor -->` comment if present, a matching Markdown heading, or a paragraph mentioning the function by name, in that order of preference.
5. **Store** — chunks are embedded (Gemini `gemini-embedding-001`, 768 dimensions) and upserted into Pinecone, namespaced per repo. Deleted functions are explicitly removed from the index rather than left as stale vectors.
6. **Retrieve** — before generating, the pipeline checks Pinecone for an exact match on this function's own chunk ID (cheap, deterministic). Only if that fails does it fall back to a similarity search, used both for style reference on undocumented functions and to anchor *where* a new section gets inserted.
7. **Generate + guard** — an LLM proposes a structured JSON patch (`old_text` → `new_text`), which must pass three layers before it's trusted:
   - **Layer 1 (deterministic):** `old_text` must appear verbatim and exactly once in the existing doc; no-op and empty patches are rejected.
   - **Layer 2 (deterministic):** the patch may only touch the function it was generated for.
   - **Layer 3 (LLM-as-judge):** a second, low-temperature call checks the patch only describes what actually changed.

   A rejection at any layer feeds its reason back into the next generation attempt (bounded at 2 attempts). If no patch passes, the pipeline does not guess — it escalates.
8. **Apply + PR** — an approved patch is re-verified against the *live* file content (which may have changed since generation) and applied. A new, undocumented function is inserted near a similar function's existing section when one was found in retrieval, rather than always appended at the end of the file. A normal PR is opened for approved patches; a draft PR carrying the rejection reason is opened for escalated ones. Nothing is ever silently dropped or auto-merged.

## Architecture

```
app/
  main.py                      FastAPI webhook receiver + full pipeline orchestration
  config.py                    Typed settings (pydantic-settings); optional fields so
                                 test collection works with zero secrets present
  logging_config.py            Structured stdout logging

  diffing/
    ast_parser.py               tree-sitter based function-level diffing
  chunking/
    chunker.py                   Diff entries -> embeddable chunks + metadata
  embeddings/
    embedder.py                   Gemini embeddings (768-dim, pinned explicitly),
                                   lazily constructed client
  vectorstore/
    pinecone_client.py            Index management, upsert/delete routing,
                                   lazily constructed client
  retrieval/
    retriever.py                   Two-tier retrieval: exact match, then similarity
  generation/
    llm_client.py                 Provider-isolated Gemini wrapper, retries 5xx with
                                   backoff, lazily constructed client
    patch_generator.py             Prompt construction + structured patch parsing
    judge.py                       Layer 3 LLM-as-judge
    pipeline.py                    Orchestrates generate -> guard -> retry -> escalate
  guardrails.py                   Layer 1 (grounding) + Layer 2 (scope) checks
  github/
    repo_reader.py                  Fetches file contents + doc snippets from GitHub
    doc_resolver.py                  Maps a source file to its conventional doc path,
                                      falling back to README.md
    doc_extractor.py                 Anchor / heading / paragraph doc-section extraction
    patch_applier.py                 Applies a verified patch to live file content;
                                      inserts new sections near similar content
    pr_manager.py                    Branch, commit, PR creation (both outcome paths)
  models/
    schemas.py                       DocPatch / PatchChange (Pydantic)

tests/                            48 tests total: pure unit tests, mocked orchestration
                                    tests, and real integration tests against
                                    Gemini/Pinecone/GitHub. Marked with
                                    @pytest.mark.integration so CI can run a fast,
                                    free, 44-test subset on every push.

pytest.ini                         Registers the `integration` marker
Dockerfile, .dockerignore          Containerized deployment
.github/workflows/test.yml         CI: fast suite on every push/PR, full suite
                                    (real API calls) on a daily schedule or manual run
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
ALLOWED_REPOS=<comma-separated "owner/repo" list; leave empty to allow any repo pointed at this deployment>
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
python -m pytest tests/ -v -m "not integration"   # fast, free, no network -- 44 tests
python -m pytest tests/ -v -m "integration"          # real Gemini/Pinecone calls -- 4 tests
python -m pytest tests/ -v                           # everything -- 48 tests
```

Coverage: diffing, chunking, embeddings, Pinecone (upsert/delete/routing), both retrieval tiers, doc-path resolution (module convention + README fallback), prompt construction, patch parsing, both deterministic guardrail layers, the LLM judge and retry/escalation loop (mocked), patch application (including near-reference insertion), both PR outcome paths, the per-repo allowlist (including a full HTTP-boundary test with a real computed signature), and the full webhook-to-pipeline wiring.

Several tests exist specifically as regression guards for real bugs found during development — a missing-body bug in chunk embedding text, a Pinecone null-metadata constraint, a regex false-positive in doc extraction, and a handful of CI-only failures caused by tests that passed locally only because a populated `.env` was masking a real gap (eager client construction at import time; a mocked test with one unmocked collaborator). See git history for specifics.

CI (`.github/workflows/test.yml`) splits by cost: the 44 fast tests run on every push and PR using GitHub's encrypted repository secrets (though they need none, by design); the full 48-test suite, including real API calls, runs on a daily schedule or manual dispatch.

## Known limitations

An honest list, not hidden scope:

- **Rename detection**: diffing is by function name, so a rename plus no other change is reported as one deletion and one unrelated addition, losing continuity. Documented, not yet solved (would need body-similarity matching across name changes).
- **`needs_human_review` currently opens a draft PR with no diff** rather than a GitHub Issue, since there's nothing to review yet — a known, slightly awkward artifact.
- **`BackgroundTasks`, not a durable task queue.** A server restart mid-job silently loses that job. A real deployment should use Celery/Redis, SQS, or similar.
- **Multi-tenancy is a per-repo allowlist, not full tenant isolation.** `GITHUB_TOKEN` and the doc-resolution convention are still shared/global across every allowlisted repo. True per-tenant credential isolation (e.g. a GitHub App with per-installation tokens) remains unbuilt.
- **Doc-section extraction** works well for API-reference-style docs (headings, anchors) and reasonably for narrative prose (paragraph fallback), but has no understanding of documentation structures beyond regex-based heuristics.
- **CI's fast/slow split means a bug that only breaks at the Gemini/Pinecone API boundary won't be caught on the PR that introduces it** — only on the next scheduled run, up to a day later. An accepted cost/speed trade-off, not a free win.

## What this project demonstrates

Built step by step, debugging real issues as they surfaced rather than against pre-written working code:

- Model and package deprecations mid-build (`text-embedding-004`, `gemini-2.0-flash`, `google-generativeai` all retired during development) and the habit of pinning explicitly rather than trusting defaults
- Real external API constraints discovered only through integration testing (Pinecone rejecting `null` metadata values)
- Timing-attack-safe secret comparison (`hmac.compare_digest`), and the 401-vs-403 distinction applied consistently (bad signature vs. valid signature from a non-allowlisted repo)
- Idempotency and race-condition reasoning applied repeatedly: webhook redelivery, doc staleness between patch generation and patch application, and GitHub's own required-`sha`-on-write guard
- Pure/impure separation ("push impurity to the edges") applied consistently from chunking through generation through the webhook orchestrator, to keep tests fast and mock-free wherever possible — including lazy client construction, added specifically to let the fast CI suite collect and run with zero secrets present
- A live, unmocked end-to-end run — real push, real webhook delivery, real guardrail pipeline, real PR — verified by screenshot, not just by test suite
- A cost-aware CI split that paid for itself immediately: separating fast from slow tests surfaced two latent bugs (eager client construction, a mocked test with one unmocked collaborator) that had been silently masked by a populated local `.env` the entire time