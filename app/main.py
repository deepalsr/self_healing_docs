# app/main.py

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Header
import hmac
import hashlib
from app.config import settings  # assumes config.py exposes a `settings` object

app = FastAPI()


def verify_signature(payload_body: bytes, signature_header: str) -> bool:
    """
    Verifies that the incoming webhook really came from GitHub,
    by recomputing the HMAC-SHA256 signature and comparing it
    in constant time.
    """
    if not signature_header:
        return False

    if not signature_header.startswith("sha256="):
        return False

    expected_signature = hmac.new(
        key=settings.WEBHOOK_SECRET.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256
    ).hexdigest()

    received_signature = signature_header.split("sha256=")[-1]

    return hmac.compare_digest(expected_signature, received_signature)


def process_webhook_event(payload: dict):
    """
    Runs in the background, after we've already responded 200 OK.
    Later steps will wire in: diffing -> chunking -> embedding
    -> retrieval -> generation -> guardrails -> PR creation.
    """
    ref = payload.get("ref", "unknown")
    print(f"[background] Processing push event on ref: {ref}")
    # TODO (future step): call diffing.run(payload) etc.


@app.post("/webhook/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str = Header(None),
    x_github_event: str = Header(None),
):
    payload_body = await request.body()

    if not verify_signature(payload_body, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="Invalid signature")

    if x_github_event != "push":
        return {"status": "ignored", "reason": f"event type '{x_github_event}' not handled"}

    payload = await request.json()

    if payload.get("ref") != "refs/heads/main":
        return {"status": "ignored", "reason": "not the main branch"}

    background_tasks.add_task(process_webhook_event, payload)

    return {"status": "accepted"}