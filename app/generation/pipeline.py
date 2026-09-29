# app/generation/pipeline.py

from dataclasses import dataclass
from app.generation.patch_generator import generate_patch, PatchParseError
from app.generation.judge import judge_patch
from app.guardrails import check_grounding, check_scope
from app.models.schemas import DocPatch

MAX_ATTEMPTS = 2  # hard cap -- see Concept 5's escalation policy


@dataclass(frozen=True)
class PipelineResult:
    status: str  # "approved" | "needs_human_review"
    patch: DocPatch | None
    reason: str | None = None
    attempts: int = 0


def generate_verified_patch(chunk: dict, retrieval_result: dict) -> PipelineResult:
    """
    Runs the full generate -> Layer1 -> Layer2 -> Layer3 pipeline, retrying
    with the rejection reason fed back into the prompt, up to MAX_ATTEMPTS.
    Never raises on a rejected patch -- always returns a clear result so the
    caller (Step 8's PR step) can decide what to do, including opening a
    flagged draft PR for human review.
    """
    existing_doc = chunk["metadata"].get("doc_snippet")
    allowed_anchors = {chunk["metadata"]["function_name"]}
    feedback = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            patch = generate_patch(chunk, retrieval_result, feedback=feedback)
        except PatchParseError as e:
            feedback = str(e)
            continue

        layer1 = check_grounding(patch, existing_doc)
        if not layer1.passed:
            feedback = layer1.reason
            continue

        layer2 = check_scope(patch, allowed_anchors)
        if not layer2.passed:
            feedback = layer2.reason
            continue

        layer3 = judge_patch(chunk, patch)
        if not layer3.approved:
            feedback = layer3.reason
            continue

        return PipelineResult(status="approved", patch=patch, attempts=attempt)

    return PipelineResult(
        status="needs_human_review",
        patch=None,
        reason=feedback,
        attempts=MAX_ATTEMPTS,
    )