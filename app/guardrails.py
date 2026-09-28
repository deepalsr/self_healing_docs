from dataclasses import dataclass
from app.models.schemas import DocPatch


@dataclass(frozen=True)
class GuardrailResult:
    passed: bool
    reason: str | None = None  # fed back to the generator on retry


def _norm(text: str) -> str:
    # Normalize ONLY line endings. Loosening more (collapsing whitespace, etc.)
    # would let near-miss hallucinations slip through.
    return text.replace("\r\n", "\n")


def check_grounding(patch: DocPatch, existing_doc: str | None) -> GuardrailResult:
    """Layer 1: pure, deterministic, no LLM."""
    if not patch.changes:
        return GuardrailResult(False, "Patch contains no changes.")

    doc = _norm(existing_doc) if existing_doc else None

    for i, change in enumerate(patch.changes):
        label = f"change #{i + 1} ({change.target_anchor})"

        if not change.new_text.strip():
            return GuardrailResult(False, f"{label}: new_text is empty.")

        if change.action == "insert":
            if doc is not None:
                return GuardrailResult(
                    False, f"{label}: 'insert' used but documentation already exists; use 'replace'."
                )
            if change.old_text != "":
                return GuardrailResult(False, f"{label}: 'insert' must have empty old_text.")
            continue

        # action == "replace"
        if doc is None:
            return GuardrailResult(
                False, f"{label}: 'replace' used but there is no existing documentation to replace."
            )

        old = _norm(change.old_text)
        if not old.strip():
            return GuardrailResult(False, f"{label}: old_text is empty for a 'replace'.")

        count = doc.count(old)
        if count == 0:
            return GuardrailResult(
                False, f"{label}: old_text not found verbatim in the existing documentation."
            )
        if count > 1:
            return GuardrailResult(
                False, f"{label}: old_text matches {count} places; it must be unique. Include more surrounding text."
            )
        if old == _norm(change.new_text):
            return GuardrailResult(False, f"{label}: new_text is identical to old_text (no-op).")

    return GuardrailResult(True)