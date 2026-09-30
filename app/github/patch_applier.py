# app/github/patch_applier.py

from dataclasses import dataclass
from app.models.schemas import DocPatch


@dataclass(frozen=True)
class ApplyResult:
    success: bool
    new_content: str | None = None
    reason: str | None = None


def apply_patch(file_content: str, patch: DocPatch) -> ApplyResult:
    """
    Applies a validated DocPatch to the FULL file's text, not just the
    extracted section. Re-checks grounding against the real file, since
    the file could have changed between when we fetched the section
    (Step 7's doc_snippet) and now -- same race-condition concern as
    last_updated_commit, just at a different point in the pipeline.
    """
    content = file_content

    for change in patch.changes:
        if change.action == "insert":
            # Insert at the end of the file. A more sophisticated version
            # could insert near related sections; out of scope for v1.
            content = content.rstrip() + "\n\n" + change.new_text.strip() + "\n"
            continue

        count = content.count(change.old_text)
        if count == 0:
            return ApplyResult(
                success=False,
                reason=(
                    f"old_text for '{change.target_anchor}' no longer found in the "
                    f"current file -- it may have changed since this patch was generated."
                ),
            )
        if count > 1:
            return ApplyResult(
                success=False,
                reason=f"old_text for '{change.target_anchor}' matches {count} places in the full file; refusing an ambiguous apply.",
            )

        content = content.replace(change.old_text, change.new_text)

    return ApplyResult(success=True, new_content=content)