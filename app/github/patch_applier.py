# app/github/patch_applier.py

from dataclasses import dataclass
from app.models.schemas import DocPatch


@dataclass(frozen=True)
class ApplyResult:
    success: bool
    new_content: str | None = None
    reason: str | None = None


def _insert_after_reference(content: str, insert_reference: str, new_text: str) -> str | None:
    """
    Pure: finds insert_reference's paragraph in content and inserts new_text
    right after it. Returns None if insert_reference isn't actually present
    (e.g. it changed since retrieval ran) -- caller falls back to end-of-file.
    """
    idx = content.find(insert_reference)
    if idx == -1:
        return None

    # Find the end of the paragraph containing the reference -- the next
    # blank line after it, or end of file if there isn't one.
    paragraph_end = content.find("\n\n", idx)
    if paragraph_end == -1:
        insertion_point = len(content)
    else:
        insertion_point = paragraph_end

    return content[:insertion_point] + "\n\n" + new_text.strip() + content[insertion_point:]


def apply_patch(file_content: str, patch: DocPatch, insert_reference: str | None = None) -> ApplyResult:
    """
    Applies a validated DocPatch to the FULL file's text. Re-checks grounding
    against the real file (see module docstring reasoning from before).

    insert_reference, when provided, is the doc text of a similar, already-
    documented function (from Tier 2 retrieval) -- new content is inserted
    near it instead of blindly at the end of the file. Falls back to
    end-of-file when no reference is given or it can't be found in the
    current content.
    """
    content = file_content

    for change in patch.changes:
        if change.action == "insert":
            new_text = change.new_text.strip()
            placed = None
            if insert_reference:
                placed = _insert_after_reference(content, insert_reference, new_text)
            content = placed if placed is not None else (content.rstrip() + "\n\n" + new_text + "\n")
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