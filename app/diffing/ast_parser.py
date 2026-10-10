# app/diffing/ast_parser.py

from importlib.resources import _common
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import difflib

RENAME_SIMILARITY_THRESHOLD = 0.8
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


def extract_definitions(source_code: str) -> dict:
    """
    Parses source code and extracts top-level function/class definitions.
    Returns: { name: {"signature": str, "body": str, "start_line": int, "end_line": int} }
    """
    tree = parser.parse(bytes(source_code, "utf8"))
    root = tree.root_node
    definitions = {}

    for node in root.children:
        if node.type in ("function_definition", "class_definition"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue

            name = source_code[name_node.start_byte:name_node.end_byte]
            full_text = source_code[node.start_byte:node.end_byte]

            # signature = everything up to the first ':' at the def line
            # (rough heuristic, good enough for v1)
            header_end = full_text.find(":\n") 
            signature = full_text[:header_end] if header_end != -1 else full_text.split("\n")[0]

            definitions[name] = {
                "signature": signature.strip(),
                "body": full_text.strip(),
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
            }

    return definitions

def detect_renames(deleted_names, added_names, old_defs, new_defs):
    """
    Greedily pair deleted/added functions whose bodies are similar enough
    to be the same function under a new name, rather than an unrelated
    deletion + addition. Pure and local — no network calls, no embeddings.

    Returns (renamed: list[dict], still_deleted: set[str], still_added: set[str])
    """
    candidates = []
    for old_name in deleted_names:
        old_body = old_defs[old_name]["body"]
        for new_name in added_names:
            new_body = new_defs[new_name]["body"]
            score = difflib.SequenceMatcher(None, old_body, new_body).ratio()
            if score >= RENAME_SIMILARITY_THRESHOLD:
                candidates.append((score, old_name, new_name))

    # Highest similarity first, so the best pairings get claimed first.
    candidates.sort(key=lambda c: c[0], reverse=True)

    claimed_old = set()
    claimed_new = set()
    renamed = []

    for score, old_name, new_name in candidates:
        if old_name in claimed_old or new_name in claimed_new:
            continue
        claimed_old.add(old_name)
        claimed_new.add(new_name)

        old_sig = old_defs[old_name]["signature"]
        new_sig = new_defs[new_name]["signature"]
        old_body = old_defs[old_name]["body"]
        new_body = new_defs[new_name]["body"]

        # Normalize out the name itself, so a pure rename doesn't get
        # misreported as a signature change just because the name differs.
        old_sig_normalized = old_sig.replace(old_name, "<fn>", 1)
        new_sig_normalized = new_sig.replace(new_name, "<fn>", 1)

        sig_changed = old_sig_normalized != new_sig_normalized
        body_changed = old_body != new_body
        if sig_changed and body_changed:
            change_type = "both"
        elif sig_changed:
            change_type = "signature"
        elif body_changed:
            change_type = "body_only"
        else:
            change_type = "none"

        renamed.append({
            "old_name": old_name,
            "new_name": new_name,
            "similarity": score,
            "old_signature": old_sig,
            "new_signature": new_sig,
            "old_body": old_body,
            "new_body": new_body,
            "change_type": change_type,
        })

    still_deleted = deleted_names - claimed_old
    still_added = added_names - claimed_new
    return renamed, still_deleted, still_added
def diff_definitions(old_code: str, new_code: str) -> dict:
    """
    Compares old vs new source code at the definition level.
    Returns a structured diff: { "added": [...], "deleted": [...], "modified": [...], "renamed": [...] }
    """
    old_defs = extract_definitions(old_code)
    new_defs = extract_definitions(new_code)

    old_names = set(old_defs.keys())
    new_names = set(new_defs.keys())

    added = new_names - old_names
    deleted = old_names - new_names
    common = old_names & new_names

    # Pull rename pairs out of added/deleted before finalizing those lists.
    renamed, deleted, added = detect_renames(deleted, added, old_defs, new_defs)

    modified = []
    for name in common:
        sig_changed = old_defs[name]["signature"] != new_defs[name]["signature"]
        body_changed = old_defs[name]["body"] != new_defs[name]["body"]

        if sig_changed or body_changed:
            if sig_changed and body_changed:
                change_type = "both"
            elif sig_changed:
                change_type = "signature"
            else:
                change_type = "body_only"

            modified.append({
                "name": name,
                "old_signature": old_defs[name]["signature"],
                "new_signature": new_defs[name]["signature"],
                "old_body": old_defs[name]["body"],
                "new_body": new_defs[name]["body"],
                "change_type": change_type,
            })

    return {
        "added": [{"name": n, **new_defs[n]} for n in added],
        "deleted": [{"name": n, **old_defs[n]} for n in deleted],
        "modified": modified,
        "renamed": renamed,
    }