# app/diffing/ast_parser.py

from importlib.resources import _common
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

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


def diff_definitions(old_code: str, new_code: str) -> dict:
    """
    Compares old vs new source code at the definition level.
    Returns a structured diff: { "added": [...], "deleted": [...], "modified": [...] }
    """
    old_defs = extract_definitions(old_code)
    new_defs = extract_definitions(new_code)

    old_names = set(old_defs.keys())
    new_names = set(new_defs.keys())

    added = new_names - old_names
    deleted = old_names - new_names
    common = old_names & new_names

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
    }