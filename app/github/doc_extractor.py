# app/github/doc_extractor.py

import re


def extract_doc_section(doc_text: str, function_name: str) -> str | None:
    """
    Extracts documentation content related to a function, trying strategies
    from most reliable to most permissive:
      1. Explicit anchor tag: <!-- doc-anchor: function_name -->
      2. Markdown heading containing the function name
      3. A paragraph mentioning the function name (bold, code-formatted, or plain)
    Returns None only if none of these find anything -- a normal, expected case,
    not an error (most repos won't have adopted the anchor convention yet).
    """
    anchor_pattern = rf"<!--\s*doc-anchor:\s*{re.escape(function_name)}\s*-->(.*?)(?=<!--\s*doc-anchor:|\Z)"
    anchor_match = re.search(anchor_pattern, doc_text, re.DOTALL)
    if anchor_match:
        return anchor_match.group(1).strip()

    heading_pattern = rf"(#{{1,6}}\s+.*\b{re.escape(function_name)}\b.*\n)(.*?)(?=\n#{{1,6}}\s+|\Z)"
    heading_match = re.search(heading_pattern, doc_text, re.DOTALL)
    if heading_match:
        return (heading_match.group(1) + heading_match.group(2)).strip()

    # Fallback: find a paragraph mentioning the function name, in any styling
    # (`func_name`, **func_name**, or plain text), using a word boundary (\b)
    # so "ingest" doesn't false-positive match inside "ingestion" or "ingest.py".
    paragraphs = re.split(r"\n\s*\n", doc_text)
    name_mention_pattern = re.compile(
        rf"(\*\*)?`?\b{re.escape(function_name)}\b`?(\*\*)?", re.IGNORECASE
    )
    for paragraph in paragraphs:
        if name_mention_pattern.search(paragraph):
            return paragraph.strip()

    return None