from typing import Literal
from pydantic import BaseModel


class PatchChange(BaseModel):
    action: Literal["replace", "insert"]
    target_anchor: str
    old_text: str
    new_text: str


class DocPatch(BaseModel):
    changes: list[PatchChange]