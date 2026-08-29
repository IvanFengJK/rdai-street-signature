"""Access to the Stage 0 schema.

Column names in this dataset are not guessable, so nothing outside this module
should hard-code them. Load them from schema.json instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SCHEMA = _REPO_ROOT / "schema.json"


def load_schema(path: str | Path = _DEFAULT_SCHEMA) -> dict[str, Any]:
    """Read schema.json."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def role(name: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the schema entry for a role: city, sequence_id, greenery,
    building_density or road_type."""
    schema = schema or load_schema()
    roles = schema["roles"]
    if name not in roles:
        raise KeyError(f"unknown role {name!r}; known roles: {sorted(roles)}")
    return roles[name]


def column_for(name: str, schema: dict[str, Any] | None = None) -> str:
    """Return the real column name backing a role.

    Raises for roles that have no shipped column and must be computed
    (building_density), so a derived field can never be silently mistaken for a
    dataset field.
    """
    entry = role(name, schema)
    if entry.get("derived") or entry.get("column") is None:
        raise ValueError(
            f"{name!r} is derived, not a dataset column. "
            f"Compute it as: {entry.get('expression')!r} "
            f"(from {entry['file']}), and call it "
            f"{entry.get('proposed_name')!r}."
        )
    return entry["column"]


def forbidden_columns(schema: dict[str, Any] | None = None) -> list[str]:
    """Perception-score columns. CLAUDE.md rule 2: never used."""
    schema = schema or load_schema()
    return list(schema["forbidden"]["columns"])
