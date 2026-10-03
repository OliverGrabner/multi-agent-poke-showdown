"""Cached lookups of Showdown's game data (names, types, descriptions) through the bridge."""

from __future__ import annotations

import re
from functools import cache

from pokerl import FORMAT
from pokerl.bridge import Bridge


def to_id(name: str) -> str:
    """Showdown's id form: 'Sludge Bomb' -> 'sludgebomb'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


class Dex:
    def __init__(self, bridge: Bridge, format: str = FORMAT):
        self.bridge = bridge
        self.format = format
        # One cache per Dex instance; entries never change during a run.
        self.lookup = cache(self._lookup)

    def _lookup(self, kind: str, id: str) -> dict | None:
        return self.bridge.call("describe", format=self.format, kind=kind, id=id)

    def move(self, name: str) -> dict:
        return self.lookup("move", to_id(name)) or {"name": name, "desc": ""}

    def ability(self, name: str) -> dict:
        return self.lookup("ability", to_id(name)) or {"name": name, "desc": ""}

    def item(self, name: str) -> dict:
        return self.lookup("item", to_id(name)) or {"name": name, "desc": ""}

    def types(self, species: str) -> list[str]:
        found = self.lookup("species", to_id(species))
        return found["types"] if found else []
