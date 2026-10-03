"""Public battle state rebuilt from one player's log channel.

Showdown's log is a list of lines such as `|move|p1a: Salazzle|Sludge Bomb|p4b: Ceruledge`.
The tracker reads them in order and remembers what that player has seen: HP, status, stat
changes, revealed moves, items and abilities, and field effects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pokerl import SIDE_OF

PREFIXES = ("move: ", "ability: ", "item: ")


@dataclass
class Line:
    kind: str
    args: list[str]
    tags: dict[
        str, str
    ]  # "[from] item: Leftovers" -> {"from": "item: Leftovers"}; "[silent]" -> {"silent": ""}


def parse(raw: str) -> Line | None:
    if not raw.startswith("|"):
        return None
    kind, *rest = raw[1:].split("|")
    args, tags = [], {}
    for part in rest:
        tag = re.match(r"\[(\w+)\]\s*(.*)", part)
        if tag:
            tags[tag.group(1)] = tag.group(2)
        else:
            args.append(part)
    return Line(kind, args, tags)


def split_ident(ident: str) -> tuple[str, str]:
    """'p1a: Salazzle' -> ('p1', 'Salazzle'). Also accepts side idents like 'p1: Alex'."""
    position, _, name = ident.partition(": ")
    return position[:2], name


def clean(effect: str) -> str:
    """'move: Leech Seed' -> 'Leech Seed'."""
    for prefix in PREFIXES:
        if effect.startswith(prefix):
            return effect[len(prefix) :]
    return effect


@dataclass
class Mon:
    """What one player has seen of one Pokémon."""

    seat: str
    name: str
    species: str
    level: int
    hp: str = "100/100"  # exact for the viewer's own Pokémon, out of 100 for everyone else
    status: str = ""
    fainted: bool = False
    boosts: dict[str, int] = field(default_factory=dict)
    effects: list[str] = field(default_factory=list)
    moves: list[str] = field(default_factory=list)  # revealed moves, in order of first use
    item: str | None = None  # None = not revealed yet, "" = no item (used up or removed)
    ability: str | None = None


class Tracker:
    def __init__(self) -> None:
        self.mons: dict[tuple[str, str], Mon] = {}
        self.active: dict[str, Mon] = {}
        self.turn = 0
        self.team_size: dict[str, int] = {}
        self.weather = ""
        self.field: list[str] = []
        self.side_effects: dict[str, list[str]] = {"p1p3": [], "p2p4": []}

    def update(self, raw_lines: list[str]) -> None:
        for raw in raw_lines:
            line = parse(raw)
            if line and line.kind:
                self.apply(line)

    def mon(self, ident: str) -> Mon | None:
        return self.mons.get(split_ident(ident))

    def apply(self, line: Line) -> None:
        args = line.args
        match line.kind:
            case "turn":
                self.turn = int(args[0])
            case "teamsize":
                self.team_size[args[0]] = int(args[1])
            case "switch" | "drag" | "replace":
                self.switch_in(args[0], args[1], args[2] if len(args) > 2 else "")
            case "detailschange" | "-formechange":
                if mon := self.mon(args[0]):
                    mon.species = args[1].split(", ")[0]
            case "-damage" | "-heal" | "-sethp":
                if mon := self.mon(args[0]):
                    self.set_condition(mon, args[1])
            case "faint":
                if mon := self.mon(args[0]):
                    mon.fainted, mon.hp = True, "0"
            case "-status":
                if mon := self.mon(args[0]):
                    mon.status = args[1]
            case "-curestatus":
                if mon := self.mon(args[0]):
                    mon.status = ""
            case "-boost" | "-unboost" | "-setboost":
                if mon := self.mon(args[0]):
                    amount = int(args[2])
                    if line.kind == "-setboost":
                        mon.boosts[args[1]] = amount
                    else:
                        sign = 1 if line.kind == "-boost" else -1
                        mon.boosts[args[1]] = mon.boosts.get(args[1], 0) + sign * amount
            case "-clearboost":
                if mon := self.mon(args[0]):
                    mon.boosts.clear()
            case "-clearnegativeboost":
                if mon := self.mon(args[0]):
                    mon.boosts = {stat: n for stat, n in mon.boosts.items() if n > 0}
            case "-clearallboost":
                for mon in self.active.values():
                    mon.boosts.clear()
            case "-start":
                if mon := self.mon(args[0]):
                    effect = clean(args[1]) if args[1] != "typechange" else f"type changed to {args[2]}"
                    if effect not in mon.effects:
                        mon.effects.append(effect)
            case "-end":
                if (mon := self.mon(args[0])) and clean(args[1]) in mon.effects:
                    mon.effects.remove(clean(args[1]))
            case "-weather":
                if "upkeep" not in line.tags:
                    self.weather = "" if args[0] == "none" else args[0]
            case "-fieldstart":
                self.field.append(clean(args[0]))
            case "-fieldend":
                if clean(args[0]) in self.field:
                    self.field.remove(clean(args[0]))
            case "-sidestart":
                self.side_effects[SIDE_OF[split_ident(args[0])[0]]].append(clean(args[1]))
            case "-sideend":
                effects = self.side_effects[SIDE_OF[split_ident(args[0])[0]]]
                if clean(args[1]) in effects:
                    effects.remove(clean(args[1]))
            case "move":
                mon = self.mon(args[0])
                if mon and "from" not in line.tags and args[1] not in mon.moves:
                    mon.moves.append(args[1])
            case "-item":
                if mon := self.mon(args[0]):
                    mon.item = args[1]
            case "-enditem":
                if mon := self.mon(args[0]):
                    mon.item = ""
            case "-ability":
                if mon := self.mon(args[0]):
                    mon.ability = args[1]
            case "-activate":
                if len(args) > 1:
                    self.reveal(args[0], args[1])
        if "from" in line.tags and args:
            # "[from] item: Leftovers" belongs to the [of] Pokémon when given, else to the line's subject.
            self.reveal(line.tags.get("of") or args[0], line.tags["from"])

    def reveal(self, ident: str, effect: str) -> None:
        mon = self.mon(ident)
        if not mon:
            return
        if effect.startswith("ability: "):
            mon.ability = clean(effect)
        elif effect.startswith("item: ") and mon.item is None:
            mon.item = clean(effect)

    def switch_in(self, ident: str, details: str, condition: str) -> None:
        seat, name = split_ident(ident)
        species, *rest = details.split(", ")
        level = next((int(part[1:]) for part in rest if re.fullmatch(r"L\d+", part)), 100)
        mon = self.mons.setdefault((seat, name), Mon(seat, name, species, level))
        if previous := self.active.get(seat):
            previous.boosts.clear()
            previous.effects.clear()
        self.active[seat] = mon
        if condition:
            self.set_condition(mon, condition)

    @staticmethod
    def set_condition(mon: Mon, condition: str) -> None:
        """'71/100 par' -> hp 71/100, paralyzed; '0 fnt' -> fainted."""
        hp, _, status = condition.partition(" ")
        mon.hp = hp
        if status == "fnt":
            mon.fainted = True
        else:
            mon.status = status
