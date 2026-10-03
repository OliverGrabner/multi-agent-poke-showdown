"""The message an agent reads each time it must act: what happened, the field, the teams, its options.

Exact information about the agent's own team and its partner's team comes from Showdown's request.
Everything about the opponents comes from what has been revealed in the log (see tracker.py).
"""

from __future__ import annotations

from pokerl import ALLY, FOES
from pokerl.dex import Dex
from pokerl.env import SeatView
from pokerl.narrate import STATS, STATUS, Narrator
from pokerl.tracker import Mon, Tracker


def effectiveness_text(multiplier: float | None) -> str:
    if multiplier is None:
        return ""
    if multiplier == 0:
        return "no effect (immune)"
    label = f"{multiplier:g}×"
    if multiplier > 1:
        return f"{label} (super effective)"
    if multiplier < 1:
        return f"{label} (resisted)"
    return label


def parse_details(details: str) -> tuple[str, int]:
    """'Salazzle, L85, F' -> ('Salazzle', 85)."""
    species, *rest = details.split(", ")
    level = next((int(part[1:]) for part in rest if part.startswith("L") and part[1:].isdigit()), 100)
    return species, level


def condition_text(condition: str) -> str:
    """Request conditions are exact: '120/254 par' -> '120/254 HP, paralyzed'; '0 fnt' -> 'fainted'."""
    hp, _, status = condition.partition(" ")
    if status == "fnt":
        return "fainted"
    return f"{hp} HP" + (f", {STATUS.get(status, status)}" if status else "")


class Observer:
    """Builds one agent's messages over a battle, remembering which terms it has already explained."""

    def __init__(self, seat: str, partner_name: str, dex: Dex):
        self.seat = seat
        self.partner_name = partner_name
        self.dex = dex
        self.tracker = Tracker()
        self.narrator = Narrator(seat, partner_name)
        self.cursor = 0
        self.explained: set[str] = set()
        self.to_explain: dict[str, str] = {}

    def observe(
        self, view: SeatView, header: str, options_title: str = "Your options"
    ) -> tuple[str, list[str]]:
        """Returns the message text and the Showdown choices behind options 1, 2, 3, ..."""
        new_lines = self.catch_up(view)
        self.to_explain = {}
        sections = [
            header,
            self.events_section(new_lines),
            self.field_section(view),
            self.teams_section(view),
            self.options_section(view, options_title),
            self.explanations_section(),
        ]
        return "\n\n".join(section for section in sections if section), view.choices

    def catch_up(self, view: SeatView) -> list[str]:
        """Read the log lines this observer has not seen yet into the tracker, and return them."""
        new_lines = view.log[self.cursor :]
        self.cursor = len(view.log)
        self.tracker.update(new_lines)
        return new_lines

    # Sections

    def events_section(self, new_lines: list[str]) -> str:
        events = self.narrator.narrate(new_lines)
        return "What happened:\n" + "\n".join(f"- {event}" for event in events) if events else ""

    def field_section(self, view: SeatView) -> str:
        tracker = self.tracker
        lines = ["On the field:"]
        if mine := active_mon(view.request):
            lines.append(f"- You: {self.own_active_line(mine, self.seat)}")
        if partners := active_mon(view.ally):
            lines.append(f"- {self.partner_name}: {self.own_active_line(partners, ALLY[self.seat])}")
        for seat in FOES[self.seat]:
            mon = tracker.active.get(seat)
            if mon and not mon.fainted:
                lines.append(f"- Opponent {seat}: {self.opponent_line(mon)}")
        conditions = []
        if tracker.weather:
            conditions.append(f"Weather: {tracker.weather}")
        if tracker.field:
            conditions.append(f"Field: {', '.join(tracker.field)}")
        side = "p1p3" if self.seat in ("p1", "p3") else "p2p4"
        other = "p2p4" if side == "p1p3" else "p1p3"
        if tracker.side_effects[side]:
            conditions.append(f"Your side: {', '.join(tracker.side_effects[side])}")
        if tracker.side_effects[other]:
            conditions.append(f"Opposing side: {', '.join(tracker.side_effects[other])}")
        lines.extend(f"- {condition}" for condition in conditions)
        return "\n".join(lines)

    def teams_section(self, view: SeatView) -> str:
        parts = []
        if view.request:
            parts.append("Your team:\n" + "\n".join(self.team_lines(view.request["side"])))
        if view.ally:
            parts.append(f"{self.partner_name}'s team:\n" + "\n".join(self.team_lines(view.ally)))
        opponents = [self.opponents_line(seat) for seat in FOES[self.seat]]
        parts.append("Opponents' Pokémon not on the field:\n" + "\n".join(opponents))
        return "\n\n".join(parts)

    def options_section(self, view: SeatView, title: str = "Your options") -> str:
        if not view.legal:
            return ""
        lines = [f"{number}. {self.option_text(action)}" for number, action in enumerate(view.legal, 1)]
        return f"{title}:\n" + "\n".join(lines)

    def explanations_section(self) -> str:
        if not self.to_explain:
            return ""
        self.explained.update(self.to_explain)
        lines = [f"- {name}: {desc}" for name, desc in self.to_explain.items()]
        return "What these do (each explained once):\n" + "\n".join(lines)

    # Lines

    def own_active_line(self, mon: dict, seat: str) -> str:
        """An active Pokémon on the agent's side: exact facts from the request plus tracked stat changes."""
        species, level = parse_details(mon["details"])
        tracked = self.tracker.active.get(seat)
        return ", ".join(
            filter(
                None,
                [self.species_text(species, level), condition_text(mon["condition"]), self.changes(tracked)],
            )
        )

    def opponent_line(self, mon: Mon) -> str:
        hp = "fainted" if mon.fainted else f"{mon.hp.split('/')[0]}% HP"
        facts = [self.species_text(mon.species, mon.level), hp]
        if mon.status:
            facts.append(STATUS.get(mon.status, mon.status))
        facts.append(self.changes(mon))
        if mon.moves:
            facts.append("moves seen: " + ", ".join(self.term("move", move) for move in mon.moves))
        if mon.ability:
            facts.append(f"ability: {self.term('ability', mon.ability)}")
        if mon.item is not None:
            facts.append(f"item: {self.term('item', mon.item)}" if mon.item else "no item")
        return ", ".join(fact for fact in facts if fact)

    def opponents_line(self, seat: str) -> str:
        """An opponent's benched or fainted Pokémon seen so far, and how many have not appeared yet."""
        seen = [mon for mon in self.tracker.mons.values() if mon.seat == seat]
        unseen = self.tracker.team_size.get(seat, 3) - len(seen)
        active = self.tracker.active.get(seat)
        entries = [self.opponent_line(mon) for mon in seen if mon is not active or mon.fainted]
        if unseen > 0:
            entries.append(f"{unseen} not seen yet")
        return f"- Opponent {seat}: " + ("; ".join(entries) or "none")

    def team_lines(self, side: dict) -> list[str]:
        lines = []
        for mon in side["pokemon"]:
            species, level = parse_details(mon["details"])
            moves = ", ".join(self.term("move", move) for move in mon["moves"])
            facts = [
                self.species_text(species, level),
                condition_text(mon["condition"]),
                f"ability: {self.term('ability', mon['ability'] or mon['baseAbility'])}",
                f"item: {self.term('item', mon['item'])}" if mon["item"] else "no item",
                f"moves: {moves}",
            ]
            marker = " (on the field)" if mon["active"] else ""
            lines.append(f"- {', '.join(facts)}{marker}")
        return lines

    def option_text(self, action: dict) -> str:
        if action["kind"] == "switch":
            species = action["species"]
            return f"Switch to {species} ({condition_text(action['condition'])})"
        if action["kind"] == "pass":
            return "Pass (nothing to do this turn)"
        move = self.term("move", action["move"])
        facts = [action["type"], action["category"]]
        if action["category"] != "Status":
            facts.append(f"{action['base_power']} power" if action["base_power"] else "variable power")
        if action["accuracy"]:
            facts.append(f"{action['accuracy']}% accuracy")
        if action["priority"]:
            facts.append(f"priority {action['priority']:+d}")
        if action["stab"] and action["category"] != "Status":
            facts.append("same-type bonus")
        return f"{move}{self.targets_text(action)} | {', '.join(facts)}"

    def option_label(self, action: dict) -> str:
        """A short name for an option, used when telling the partner what was chosen."""
        if action["kind"] == "switch":
            return f"Switch to {action['species']}"
        if action["kind"] == "pass":
            return "Pass"
        if action["target"]:
            return f"{action['move']} → {self.pokemon_name(action['target'])}"
        return action["move"]

    def targets_text(self, action: dict) -> str:
        hits = action["hits"]
        if not hits:
            return ""
        named = []
        for hit in hits:
            effect = effectiveness_text(hit["effectiveness"])
            named.append(f"{self.pokemon_name(hit)}, {effect}" if effect else self.pokemon_name(hit))
        if action["target"]:
            return " → " + named[0]
        return " → hits " + "; ".join(named)

    def pokemon_name(self, ref: dict) -> str:
        """A Pokémon on the field, named from this agent's point of view."""
        if ref["seat"] == self.seat:
            return f"your {ref['species']}"
        if ref["seat"] == ALLY[self.seat]:
            return f"{self.partner_name}'s {ref['species']} (your partner)"
        return f"opposing {ref['species']} ({ref['seat']})"

    # Helpers

    def species_text(self, species: str, level: int) -> str:
        types = "/".join(self.dex.types(species))
        return f"{species} L{level}" + (f" ({types})" if types else "")

    def changes(self, mon: Mon | None) -> str:
        if not mon:
            return ""
        boosts = [f"{STATS.get(stat, stat)} {n:+d}" for stat, n in mon.boosts.items() if n]
        return ", ".join(boosts + mon.effects)

    def term(self, kind: str, name: str) -> str:
        """The display name of a move, ability or item; queues its description the first time."""
        entry = getattr(self.dex, kind)(name)
        display = entry["name"]
        if display not in self.explained and entry.get("desc"):
            self.to_explain[display] = entry["desc"]
        return display


def active_mon(side_or_request: dict | None) -> dict | None:
    if not side_or_request:
        return None
    side = side_or_request.get("side", side_or_request)
    return next((mon for mon in side["pokemon"] if mon["active"]), None)
