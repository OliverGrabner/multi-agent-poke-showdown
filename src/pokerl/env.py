"""Four-seat Multi Battle environment on top of the Showdown simulator."""

from __future__ import annotations

import copy
import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Any

from pokerl import FORMAT, SEATS
from pokerl.bridge import Bridge


def make_seed(*parts: object) -> str:
    """Derive a Showdown PRNG seed from arbitrary labels, e.g. make_seed("pilot", 17)."""
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()
    return f"sodium,{digest[:32]}"


class InvalidAction(ValueError):
    """An action that is not in the seat's legal list."""


@dataclass
class SeatView:
    """What one player sees: their own log channel, their request, their ally's team, legal actions."""

    seat: str
    request: dict[str, Any] | None
    ally: dict[str, Any] | None
    legal: list[dict[str, Any]]
    log: list[str]
    new_log: list[str]

    @property
    def needs_action(self) -> bool:
        return bool(self.legal)

    @property
    def choices(self) -> list[str]:
        return [action["choice"] for action in self.legal]


@dataclass
class StepResult:
    views: dict[str, SeatView]
    errors: dict[str, str] = field(default_factory=dict)
    ended: bool = False
    winning_side: str | None = None


_battle_ids = itertools.count()


class MultiBattleEnv:
    """One battle. Seats p1+p3 play against p2+p4; each seat controls one active Pokemon.

    `reset` starts a battle and `step` takes actions for any subset of the seats that are waiting.
    The turn advances once every waiting seat has acted. Actions are Showdown choice strings taken
    from `SeatView.legal`.
    """

    def __init__(self, bridge: Bridge, format: str = FORMAT):
        self.bridge = bridge
        self.format = format
        self.battle_id: str | None = None
        self.seed: str | None = None
        self.turn = 0
        self.request_state = ""  # "move", "switch" (mid-turn replacement) or "" between turns
        self.ended = False
        self.winning_side: str | None = None
        self.omniscient_log: list[str] = []
        self.views: dict[str, SeatView] = {}

    def reset(
        self, seed: str, teams: dict[str, str], names: dict[str, str] | None = None
    ) -> dict[str, SeatView]:
        self.close()
        self.battle_id = f"b{next(_battle_ids)}"
        self.seed = seed
        self.omniscient_log = []
        self.views = {seat: SeatView(seat, None, None, [], [], []) for seat in SEATS}
        snapshot = self.bridge.call(
            "start", battle_id=self.battle_id, format=self.format, seed=seed, teams=teams, names=names or {}
        )
        self._absorb(snapshot)
        return self.views

    def step(self, actions: dict[str, str]) -> StepResult:
        if self.ended:
            raise RuntimeError("Battle has ended")
        for seat, choice in actions.items():
            if choice not in self.views[seat].choices:
                raise InvalidAction(
                    f"{seat}: {choice!r} is not legal; options are {self.views[seat].choices}"
                )
        snapshot = self.bridge.call("choose", battle_id=self.battle_id, choices=actions)
        self._absorb(snapshot)
        return StepResult(self.views, snapshot["errors"], self.ended, self.winning_side)

    @property
    def pending(self) -> list[str]:
        return [seat for seat in SEATS if self.views[seat].needs_action]

    def save(self) -> dict[str, Any]:
        """A snapshot that `load` can resume, any number of times."""
        saved = self.bridge.call("save", battle_id=self.battle_id)
        return {
            "format": self.format,
            "seed": self.seed,
            "bridge": saved,
            "turn": self.turn,
            "omniscient_log": list(self.omniscient_log),
            "views": copy.deepcopy(self.views),
        }

    def load(self, snapshot: dict[str, Any]) -> dict[str, SeatView]:
        self.close()
        self.battle_id = f"b{next(_battle_ids)}"
        self.format = snapshot["format"]
        self.seed = snapshot["seed"]
        self.omniscient_log = list(snapshot["omniscient_log"])
        self.views = copy.deepcopy(snapshot["views"])
        bridge_state = snapshot["bridge"]
        result = self.bridge.call(
            "load", battle_id=self.battle_id, state=bridge_state["state"], cursor=bridge_state["cursor"]
        )
        self._absorb(result)
        return self.views

    def input_log(self) -> list[str]:
        """Showdown's own input log: seed plus every choice, enough to replay the battle exactly."""
        return self.bridge.call("input_log", battle_id=self.battle_id)

    def close(self) -> None:
        if self.battle_id is not None:
            self.bridge.call("close", battle_id=self.battle_id)
            self.battle_id = None

    def _absorb(self, snapshot: dict[str, Any]) -> None:
        self.turn = snapshot["turn"]
        self.request_state = snapshot["request_state"]
        self.ended = snapshot["ended"]
        self.winning_side = snapshot["winning_side"]
        self.omniscient_log.extend(snapshot["omniscient_log"])
        for seat, data in snapshot["seats"].items():
            view = self.views[seat]
            view.request = data["request"]
            view.ally = data["ally"]
            view.legal = data["legal"]
            view.new_log = data["new_log"]
            view.log.extend(data["new_log"])
