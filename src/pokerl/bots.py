"""Scripted policies that can fill any seat."""

from __future__ import annotations

import random
from typing import Protocol

from pokerl import FOES, PLAYER
from pokerl.env import SeatView


class Policy(Protocol):
    name: str

    def act(self, view: SeatView, rng: random.Random) -> str: ...


class RandomBot:
    """Uniform over every legal action, including targeting its ally."""

    name = "random"

    def act(self, view: SeatView, rng: random.Random) -> str:
        return rng.choice(view.choices)


def move_score(action: dict) -> float | None:
    """Rough expected damage to foes minus damage to the ally; None for non-attacks."""
    if action["kind"] != "move" or action["category"] == "Status":
        return None
    foes = [hit for hit in action["hits"] if not hit["is_ally"] and not hit["fainted"]]
    allies = [hit for hit in action["hits"] if hit["is_ally"] and not hit["fainted"]]
    if not foes:
        return None
    # Moves like Low Kick or Gyro Ball have no fixed power; assume an average one.
    power = action["base_power"] or 60
    accuracy = (action["accuracy"] or 100) / 100
    stab = 1.5 if action["stab"] else 1.0
    spread = 0.75 if len(action["hits"]) > 1 else 1.0
    multiplier = sum(hit["effectiveness"] or 0 for hit in foes) - sum(
        hit["effectiveness"] or 0 for hit in allies
    )
    return power * accuracy * stab * spread * multiplier


class MaxPowerBot:
    """Picks the attack with the highest power x accuracy x STAB x type effectiveness against a foe.

    Never switches voluntarily; on a forced switch it picks a random replacement.
    """

    name = "maxpower"

    def act(self, view: SeatView, rng: random.Random) -> str:
        scored = [
            (score, action["choice"]) for action in view.legal if (score := move_score(action)) is not None
        ]
        best = max((score for score, _ in scored), default=0)
        if best > 0:
            return rng.choice([choice for score, choice in scored if score == best])
        # No useful attack: any move that does not target the ally, else whatever is legal.
        fallback = [
            action["choice"]
            for action in view.legal
            if action["kind"] == "move"
            and not (action["target"] and action["target"]["seat"] not in FOES[view.seat])
        ]
        return rng.choice(fallback or view.choices)


POLICIES: dict[str, type] = {"random": RandomBot, "maxpower": MaxPowerBot}


def make_policy(name: str) -> Policy:
    try:
        return POLICIES[name]()
    except KeyError:
        raise ValueError(f"Unknown policy {name!r}; choose from {sorted(POLICIES)}") from None


def step_rng(seed: str, seat: str, step: int) -> random.Random:
    """Bot randomness depends only on (battle seed, seat, step), so saved battles resume identically."""
    return random.Random(f"{seed}:{seat}:{step}")


class BotSide:
    """A side whose two seats are each played by a scripted policy, with no talking."""

    def __init__(self, policies: dict[str, Policy]):
        self.policies = policies

    def names(self) -> dict[str, str]:
        return {seat: f"{PLAYER[seat]} - {policy.name}" for seat, policy in self.policies.items()}

    def decide(self, env, seats: list[str], step: int, rejected: dict[str, str]) -> dict[str, str]:
        return {
            seat: self.policies[seat].act(env.views[seat], step_rng(env.seed, seat, step)) for seat in seats
        }

    def record(self) -> dict:
        return {"kind": "bots", "policies": {seat: policy.name for seat, policy in self.policies.items()}}
