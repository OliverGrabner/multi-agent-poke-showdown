"""Fixed team pools and battle specs (seed + team assignment, with optional side-swapped mirrors)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from pokerl import TEAM_FORMAT
from pokerl.bridge import Bridge
from pokerl.env import make_seed

DEFAULT_POOL = Path(__file__).resolve().parents[2] / "data" / "teams" / "pool-v1.json"


def generate_pool(bridge: Bridge, size: int, label: str, format: str = TEAM_FORMAT) -> dict:
    """Generate `size` random-battle teams from seeds derived from `label`."""
    seeds = [make_seed("team", label, i) for i in range(size)]
    generated = bridge.call("teams", format=format, seeds=seeds)
    version = bridge.call("ping")["showdown"]
    teams = [{"id": f"{label}-{i:03d}", **team} for i, team in enumerate(generated)]
    return {"format": format, "label": label, "showdown_version": version, "teams": teams}


def load_pool(path: Path = DEFAULT_POOL) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class BattleSpec:
    """Everything needed to start one battle. `pair` links a battle to its side-swapped mirror."""

    battle_key: str
    seed: str
    team_ids: dict[str, str]
    teams: dict[str, str]
    pair: int
    mirrored: bool


def draw_teams(pool: dict, rng: random.Random) -> list[dict]:
    """Four teams with no species shared across the battle (Species Clause spirit)."""
    for _ in range(1000):
        picked = rng.sample(pool["teams"], 4)
        species = [s for team in picked for s in team["species"]]
        if len(species) == len(set(species)):
            return picked
    raise ValueError("Team pool too small to draw four teams with distinct species")


def make_specs(pool: dict, pairs: int, label: str, mirror: bool = True) -> list[BattleSpec]:
    """`pairs` seeds; with `mirror`, each seed is played twice with the two sides' teams swapped.

    Swapping teams between sides while seat policies stay fixed cancels team and seed luck
    within each pair, so comparisons need fewer battles.
    """
    specs = []
    for i in range(pairs):
        seed = make_seed("battle", label, i)
        rng = random.Random(seed)
        t1, t2, t3, t4 = draw_teams(pool, rng)
        layouts = [{"p1": t1, "p2": t2, "p3": t3, "p4": t4}]
        if mirror:
            layouts.append({"p1": t2, "p2": t1, "p3": t4, "p4": t3})
        for mirrored, layout in enumerate(layouts):
            specs.append(
                BattleSpec(
                    battle_key=f"{label}-{i:05d}{'m' if mirrored else ''}",
                    seed=seed,
                    team_ids={seat: team["id"] for seat, team in layout.items()},
                    teams={seat: team["packed"] for seat, team in layout.items()},
                    pair=i,
                    mirrored=bool(mirrored),
                )
            )
    return specs
