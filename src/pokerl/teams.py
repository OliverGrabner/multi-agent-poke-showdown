"""Fixed team pools and battle specs (seed + team assignment, with optional side-swapped mirrors).

Two kinds of pool: random-battle teams (one per player, drawn at random for each battle), and
strategy teams (a doubles team split between the two partners, played in fixed matchups).
"""

from __future__ import annotations

import itertools
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


def build_strategy_pool(bridge: Bridge, path: Path) -> dict:
    """Strategy teams from a text file of `=== Name | first` / `=== Name | second` blocks.

    Each block is one partner's three Pokémon in Showdown's export format, lead first; `#` lines
    are comments. Both halves of a team go to the same side, one per partner.
    """
    teams: dict[str, dict] = {}
    for block in Path(path).read_text(encoding="utf-8").split("\n=== ")[1:]:
        header, _, body = block.partition("\n")
        name, _, half = (part.strip() for part in header.partition("|"))
        text = "\n".join(line for line in body.splitlines() if not line.startswith("#"))
        team = teams.setdefault(name, {"id": name, "halves": {}})
        team["halves"][half] = bridge.call("pack", text=text)
    return {"kind": "strategy", "label": Path(path).stem, "teams": list(teams.values())}


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

    A strategy pool plays mirror matches, one team per seed in turn (see matchup_specs).

    Swapping teams between sides while seat policies stay fixed cancels team and seed luck
    within each pair, so comparisons need fewer battles.
    """
    if pool.get("kind") == "strategy":
        return matchup_specs(pool, pairs, label, mirror)
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


def matchup_specs(
    pool: dict, pairs: int, label: str, mirror: bool = True, same_team: bool = True
) -> list[BattleSpec]:
    """Seed i plays matchup i, cycling through the matchups.

    With `same_team`, both sides get the same team (a mirror match, even by construction): bot
    play showed matchups between different archetypes are lopsided (e.g. sun beat rain 39 to 1).
    Without it, every pair of different teams is played, sides swapped in the mirror battle.
    A mirror match has nothing to swap, so it is played once per seed.
    A team goes to one side: its first half to p1 or p2, its second half to that player's partner.
    """
    teams = pool["teams"]
    matchups = [(t, t) for t in teams] if same_team else list(itertools.combinations(teams, 2))
    specs = []
    for i in range(pairs):
        seed = make_seed("battle", label, i)
        blue, red = matchups[i % len(matchups)]
        sides = [(blue, red), (red, blue)] if mirror and not same_team else [(blue, red)]
        for mirrored, (a, b) in enumerate(sides):
            halves = {
                "p1": (a, "first"), "p3": (a, "second"), "p2": (b, "first"), "p4": (b, "second")
            }  # fmt: skip
            specs.append(
                BattleSpec(
                    battle_key=f"{label}-{i:05d}{'m' if mirrored else ''}",
                    seed=seed,
                    team_ids={seat: f"{team['id']} ({half})" for seat, (team, half) in halves.items()},
                    teams={seat: team["halves"][half]["packed"] for seat, (team, half) in halves.items()},
                    pair=i,
                    mirrored=bool(mirrored),
                )
            )
    return specs
