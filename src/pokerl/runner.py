"""Play battles between two side controllers, log every battle as one JSON line, summarize results."""

from __future__ import annotations

import json
import math
import time
import traceback
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from pokerl import SIDE_OF
from pokerl.bots import BotSide, make_policy
from pokerl.bridge import Bridge
from pokerl.env import MultiBattleEnv
from pokerl.teams import BattleSpec

MAX_TURNS = 300
MAX_REJECTIONS = 20


class Controller(Protocol):
    """Decides the actions for the seats of one side (scripted bots, or talking agents)."""

    def names(self) -> dict[str, str]: ...

    def decide(
        self, env: MultiBattleEnv, seats: list[str], step: int, rejected: dict[str, str]
    ) -> dict[str, str]: ...

    def record(self) -> dict: ...


def bot_sides(seat_policies: dict[str, str]) -> dict[str, BotSide]:
    """{'p1': 'maxpower', ...} -> one BotSide per side."""
    return {
        side: BotSide(
            {seat: make_policy(seat_policies[seat]) for seat in seat_policies if SIDE_OF[seat] == side}
        )
        for side in ("p1p3", "p2p4")
    }


def play(
    env: MultiBattleEnv, spec: BattleSpec, sides: dict[str, Controller], max_turns: int = MAX_TURNS
) -> dict:
    """Play one battle to the end and return its log record."""
    start = time.perf_counter()
    names = {seat: name for controller in sides.values() for seat, name in controller.names().items()}
    env.reset(spec.seed, spec.teams, names)
    steps, rejections, step = [], [], 0
    rejected: dict[str, str] = {}
    truncated = False
    while not env.ended:
        if env.turn > max_turns:
            truncated = True
            break
        pending = env.pending
        if not pending:
            raise RuntimeError(f"No seat can act at turn {env.turn}, but the battle has not ended")
        turn = env.turn
        actions = {}
        for side, controller in sides.items():
            seats = [seat for seat in pending if SIDE_OF[seat] == side]
            if seats:
                actions.update(controller.decide(env, seats, step, rejected))
        result = env.step(actions)
        rejected = result.errors
        steps.append({"turn": turn, "choices": actions, "rejected": sorted(result.errors)})
        for seat, message in result.errors.items():
            rejections.append({"turn": turn, "seat": seat, "choice": actions[seat], "error": message})
        if len(rejections) > MAX_REJECTIONS:
            raise RuntimeError(f"Too many rejected choices: {rejections[-3:]}")
        step += 1
    record = {
        "battle_key": spec.battle_key,
        **{key: value for key, value in asdict(spec).items() if key not in ("battle_key", "teams")},
        "players": names,
        "sides": {side: controller.record() for side, controller in sides.items()},
        "winning_side": env.winning_side,
        "truncated": truncated,
        "turns": env.turn,
        "steps": steps,
        "rejections": rejections,
        "omniscient_log": env.omniscient_log,
        "input_log": env.input_log(),
        "seconds": round(time.perf_counter() - start, 4),
    }
    env.close()
    return record


_worker_bridge: Bridge | None = None


def _init_worker() -> None:
    global _worker_bridge
    _worker_bridge = Bridge()


def _play_spec(spec: BattleSpec, seat_policies: dict[str, str], max_turns: int) -> dict:
    env = MultiBattleEnv(_worker_bridge)
    try:
        return play(env, spec, bot_sides(seat_policies), max_turns)
    except Exception:
        return {
            "battle_key": spec.battle_key,
            "seed": spec.seed,
            "team_ids": spec.team_ids,
            "players": seat_policies,
            "crash": traceback.format_exc(),
        }


def run_batch(
    specs: Iterable[BattleSpec],
    seat_policies: dict[str, str],
    out_path: Path,
    workers: int = 1,
    max_turns: int = MAX_TURNS,
) -> list[dict]:
    """Play every spec, appending one JSON line per battle to `out_path` as results arrive."""
    specs = list(specs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_path.open("a", encoding="utf-8") as out:
        if workers <= 1:
            _init_worker()
            try:
                for spec in specs:
                    record = _play_spec(spec, seat_policies, max_turns)
                    out.write(json.dumps(record) + "\n")
                    records.append(record)
            finally:
                _worker_bridge.close()
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
                chunk = max(1, len(specs) // (workers * 8))
                results = pool.map(
                    _play_spec, specs, [seat_policies] * len(specs), [max_turns] * len(specs), chunksize=chunk
                )
                for record in results:
                    out.write(json.dumps(record) + "\n")
                    records.append(record)
    return records


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def summarize(records: list[dict]) -> dict:
    """Win rate of the p1+p3 side, with a 95% Wilson interval; draws and truncations count as half."""
    finished = [r for r in records if "crash" not in r]
    crashes = len(records) - len(finished)
    wins = sum(r["winning_side"] == "p1p3" for r in finished)
    draws = sum(r["winning_side"] is None for r in finished)
    score = wins + draws / 2
    n = len(finished)
    low, high = wilson(score, n)
    turns = [r["turns"] for r in finished]
    return {
        "battles": len(records),
        "crashes": crashes,
        "truncated": sum(r["truncated"] for r in finished),
        "draws": draws,
        "p1p3_wins": wins,
        "p1p3_win_rate": round(score / n, 4) if n else None,
        "p1p3_win_rate_95ci": [round(low, 4), round(high, 4)],
        "mean_turns": round(sum(turns) / n, 2) if n else None,
        "rejected_choices": sum(len(r["rejections"]) for r in finished),
    }
