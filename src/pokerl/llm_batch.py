"""Battles where either side is model-backed (in any condition) or a pair of scripted bots.

Battles run in parallel threads, each with its own simulator, sharing one client per model.
Every finished battle is appended to the output file at once, and battles already in the file are
skipped, so an interrupted run (say, a Slurm job hitting its time limit) resumes where it stopped.
"""

from __future__ import annotations

import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pokerl import PLAYER, SIDES, TEAM_COLOR
from pokerl.bots import POLICIES, BotSide, make_policy
from pokerl.bridge import Bridge
from pokerl.codex_client import CODEX_MODELS, CodexClient
from pokerl.dex import Dex
from pokerl.env import MultiBattleEnv
from pokerl.llm_agent import CHOOSE_ONLY, TOOLS, LLMAgent
from pokerl.models import MODELS, BudgetExceeded, ChatClient
from pokerl.prompts import MODES, solo_prompt, system_prompt
from pokerl.runner import play
from pokerl.talk import SoloTeam, TalkingTeam
from pokerl.teams import BattleSpec

SIDE_MODES = (*MODES, "solo")  # "solo": one model controls both Pokémon


def models_in(player: str) -> list[str]:
    """'qwen' -> ['qwen']; 'a+b' -> ['a', 'b']; a bot policy -> []."""
    return [] if player in POLICIES else player.split("+")


def check_side(player: str, mode: str) -> None:
    """A bot policy, a model name, or two model names joined by '+' (one per teammate)."""
    if mode not in SIDE_MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose from {SIDE_MODES}")
    for model in models_in(player):
        if model not in MODELS and model not in CODEX_MODELS:
            known = sorted([*MODELS, *CODEX_MODELS])
            raise ValueError(f"{model!r} is neither a model ({known}) nor a bot ({sorted(POLICIES)})")
    if mode == "solo" and len(models_in(player)) > 1:
        raise ValueError("A solo side is one model controlling both Pokémon; give one model name")


def make_client(model: str) -> ChatClient | CodexClient:
    return CodexClient(model) if model in CODEX_MODELS else ChatClient(model)


def make_side(
    side: str, player: str, mode: str, clients: dict[str, ChatClient], dex: Dex, keep_talking: bool = True
):
    """The controller for one side. 'a+b' gives the first seat model a and the second model b."""
    first, second = SIDES[side]
    if player in POLICIES:
        return BotSide({first: make_policy(player), second: make_policy(player)})
    models = models_in(player)
    if mode == "solo":
        agent = LLMAgent(
            TEAM_COLOR[side],
            clients[models[0]].for_agent(),
            solo_prompt(PLAYER[first], PLAYER[second]),
            CHOOSE_ONLY,
        )
        return SoloTeam(agent, (first, second), dex, label=models[0])
    seat_models = {first: models[0], second: models[-1]}
    tools = CHOOSE_ONLY if mode == "no-talk" else TOOLS
    agents = {
        seat: LLMAgent(
            PLAYER[seat],
            clients[model].for_agent(),
            system_prompt(PLAYER[seat], PLAYER[other], mode, keep_talking=keep_talking),
            tools,
        )
        for (seat, model), other in zip(seat_models.items(), (second, first), strict=True)
    }
    return TalkingTeam(agents, dex, labels=seat_models, mode=mode, keep_talking=keep_talking)


def read_records(path: Path) -> list[dict]:
    """The latest record per battle in a battles.jsonl file; a finished battle beats a crash."""
    if not path.exists():
        return []
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "crash" not in record or "crash" in latest.get(record["battle_key"], {"crash": True}):
            latest[record["battle_key"]] = record
    return list(latest.values())


def run_llm_batch(
    specs: list[BattleSpec],
    side_a: str,
    side_b: str,
    out_path: Path,
    mode_a: str = "free",
    mode_b: str = "free",
    keep_talking: bool = True,
    workers: int = 1,
) -> list[dict]:
    """Play every spec not already finished in `out_path`; return all records in the file."""
    check_side(side_a, mode_a)
    check_side(side_b, mode_b)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    finished = {r["battle_key"] for r in read_records(out_path) if "crash" not in r}
    todo = [spec for spec in specs if spec.battle_key not in finished]
    clients = {model: make_client(model) for model in {*models_in(side_a), *models_in(side_b)}}

    local = threading.local()  # one simulator per thread
    bridges: list[Bridge] = []
    write_lock = threading.Lock()

    def play_one(spec: BattleSpec) -> None:
        if not hasattr(local, "env"):
            local.bridge = Bridge()
            bridges.append(local.bridge)
            local.env, local.dex = MultiBattleEnv(local.bridge), Dex(local.bridge)
        sides = {
            "p1p3": make_side("p1p3", side_a, mode_a, clients, local.dex, keep_talking),
            "p2p4": make_side("p2p4", side_b, mode_b, clients, local.dex, keep_talking),
        }
        try:
            record = play(local.env, spec, sides)
        except BudgetExceeded:
            raise
        except Exception:
            local.env.close()
            record = {
                "battle_key": spec.battle_key,
                "crash": traceback.format_exc(),
                "sides": {side: controller.record() for side, controller in sides.items()},
            }
        with write_lock, out_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record) + "\n")

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(play_one, spec) for spec in todo]
            for future in as_completed(futures):
                if isinstance(future.exception(), BudgetExceeded):
                    pool.shutdown(cancel_futures=True)
                    raise future.exception()
                future.result()
    finally:
        for bridge in bridges:
            bridge.close()
    return read_records(out_path)
