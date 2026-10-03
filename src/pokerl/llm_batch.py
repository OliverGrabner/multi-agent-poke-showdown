"""Battles where either side is an LLM team or a pair of scripted bots, and a summary of the talk."""

from __future__ import annotations

import json
import traceback
from collections import Counter
from pathlib import Path

from pokerl import PLAYER, SIDES
from pokerl.bots import POLICIES, BotSide, make_policy
from pokerl.bridge import Bridge
from pokerl.dex import Dex
from pokerl.env import MultiBattleEnv
from pokerl.llm_agent import LLMAgent
from pokerl.models import MODELS, BudgetExceeded, ChatClient, output_tokens
from pokerl.runner import play
from pokerl.talk import MAX_MESSAGES, TalkingTeam
from pokerl.teams import BattleSpec


def check_player(player: str) -> None:
    """A bot policy, a model name, or two model names joined by '+' (one per teammate)."""
    if player in POLICIES:
        return
    for model in player.split("+"):
        if model not in MODELS:
            raise ValueError(
                f"{model!r} is neither a model ({sorted(MODELS)}) nor a bot ({sorted(POLICIES)})"
            )


def make_side(side: str, player: str, clients: dict[str, ChatClient], dex: Dex, keep_talking: bool = True):
    """A BotSide for a bot policy, else a TalkingTeam. 'a+b' gives the first seat model a, the second b."""
    first, second = SIDES[side]
    if player in POLICIES:
        return BotSide({first: make_policy(player), second: make_policy(player)})
    models = player.split("+")
    if len(models) == 1:
        models = models * 2
    seat_models = {first: models[0], second: models[1]}
    for model in models:
        if model not in clients:
            clients[model] = ChatClient(model)
    agents = {
        first: LLMAgent(PLAYER[first], PLAYER[second], clients[seat_models[first]], keep_talking),
        second: LLMAgent(PLAYER[second], PLAYER[first], clients[seat_models[second]], keep_talking),
    }
    return TalkingTeam(agents, dex, labels=seat_models, keep_talking=keep_talking)


def run_llm_batch(
    specs: list[BattleSpec], side_a: str, side_b: str, out_path: Path, keep_talking: bool = True
) -> list[dict]:
    """Play every spec in order, appending one JSON line per battle. Stops at once if a budget runs out."""
    check_player(side_a)
    check_player(side_b)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clients: dict[str, ChatClient] = {}
    records = []
    with Bridge() as bridge, out_path.open("a", encoding="utf-8") as out:
        env = MultiBattleEnv(bridge)
        dex = Dex(bridge)
        for spec in specs:
            sides = {
                "p1p3": make_side("p1p3", side_a, clients, dex, keep_talking),
                "p2p4": make_side("p2p4", side_b, clients, dex, keep_talking),
            }
            try:
                record = play(env, spec, sides)
            except BudgetExceeded:
                raise
            except Exception:
                record = {
                    "battle_key": spec.battle_key,
                    "crash": traceback.format_exc(),
                    "sides": {side: controller.record() for side, controller in sides.items()},
                }
            out.write(json.dumps(record) + "\n")
            out.flush()
            records.append(record)
    return records


def summarize_talk(records: list[dict], side: str) -> dict:
    """Model calls, cost, output validity and how much one side's teammates talked."""
    counts = Counter()
    says_per_round = []
    for record in records:
        if "crash" in record or record["sides"][side]["kind"] != "talking":
            continue
        team = record["sides"][side]
        events = team["transcript"]
        counts.update(event["event"] for event in events)
        rounds = {event["step"] for event in events if event["event"] == "observation"}
        says = Counter(event["step"] for event in events if event["event"] == "say")
        says_per_round.extend(says.get(step, 0) for step in rounds)
        for log in team["agent_logs"].values():
            for call in log["calls"]:
                counts["calls"] += 1
                counts["prompt_tokens"] += call["usage"].get("prompt_tokens", 0)
                counts["output_tokens"] += output_tokens(call["usage"])
                counts["cost_microdollars"] += round(call["cost"] * 1e6)
    calls = counts["calls"]
    if not calls:
        return {}
    return {
        "model_calls": calls,
        "cost_usd": round(counts["cost_microdollars"] / 1e6, 4),
        "prompt_tokens": counts["prompt_tokens"],
        "output_tokens_with_thinking": counts["output_tokens"],
        "valid_reply_rate": round(1 - counts["invalid"] / calls, 4),
        "invalid_replies": counts["invalid"],
        "fallback_choices": counts["fallback"],
        "rounds": len(says_per_round),
        "mean_messages_per_round": round(sum(says_per_round) / len(says_per_round), 2)
        if says_per_round
        else None,
        "rounds_hitting_message_ceiling": sum(n >= MAX_MESSAGES for n in says_per_round),
    }
