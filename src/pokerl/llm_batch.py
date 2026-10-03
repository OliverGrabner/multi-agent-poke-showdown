"""Battles with an LLM team (p1 + p3) against scripted bots, and a summary of how the agents talked."""

from __future__ import annotations

import json
import traceback
from collections import Counter
from pathlib import Path

from pokerl.bots import BotSide, make_policy
from pokerl.bridge import Bridge
from pokerl.dex import Dex
from pokerl.env import MultiBattleEnv
from pokerl.llm_agent import LLMAgent
from pokerl.models import BudgetExceeded, ChatClient
from pokerl.runner import play
from pokerl.talk import MAX_MESSAGES, TalkingTeam
from pokerl.teams import BattleSpec

NAMES = {"p1": "Alex", "p3": "Sam"}


def run_llm_batch(specs: list[BattleSpec], model: str, opponent: str, out_path: Path) -> list[dict]:
    """Play every spec in order, appending one JSON line per battle. Stops at once if the budget runs out."""
    client = ChatClient(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with Bridge() as bridge, out_path.open("a", encoding="utf-8") as out:
        env = MultiBattleEnv(bridge)
        dex = Dex(bridge)
        for spec in specs:
            agents = {
                "p1": LLMAgent(NAMES["p1"], NAMES["p3"], client),
                "p3": LLMAgent(NAMES["p3"], NAMES["p1"], client),
            }
            team = TalkingTeam(agents, dex)
            sides = {
                "p1p3": team,
                "p2p4": BotSide({"p2": make_policy(opponent), "p4": make_policy(opponent)}),
            }
            try:
                record = play(env, spec, sides)
            except BudgetExceeded:
                raise
            except Exception:
                record = {
                    "battle_key": spec.battle_key,
                    "crash": traceback.format_exc(),
                    "team": team.record(),
                }
            out.write(json.dumps(record) + "\n")
            out.flush()
            records.append(record)
    return records


def summarize_talk(records: list[dict]) -> dict:
    """Model calls, cost, output validity and how much the teammates talked."""
    counts = Counter()
    says_per_round = []
    for record in records:
        if "crash" in record:
            continue
        side = record["sides"]["p1p3"]
        events = side["transcript"]
        counts.update(event["event"] for event in events)
        rounds = {event["step"] for event in events if event["event"] == "observation"}
        says = Counter(event["step"] for event in events if event["event"] == "say")
        says_per_round.extend(says.get(step, 0) for step in rounds)
        for log in side["agent_logs"].values():
            for call in log["calls"]:
                counts["calls"] += 1
                counts["prompt_tokens"] += call["usage"].get("prompt_tokens", 0)
                counts["completion_tokens"] += call["usage"].get("completion_tokens", 0)
                counts["cost_microdollars"] += round(call["cost"] * 1e6)
    calls = counts["calls"]
    return {
        "model_calls": calls,
        "cost_usd": round(counts["cost_microdollars"] / 1e6, 4),
        "prompt_tokens": counts["prompt_tokens"],
        "completion_tokens": counts["completion_tokens"],
        "valid_reply_rate": round(1 - counts["invalid"] / calls, 4) if calls else None,
        "invalid_replies": counts["invalid"],
        "fallback_choices": counts["fallback"],
        "rounds": len(says_per_round),
        "mean_messages_per_round": round(sum(says_per_round) / len(says_per_round), 2)
        if says_per_round
        else None,
        "rounds_hitting_message_ceiling": sum(n >= MAX_MESSAGES for n in says_per_round),
    }
