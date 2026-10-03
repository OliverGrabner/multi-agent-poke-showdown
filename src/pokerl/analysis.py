"""Objective numbers from battle logs: results, rounds, talk volume, tokens and reply validity.

Everything here is counted directly from the logs. Judgments such as whether an agent did what it
said, or whether a pair coordinated well, are left to reading individual battles.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean

from pokerl.models import output_tokens
from pokerl.runner import summarize
from pokerl.talk import MAX_MESSAGES


def side_report(records: list[dict], side: str) -> dict:
    """Talk volume, model calls, tokens and validity for one side, if it is model-backed."""
    counts = Counter()
    says_per_round: list[int] = []
    words: list[int] = []
    battles = 0
    for record in records:
        team = record["sides"][side]
        if "crash" in record or team["kind"] == "bots":
            continue
        battles += 1
        events = team["transcript"]
        counts.update(event["event"] for event in events)
        rounds = sorted({event["step"] for event in events if event["event"] == "observation"})
        says = Counter(event["step"] for event in events if event["event"] == "say")
        says_per_round.extend(says.get(step, 0) for step in rounds)
        words.extend(len(event["text"].split()) for event in events if event["event"] == "say")
        for log in team["agent_logs"].values():
            for call in log["calls"]:
                counts["calls"] += 1
                counts["prompt_tokens"] += call["usage"].get("prompt_tokens", 0)
                counts["output_tokens"] += output_tokens(call["usage"])
                counts["cost_microdollars"] += round(call["cost"] * 1e6)
    if not battles:
        return {}
    calls = counts["calls"]
    rounds = len(says_per_round)
    return {
        "kind": records[0]["sides"][side].get("mode", records[0]["sides"][side]["kind"]),
        "players": records[0]["sides"][side]["agents"],
        "battles": battles,
        "rounds": rounds,
        "rounds_per_battle": round(rounds / battles, 2),
        "model_calls": calls,
        "model_calls_per_battle": round(calls / battles, 1),
        "messages": sum(says_per_round),
        "messages_per_round": round(sum(says_per_round) / rounds, 2) if rounds else None,
        "words_per_message": round(mean(words), 1) if words else None,
        "rounds_hitting_message_ceiling": sum(n >= MAX_MESSAGES for n in says_per_round),
        "prompt_tokens": counts["prompt_tokens"],
        "output_tokens_with_thinking": counts["output_tokens"],
        "prompt_tokens_per_battle": round(counts["prompt_tokens"] / battles),
        "output_tokens_per_battle": round(counts["output_tokens"] / battles),
        "valid_reply_rate": round(1 - counts["invalid"] / calls, 4) if calls else None,
        "invalid_replies": counts["invalid"],
        "fallback_choices": counts["fallback"],
        "cost_usd": round(counts["cost_microdollars"] / 1e6, 4),
    }


def report(records: list[dict]) -> dict:
    """Results for the whole batch, plus a report per model-backed side."""
    finished = [r for r in records if "crash" not in r]
    seconds = [r["seconds"] for r in finished]
    return {
        **summarize(records),
        "seconds_per_battle": round(mean(seconds), 1) if seconds else None,
        "side_a (p1+p3)": side_report(finished, "p1p3"),
        "side_b (p2+p4)": side_report(finished, "p2p4"),
    }
