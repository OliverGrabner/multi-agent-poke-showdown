"""Build the local results page: one HTML file with the numbers inlined, opened in a browser.

    python scripts/dashboard/build.py RUNS_DIR OUT.html

RUNS_DIR holds one folder per run (battles.jsonl inside), named as in RUNS below. Everything is
counted directly from the battle logs; rerun after new battles arrive.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from pokerl.analysis import side_report
from pokerl.llm_batch import read_records
from pokerl.runner import wilson

# Runs against the max-power bot; the Qwen team is side p1p3. Ordered by how much the team may talk.
RUNS = {
    "base-notalk": "No talk",
    "base-onemsg": "One message each",
    "base-free": "Free talk",
    "base-solo": "One model controls both",
}
# Two-team runs: (teams, label of side p1p3, label of side p2p4). Random teams, or strategy teams
# (data/teams/strategy-v1.txt) played as mirror matches.
MATCHES = {
    "h2h-free-notalk": ("random", "Free talk", "No talk"),
    "strat-free-vs-maxpower": ("strategy", "Free talk", "Max-power bot"),
    "strat-free-vs-notalk": ("strategy", "Free talk", "No talk"),
    "strat-nothink-vs-low": ("strategy", "No thinking", "Low thinking"),
    "big-vs-small": ("random", "Qwen3.8-27B", "Qwen3.5-2B"),
}


def finished(records: list[dict]) -> list[dict]:
    return [r for r in records if "crash" not in r]


def wins(records: list[dict], side: str) -> dict:
    done = finished(records)
    won = sum(r["winning_side"] == side for r in done)
    low, high = wilson(won, len(done))
    return {"wins": won, "battles": len(done), "rate": won / len(done), "ci": [low, high]}


def paired_with(base: list[dict], other: list[dict]) -> dict:
    """Battles both runs played on the same seed: how many only one of them won."""
    won = {r["battle_key"]: r["winning_side"] == "p1p3" for r in finished(base)}
    won_other = {r["battle_key"]: r["winning_side"] == "p1p3" for r in finished(other)}
    keys = won.keys() & won_other.keys()
    return {
        "shared": len(keys),
        "only_base": sum(won[k] and not won_other[k] for k in keys),
        "only_other": sum(won_other[k] and not won[k] for k in keys),
    }


def draw_pairs(records: list[dict]) -> dict:
    """Team draws whose two battles (sides swapped) both finished: did side p1p3 win both, one or none?"""
    by_seed: dict[str, list[bool]] = {}
    for r in finished(records):
        by_seed.setdefault(r["battle_key"].rstrip("m"), []).append(r["winning_side"] == "p1p3")
    complete = [won for won in by_seed.values() if len(won) == 2]
    return {
        "complete": len(complete),
        "a_both": sum(all(won) for won in complete),
        "split": sum(sum(won) == 1 for won in complete),
        "b_both": sum(not any(won) for won in complete),
    }


def by_team(records: list[dict]) -> dict:
    """Wins of side p1p3 per strategy team (both sides play the same team)."""
    counts: dict[str, list[int]] = {}
    for r in finished(records):
        team = r["team_ids"]["p1"].split(" (")[0]
        won, played = counts.get(team, [0, 0])
        counts[team] = [won + (r["winning_side"] == "p1p3"), played + 1]
    return counts


def battle_rows(records: list[dict], run: str, sides: tuple[str, ...]) -> list[dict]:
    """One row per finished battle, with every message and choice of the model-backed sides."""
    return [
        {
            "run": run,
            "key": r["battle_key"],
            "winner": r["winning_side"],
            "turns": r["turns"],
            "team": r["team_ids"]["p1"].split(" (")[0] if "(" in r["team_ids"]["p1"] else "random",
            "players": r["players"],
            "events": [
                [e["turn"], e["seat"], e["event"], e.get("text") or e["label"]]
                for side in sides
                for e in r["sides"][side].get("transcript", [])
                if e["event"] in ("say", "choose")
            ],
        }
        for r in finished(records)
    ]


def main(runs_dir: Path, out_path: Path) -> None:
    def load(name: str) -> list[dict]:
        return read_records(runs_dir / name / "battles.jsonl")

    free_talk = load("base-free")
    built = datetime.now().strftime("%Y-%m-%d %H:%M")
    data = {"built": built, "conditions": [], "matches": [], "battles": []}
    for name, label in RUNS.items():
        records = load(name)
        data["conditions"].append(
            {
                "run": name,
                "label": label,
                "crashes": len(records) - len(finished(records)),
                **wins(records, "p1p3"),
                "usage": side_report(records, "p1p3"),
                "vs_free_talk": None if name == "base-free" else paired_with(free_talk, records),
            }
        )
        data["battles"] += battle_rows(records, name, ("p1p3",))
    for name, (teams, *labels) in MATCHES.items():
        records = load(name)
        data["matches"].append(
            {
                "run": name,
                "teams": teams,
                "labels": labels,
                "crashes": len(records) - len(finished(records)),
                "sides": [wins(records, "p1p3"), wins(records, "p2p4")],
                "usage": [side_report(records, "p1p3"), side_report(records, "p2p4")],
                "by_team": by_team(records) if teams == "strategy" else None,
                "pairs": draw_pairs(records) if teams == "random" else None,
            }
        )
        data["battles"] += battle_rows(records, name, ("p1p3", "p2p4"))
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", r"<\/")
    page = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    out_path.write_text(page.replace("__DATA__", payload), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(data['battles'])} battles)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
