"""Build the local results page: one HTML file with the numbers inlined, opened in a browser.

    python scripts/dashboard/build.py RUNS_DIR OUT.html

RUNS_DIR holds one folder per run (battles.jsonl inside), named as in RUNS below. Everything is
counted directly from the battle logs; rerun after new battles arrive.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from pokerl import FOES
from pokerl.analysis import side_report
from pokerl.llm_batch import read_records
from pokerl.runner import wilson

# Runs against the max-power bot (side p1p3 is the Qwen team), then head-to-head runs.
RUNS = {
    "base-notalk": ("A", "No talk"),
    "base-onemsg": ("B", "One message each"),
    "base-free": ("C", "Free talk"),
    "base-solo": ("D", "One model, both Pokémon"),
}
HEAD_TO_HEAD = {"h2h-free-notalk": ("Free talk", "No talk")}

PROTECT = {"Protect", "Detect", "Spiky Shield", "King's Shield", "Baneful Bunker", "Silk Trap",
           "Burning Bulwark", "Obstruct", "Wide Guard", "Quick Guard"}  # fmt: skip
SPEED_CONTROL = {"Tailwind", "Trick Room", "Icy Wind", "Electroweb", "Thunder Wave", "Bulldoze",
                 "Glare", "Scary Face", "String Shot", "Rock Tomb", "Nuzzle"}  # fmt: skip


def option_names(observation: str) -> set[str]:
    """Move or switch names from the numbered options at the end of an observation."""
    options = re.findall(r"^\d+\. (.+)$", observation, re.M)
    return {re.split(r" →| \|", option)[0] for option in options}


def foe_faints_per_turn(record: dict, side: str) -> Counter:
    foes = {"p1p3": "p[24]", "p2p4": "p[13]"}[side]
    faints, turn = Counter(), 0
    for line in record["omniscient_log"]:
        if line.startswith("|turn|"):
            turn = int(line.split("|")[2])
        elif re.match(rf"\|faint\|{foes}a", line):
            faints[turn] += 1
    return faints


def play_style(records: list[dict], side: str) -> dict:
    """How the team used its turns: focus fire vs split targets, Protect, speed control."""
    counts = Counter()
    for record in records:
        if "crash" in record:
            continue
        faints = foe_faints_per_turn(record, side)
        last_seen: dict[str, str] = {}
        chosen_by_step: dict[int, dict] = {}
        for event in record["sides"][side]["transcript"]:
            if event["event"] == "observation":
                last_seen[event["seat"]] = event["text"]
            elif event["event"] == "choose":
                seat, move = event["seat"], event["label"].split(" →")[0]
                chosen_by_step.setdefault(event["step"], {})[seat] = event
                names = option_names(last_seen.get(seat, ""))
                counts["decisions"] += 1
                counts["switches"] += move.startswith("Switch to")
                for kind, group in (("protect", PROTECT), ("speed", SPEED_CONTROL)):
                    if names & group:
                        counts[f"{kind}_available"] += 1
                        counts[f"{kind}_used"] += move in group
        for chosen in chosen_by_step.values():
            targets = []
            for seat, event in chosen.items():
                found = re.search(r"\((p[1-4])\)$", event["label"])
                if found and found.group(1) in FOES[seat]:
                    targets.append(found.group(1))
            if len(targets) == 2:
                kind = "focus" if targets[0] == targets[1] else "split"
                counts[f"{kind}_turns"] += 1
                counts[f"{kind}_kos"] += faints[int(next(iter(chosen.values()))["turn"])]
    return dict(counts)


def battle_rows(records: list[dict], run: str, sides: tuple[str, ...]) -> list[dict]:
    """One row per finished battle, with the talk and choices of the model-backed sides."""
    rows = []
    for record in records:
        if "crash" in record:
            continue
        rows.append(
            {
                "run": run,
                "key": record["battle_key"],
                "winner": record["winning_side"],
                "turns": record["turns"],
                "minutes": round(record["seconds"] / 60, 1),
                "players": record["players"],
                "events": [
                    [event["turn"], event["seat"], event["event"], event.get("text") or event["label"]]
                    for side in sides
                    for event in record["sides"][side].get("transcript", [])
                    if event["event"] in ("say", "choose")
                ],
            }
        )
    return rows


def win_summary(records: list[dict], side: str = "p1p3") -> dict:
    finished = [r for r in records if "crash" not in r]
    wins = sum(r["winning_side"] == side for r in finished)
    low, high = wilson(wins, len(finished))
    return {
        "wins": wins,
        "battles": len(finished),
        "crashes": len(records) - len(finished),
        "rate": wins / len(finished) if finished else None,
        "ci": [low, high],
        "turns": [r["turns"] for r in finished],
        "won": [r["winning_side"] == side for r in finished],
    }


def paired(base: list[dict], other: list[dict]) -> dict:
    """Battle-by-battle comparison on the same seeds and teams."""
    won = {r["battle_key"]: r["winning_side"] == "p1p3" for r in base if "crash" not in r}
    won_other = {r["battle_key"]: r["winning_side"] == "p1p3" for r in other if "crash" not in r}
    keys = won.keys() & won_other.keys()
    return {
        "shared": len(keys),
        "both": sum(won[k] and won_other[k] for k in keys),
        "only_base": sum(won[k] and not won_other[k] for k in keys),
        "only_other": sum(won_other[k] and not won[k] for k in keys),
        "neither": sum(not won[k] and not won_other[k] for k in keys),
    }


def mirror_pairs(records: list[dict]) -> dict:
    """Seeds where both battles (teams swapped) finished: did the p1p3 side win both, one or none?"""
    by_seed: dict[str, list[bool]] = {}
    for record in records:
        if "crash" not in record:
            by_seed.setdefault(record["battle_key"].rstrip("m"), []).append(record["winning_side"] == "p1p3")
    complete = [wins for wins in by_seed.values() if len(wins) == 2]
    return {
        "complete": len(complete),
        "a_both": sum(all(w) for w in complete),
        "split": sum(sum(w) == 1 for w in complete),
        "b_both": sum(not any(w) for w in complete),
    }


def main(runs_dir: Path, out_path: Path) -> None:
    runs = {name: read_records(runs_dir / name / "battles.jsonl") for name in RUNS}
    h2h = {name: read_records(runs_dir / name / "battles.jsonl") for name in HEAD_TO_HEAD}
    built = datetime.now().strftime("%Y-%m-%d %H:%M")
    data = {"built": built, "conditions": [], "head_to_head": [], "battles": []}
    for name, (letter, label) in RUNS.items():
        records = runs[name]
        data["conditions"].append(
            {
                "run": name,
                "letter": letter,
                "label": label,
                **win_summary(records),
                "usage": side_report(records, "p1p3"),
                "style": play_style(records, "p1p3"),
                "vs_free_talk": None if name == "base-free" else paired(runs["base-free"], records),
            }
        )
        data["battles"] += battle_rows(records, name, ("p1p3",))
    for name, (label_a, label_b) in HEAD_TO_HEAD.items():
        records = h2h[name]
        data["head_to_head"].append(
            {
                "run": name,
                "labels": [label_a, label_b],
                **win_summary(records),
                "pairs": mirror_pairs(records),
                "usage": [side_report(records, "p1p3"), side_report(records, "p2p4")],
                "style": [play_style(records, "p1p3"), play_style(records, "p2p4")],
            }
        )
        data["battles"] += battle_rows(records, name, ("p1p3", "p2p4"))
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", r"<\/")
    page = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    out_path.write_text(page.replace("__DATA__", payload), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(data['battles'])} battles)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
