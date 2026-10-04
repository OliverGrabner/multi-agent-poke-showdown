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
# Head-to-head runs: (label of side p1p3, label of side p2p4). Both sides are Qwen.
HEAD_TO_HEAD = {"h2h-free-notalk": ("Free talk", "No talk")}


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


def battle_rows(records: list[dict], run: str, sides: tuple[str, ...]) -> list[dict]:
    """One row per finished battle, with every message and choice of the model-backed sides."""
    return [
        {
            "run": run,
            "key": r["battle_key"],
            "winner": r["winning_side"],
            "turns": r["turns"],
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
    data = {"built": built, "conditions": [], "head_to_head": [], "battles": []}
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
    for name, labels in HEAD_TO_HEAD.items():
        records = load(name)
        data["head_to_head"].append(
            {
                "run": name,
                "labels": labels,
                "crashes": len(records) - len(finished(records)),
                "sides": [wins(records, "p1p3"), wins(records, "p2p4")],
            }
        )
        data["battles"] += battle_rows(records, name, ("p1p3", "p2p4"))
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", r"<\/")
    page = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    out_path.write_text(page.replace("__DATA__", payload), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(data['battles'])} battles)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
