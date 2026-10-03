"""Command line: generate team pools, run bot batches, export replays, run the Phase 1 checks."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from pokerl import SEATS
from pokerl.bots import RandomBot, step_rng
from pokerl.bridge import Bridge
from pokerl.env import MultiBattleEnv
from pokerl.replay import write_replay
from pokerl.runner import bot_sides, play, run_batch, summarize
from pokerl.teams import DEFAULT_POOL, generate_pool, load_pool, make_specs


def seat_policies(args: argparse.Namespace) -> dict[str, str]:
    side_a, side_b = args.side_a, args.side_b
    chosen = {"p1": side_a, "p3": side_a, "p2": side_b, "p4": side_b}
    for seat in SEATS:
        if getattr(args, seat):
            chosen[seat] = getattr(args, seat)
    return chosen


def cmd_teams(args: argparse.Namespace) -> None:
    with Bridge() as bridge:
        pool = generate_pool(bridge, args.size, args.label)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {len(pool['teams'])} teams to {out}")


def cmd_run(args: argparse.Namespace) -> None:
    pool = load_pool(args.pool)
    specs = make_specs(pool, args.pairs, args.label, mirror=not args.no_mirror)
    out_dir = Path(args.out_dir) / args.label
    battles = out_dir / "battles.jsonl"
    if battles.exists():
        raise SystemExit(f"{battles} exists; pick a new --label")
    policies = seat_policies(args)
    start = time.perf_counter()
    records = run_batch(specs, policies, battles, workers=args.workers)
    elapsed = time.perf_counter() - start
    summary = {"policies": policies, **summarize(records), "seconds": round(elapsed, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    for record in records[: args.replays]:
        if "crash" not in record:
            write_replay(record, out_dir / "replays" / f"{record['battle_key']}.html")
    print(json.dumps(summary, indent=1))


def cmd_replay(args: argparse.Namespace) -> None:
    for line in Path(args.battles).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["battle_key"] == args.key:
            path = write_replay(
                record, args.out or Path(args.battles).parent / "replays" / f"{args.key}.html"
            )
            print(f"Wrote {path}")
            return
    raise SystemExit(f"No battle {args.key} in {args.battles}")


def reproducibility_check(bridge: Bridge, pool: dict, battles: int) -> dict:
    """Same seed and choices give the same battle, and a mid-battle save resumes identically twice."""
    specs = make_specs(pool, battles, "repro", mirror=False)
    bot = RandomBot()
    env = MultiBattleEnv(bridge)
    same_replay = sum(
        play(env, spec, bot_sides(dict.fromkeys(SEATS, "random")))["omniscient_log"]
        == play(env, spec, bot_sides(dict.fromkeys(SEATS, "random")))["omniscient_log"]
        for spec in specs
    )

    def advance(step: int, until_turn: int | None = None) -> int:
        while not env.ended and (until_turn is None or env.turn < until_turn):
            env.step({s: bot.act(env.views[s], step_rng(env.seed, s, step)) for s in env.pending})
            step += 1
        return step

    same_resume = 0
    for i, spec in enumerate(specs):
        env.reset(spec.seed, spec.teams)
        step = advance(0, until_turn=2 + i % 10)
        saved = env.save()
        advance(step)
        uninterrupted = list(env.omniscient_log)
        env.load(saved)
        advance(step)
        first = list(env.omniscient_log)
        env.load(saved)
        advance(step)
        same_resume += uninterrupted == first == env.omniscient_log
    env.close()
    return {"battles": battles, "identical_replays": same_replay, "identical_resumes": same_resume}


def cmd_check(args: argparse.Namespace) -> None:
    """Phase 1 acceptance: stability over many battles, max-power vs random, reproducibility."""
    pool = load_pool(args.pool)
    out_dir = Path(args.out_dir) / args.label
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; pick a new --label")
    report = {}
    start = time.perf_counter()
    records = run_batch(
        make_specs(pool, args.pairs, f"{args.label}-maxpower"),
        {"p1": "maxpower", "p3": "maxpower", "p2": "random", "p4": "random"},
        out_dir / "maxpower-vs-random.jsonl",
        workers=args.workers,
    )
    report["maxpower_vs_random"] = summarize(records)
    records = run_batch(
        make_specs(pool, args.pairs // 4, f"{args.label}-random"),
        dict.fromkeys(SEATS, "random"),
        out_dir / "random-vs-random.jsonl",
        workers=args.workers,
    )
    report["random_vs_random"] = summarize(records)
    with Bridge() as bridge:
        report["reproducibility"] = reproducibility_check(bridge, pool, args.repro)
    report["seconds"] = round(time.perf_counter() - start, 1)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pokerl")
    sub = parser.add_subparsers(dest="command", required=True)

    teams = sub.add_parser("teams", help="generate a fixed team pool")
    teams.add_argument("--size", type=int, default=32)
    teams.add_argument("--label", default="v1")
    teams.add_argument("--out", default=str(DEFAULT_POOL))
    teams.set_defaults(func=cmd_teams)

    run = sub.add_parser("run", help="play a batch of battles between scripted policies")
    run.add_argument("--label", required=True, help="names the output folder and seeds the battles")
    run.add_argument("--side-a", default="random", help="policy for p1 and p3")
    run.add_argument("--side-b", default="random", help="policy for p2 and p4")
    for seat in SEATS:
        run.add_argument(f"--{seat}", help=f"override the policy for {seat}")
    run.add_argument("--pairs", type=int, default=50, help="seeds; each is played twice unless --no-mirror")
    run.add_argument("--no-mirror", action="store_true")
    run.add_argument("--replays", type=int, default=5, help="export replays for the first N battles")
    run.set_defaults(func=cmd_run)

    replay = sub.add_parser("replay", help="export one logged battle as a replay page")
    replay.add_argument("battles", help="a battles.jsonl file")
    replay.add_argument("key", help="battle_key of the battle")
    replay.add_argument("--out")
    replay.set_defaults(func=cmd_replay)

    check = sub.add_parser("check", help="Phase 1 acceptance checks")
    check.add_argument("--label", default="phase1-check")
    check.add_argument("--pairs", type=int, default=500, help="max-power vs random seeds (x2 mirrored)")
    check.add_argument("--repro", type=int, default=20, help="battles for the reproducibility checks")
    check.set_defaults(func=cmd_check)

    for command in (run, check):
        command.add_argument("--pool", default=str(DEFAULT_POOL))
        command.add_argument("--out-dir", default="runs")
        command.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
