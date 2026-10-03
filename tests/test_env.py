import random

import pytest

from pokerl import SEATS
from pokerl.bots import MaxPowerBot, RandomBot, move_score
from pokerl.bridge import Bridge
from pokerl.cli import reproducibility_check
from pokerl.env import InvalidAction, MultiBattleEnv
from pokerl.runner import play
from pokerl.teams import load_pool, make_specs


@pytest.fixture(scope="module")
def bridge():
    with Bridge() as b:
        yield b


@pytest.fixture(scope="module")
def pool():
    return load_pool()


def test_bridge_pins_showdown(bridge):
    assert bridge.call("ping")["showdown"] == "0.11.11"


def test_random_battles_finish_and_only_hidden_traps_are_rejected(bridge, pool):
    env = MultiBattleEnv(bridge)
    policies = {seat: RandomBot() for seat in SEATS}
    for spec in make_specs(pool, 15, "test-fuzz"):
        record = play(env, spec, policies)
        assert record["winning_side"] in ("p1p3", "p2p4", None)
        assert not record["truncated"]
        # Legal lists are exact except for traps the player cannot see (Shadow Tag, Arena Trap...).
        assert all("trapped" in r["error"] for r in record["rejections"]), record["rejections"]


def test_reproducible_and_resumable(bridge, pool):
    result = reproducibility_check(bridge, pool, battles=3)
    assert result == {"battles": 3, "identical_replays": 3, "identical_resumes": 3}


def test_each_seat_sees_only_its_own_exact_hp(bridge, pool):
    spec = make_specs(pool, 1, "test-views", mirror=False)[0]
    env = MultiBattleEnv(bridge)
    views = env.reset(spec.seed, spec.teams)
    switch_lines = {seat: [line for line in views[seat].log if line.startswith("|switch|")] for seat in SEATS}
    for seat in SEATS:
        for line in switch_lines[seat]:
            hp = line.split("|")[4]
            owner = line.split("|")[2][:2]
            if owner == seat:
                assert not hp.endswith("/100") or hp == "100/100"
            else:
                assert hp.endswith("/100"), (seat, line)
    # The request carries the ally's full team, moves included.
    assert views["p1"].ally["id"] == "p3"
    assert all(mon["moves"] for mon in views["p1"].ally["pokemon"])
    env.close()


def test_illegal_action_is_refused(bridge, pool):
    spec = make_specs(pool, 1, "test-illegal", mirror=False)[0]
    env = MultiBattleEnv(bridge)
    env.reset(spec.seed, spec.teams)
    with pytest.raises(InvalidAction):
        env.step({"p1": "move 9 1"})
    env.close()


def test_targets_are_explicit_and_named(bridge, pool):
    spec = make_specs(pool, 1, "test-targets", mirror=False)[0]
    env = MultiBattleEnv(bridge)
    views = env.reset(spec.seed, spec.teams)
    for seat in SEATS:
        for action in views[seat].legal:
            if action["kind"] == "move" and action["target"]:
                loc = int(action["choice"].split()[2])
                expected = {
                    "p1": {1: "p2", 2: "p4", -2: "p3"},
                    "p3": {1: "p2", 2: "p4", -1: "p1"},
                    "p2": {1: "p1", 2: "p3", -2: "p4"},
                    "p4": {1: "p1", 2: "p3", -1: "p2"},
                }
                assert action["target"]["seat"] == expected[seat][loc], action
    env.close()


def test_mirrored_specs_swap_sides():
    pool = load_pool()
    first, mirror = make_specs(pool, 1, "test-mirror")
    assert first.seed == mirror.seed and mirror.mirrored
    assert first.team_ids["p1"] == mirror.team_ids["p2"] and first.team_ids["p3"] == mirror.team_ids["p4"]


def test_move_score_prefers_super_effective_and_avoids_ally():
    def attack(effectiveness, ally=False, power=80):
        hit = {"is_ally": ally, "fainted": False, "effectiveness": effectiveness}
        return {
            "kind": "move",
            "category": "Special",
            "base_power": power,
            "accuracy": 100,
            "stab": False,
            "hits": [hit],
        }

    assert move_score(attack(2)) > move_score(attack(1)) > move_score(attack(0.5))
    assert move_score(attack(2, ally=True)) is None


def test_maxpower_beats_random_over_mirrored_pairs(bridge, pool):
    env = MultiBattleEnv(bridge)
    policies = {"p1": MaxPowerBot(), "p3": MaxPowerBot(), "p2": RandomBot(), "p4": RandomBot()}
    wins = sum(
        play(env, spec, policies)["winning_side"] == "p1p3" for spec in make_specs(pool, 20, "test-mp")
    )
    assert wins >= 28  # of 40; the full check uses 1,000 battles


def test_step_rng_is_stateless():
    from pokerl.runner import step_rng

    assert step_rng("s", "p1", 3).random() == step_rng("s", "p1", 3).random()
    assert isinstance(step_rng("s", "p1", 3), random.Random)
