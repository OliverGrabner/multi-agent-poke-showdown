import random

import pytest

from pokerl import SEATS
from pokerl.bots import move_score, step_rng
from pokerl.bridge import Bridge
from pokerl.cli import reproducibility_check
from pokerl.env import InvalidAction, MultiBattleEnv
from pokerl.runner import bot_sides, play
from pokerl.teams import build_strategy_pool, load_pool, make_specs


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
    sides = bot_sides(dict.fromkeys(SEATS, "random"))
    for spec in make_specs(pool, 15, "test-fuzz"):
        record = play(env, spec, sides)
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


def test_terastallization_is_off(bridge, pool):
    spec = make_specs(pool, 1, "test-tera", mirror=False)[0]
    env = MultiBattleEnv(bridge)
    views = env.reset(spec.seed, spec.teams)
    assert "|rule|Terastal Clause: You cannot Terastallize" in views["p1"].log
    assert all("terastallize" not in choice for seat in SEATS for choice in views[seat].choices)
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
    sides = bot_sides({"p1": "maxpower", "p3": "maxpower", "p2": "random", "p4": "random"})
    wins = sum(play(env, spec, sides)["winning_side"] == "p1p3" for spec in make_specs(pool, 20, "test-mp"))
    assert wins >= 28  # of 40; the full check uses 1,000 battles


def test_step_rng_is_stateless():
    assert step_rng("s", "p1", 3).random() == step_rng("s", "p1", 3).random()
    assert isinstance(step_rng("s", "p1", 3), random.Random)


def test_strategy_teams_split_between_partners_and_play_mirror_matches(bridge):
    pool = build_strategy_pool(bridge, "data/teams/strategy-v1.txt")
    assert [team["id"] for team in pool["teams"]] == ["Trick Room", "Tailwind", "Rain", "Sun"]
    trick_room = pool["teams"][0]["halves"]
    assert trick_room["first"]["species"][0] == "Indeedee-F"  # the lead comes first
    assert trick_room["second"]["species"][0] == "Hatterene"
    specs = make_specs(pool, 5, "test")
    assert len(specs) == 5  # one battle per seed: a mirror match has no sides to swap
    assert specs[0].teams["p1"] == specs[0].teams["p2"] == trick_room["first"]["packed"]
    assert specs[0].teams["p3"] == specs[0].teams["p4"] == trick_room["second"]["packed"]
    assert specs[4].team_ids["p1"] == "Trick Room (first)"  # the cycle starts over
    record = play(MultiBattleEnv(bridge), specs[1], bot_sides(dict.fromkeys(SEATS, "maxpower")))
    assert record["winning_side"] in ("p1p3", "p2p4") and record["turns"] > 0
