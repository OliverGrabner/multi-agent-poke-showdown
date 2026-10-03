import random
import re

import pytest

from pokerl.bots import BotSide, MaxPowerBot
from pokerl.bridge import Bridge
from pokerl.dex import Dex
from pokerl.env import MultiBattleEnv
from pokerl.runner import play
from pokerl.talk import MAX_MESSAGES, Choose, InvalidReply, Say, ScriptedAgent, TalkingTeam
from pokerl.teams import load_pool, make_specs


@pytest.fixture(scope="module")
def bridge():
    with Bridge() as b:
        yield b


def option_count(text: str) -> int:
    return len(re.findall(r"^\d+\. ", text.split("Your options:")[-1], flags=re.M))


def play_with(bridge, script_a, script_b, label: str, battles: int = 1) -> list[dict]:
    """Our talking team (p1+p3) against max-power bots; returns the battle records."""
    env = MultiBattleEnv(bridge)
    records = []
    for spec in make_specs(load_pool(), battles, label, mirror=False):
        alex, sam = ScriptedAgent("Alex", None), ScriptedAgent("Sam", None)
        alex.script, sam.script = script_a(alex), script_b(sam)
        team = TalkingTeam({"p1": alex, "p3": sam}, Dex(bridge))
        bots = BotSide({"p2": MaxPowerBot(), "p4": MaxPowerBot()})
        records.append(play(env, spec, {"p1p3": team, "p2p4": bots}))
    return records


def talk_once_then_choose(agent: ScriptedAgent):
    """Says one thing when a round starts (unless playing alone), then picks a random listed option."""
    rng = random.Random(agent.name)

    def script(text: str):
        observation = next(t for t in reversed(agent.inputs) if t.startswith("Turn "))
        alone = "you choose alone" in observation
        if not alone and (text.startswith("Turn ") or "nothing to choose" in observation):
            return Say("Plan: attack the opposing Pokémon.")
        return Choose(rng.randint(1, option_count(observation)))

    return script


def chatterbox(agent: ScriptedAgent):
    def script(text: str):
        if "That did not work" in text and "Call choose" in text:
            return Choose(1)
        return Say("Still thinking...")

    return script


def always_broken(agent: ScriptedAgent):
    def script(text: str):
        raise InvalidReply("no tool call found")

    return script


def events(record: dict, kind: str) -> list[dict]:
    return [e for e in record["sides"]["p1p3"]["transcript"] if e["event"] == kind]


def test_talking_team_finishes_battles_and_reveals_choices(bridge):
    records = play_with(bridge, talk_once_then_choose, talk_once_then_choose, "talk-basic", battles=3)
    for record in records:
        assert record["winning_side"] in ("p1p3", "p2p4")
        assert events(record, "say") and events(record, "choose")
        assert not events(record, "invalid") and not events(record, "fallback")


def test_partner_is_told_the_exact_choice(bridge):
    env = MultiBattleEnv(bridge)
    spec = make_specs(load_pool(), 1, "talk-reveal", mirror=False)[0]
    alex = ScriptedAgent("Alex", lambda text: Choose(1))
    sam = ScriptedAgent("Sam", lambda text: Say("ok") if "chose:" not in text else Choose(1))
    team = TalkingTeam({"p1": alex, "p3": sam}, Dex(bridge))
    play(env, spec, {"p1p3": team, "p2p4": BotSide({"p2": MaxPowerBot(), "p4": MaxPowerBot()})})
    assert any("Alex chose: " in text for text in sam.inputs)


def test_first_speaker_alternates_each_turn(bridge):
    record = play_with(bridge, talk_once_then_choose, talk_once_then_choose, "talk-order")[0]
    transcript = record["sides"]["p1p3"]["transcript"]
    told_first = [e for e in transcript if e["event"] == "observation" and "You speak first." in e["text"]]
    assert len({e["turn"] for e in told_first}) >= 2
    for told in told_first:
        assert told["seat"] == ("p1" if told["turn"] % 2 == 0 else "p3")
        # Whoever is told they speak first is the first to act in that round.
        first_actor = next(
            e["seat"] for e in transcript if e["step"] == told["step"] and e["event"] in ("say", "choose")
        )
        assert first_actor == told["seat"]


def test_talk_ceiling_stops_runaway_talk(bridge):
    record = play_with(bridge, chatterbox, chatterbox, "talk-ceiling")[0]
    says_per_round = {}
    for event in events(record, "say"):
        says_per_round[event["step"]] = says_per_round.get(event["step"], 0) + 1
    assert max(says_per_round.values()) == MAX_MESSAGES
    assert record["winning_side"] in ("p1p3", "p2p4")


def test_unusable_replies_fall_back_to_a_random_option(bridge):
    record = play_with(bridge, always_broken, always_broken, "talk-broken")[0]
    assert events(record, "fallback")
    # Every fallback follows exactly three failed attempts.
    assert len(events(record, "invalid")) >= 3 * len(events(record, "fallback"))
    assert record["winning_side"] in ("p1p3", "p2p4")


def test_observation_lists_every_legal_option(bridge):
    env = MultiBattleEnv(bridge)
    spec = make_specs(load_pool(), 1, "talk-options", mirror=False)[0]
    alex = ScriptedAgent("Alex", lambda text: Choose(1))
    sam = ScriptedAgent("Sam", lambda text: Choose(1))
    team = TalkingTeam({"p1": alex, "p3": sam}, Dex(bridge))
    views = env.reset(spec.seed, spec.teams)
    team.decide(env, ["p1", "p3"], 0, {})
    assert option_count(alex.inputs[0]) == len(views["p1"].legal)
    assert "Your team:" in alex.inputs[0] and "Sam's team:" in alex.inputs[0]
    env.close()
