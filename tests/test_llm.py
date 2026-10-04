import json

import pytest

from pokerl.llm_agent import LLMAgent
from pokerl.models import BudgetExceeded, ChatClient, Completion, Ledger, load_env
from pokerl.prompts import system_prompt
from pokerl.talk import Choose, InvalidReply, Say


def tool_call(call_id: str, name: str, **arguments) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


class FakeClient:
    """Returns prepared assistant messages instead of calling a model."""

    name = "fake"

    def __init__(self, *messages: dict):
        self.replies = list(messages)
        self.seen: list[list[dict]] = []

    def for_agent(self) -> "FakeClient":
        return self

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        self.seen.append([dict(m) for m in messages])
        return Completion(
            self.replies.pop(0), {"prompt_tokens": 10, "completion_tokens": 2}, 0.0, 0.01, "stop"
        )


def assistant(*calls: dict, content: str | None = None) -> dict:
    message = {"role": "assistant", "tool_calls": list(calls)}
    if content:
        message["content"] = content
    return message


def test_say_then_choose_threads_tool_results():
    client = FakeClient(
        assistant(tool_call("a", "say", message="Hit Mewtwo?")), assistant(tool_call("b", "choose", option=2))
    )
    agent = LLMAgent("Alex", client, system_prompt("Alex", "Sam"))
    assert agent.respond("Turn 1. You speak first.") == Say("Hit Mewtwo?")
    assert agent.respond('Sam says: "Yes."') == Choose(2)
    agent.finish("")
    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert agent.messages[3] == {"role": "tool", "tool_call_id": "a", "content": 'Sam says: "Yes."'}
    assert agent.messages[5]["tool_call_id"] == "b"
    assert "Alex" in agent.messages[0]["content"] and "Sam" in agent.messages[0]["content"]


def test_next_round_starts_with_a_user_message():
    client = FakeClient(
        assistant(tool_call("a", "choose", option=1)), assistant(tool_call("b", "choose", option=1))
    )
    agent = LLMAgent("Alex", client, system_prompt("Alex", "Sam"))
    agent.respond("Turn 1.")
    agent.finish("")
    agent.respond("Turn 2.")
    assert agent.messages[-2] == {"role": "user", "content": "Turn 2."}


def test_reply_without_a_tool_call_is_invalid_and_feedback_is_a_user_message():
    client = FakeClient(
        assistant(content="I think I'll attack."), assistant(tool_call("a", "choose", option=1))
    )
    agent = LLMAgent("Alex", client, system_prompt("Alex", "Sam"))
    with pytest.raises(InvalidReply):
        agent.respond("Turn 1.")
    assert agent.respond("That did not work: call say or choose.") == Choose(1)
    assert agent.messages[3] == {"role": "user", "content": "That did not work: call say or choose."}


def test_two_tool_calls_are_invalid_and_both_get_results():
    client = FakeClient(
        assistant(tool_call("a", "say", message="hi"), tool_call("b", "choose", option=1)),
        assistant(tool_call("c", "choose", option=1)),
    )
    agent = LLMAgent("Alex", client, system_prompt("Alex", "Sam"))
    with pytest.raises(InvalidReply):
        agent.respond("Turn 1.")
    agent.respond("That did not work.")
    results = [m for m in agent.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["a", "b"]


def test_option_as_digit_string_is_accepted_and_garbage_is_not():
    bad_json = {"id": "b", "type": "function", "function": {"name": "choose", "arguments": "{option: 3"}}
    client = FakeClient(assistant(tool_call("a", "choose", option="3")), assistant(bad_json))
    agent = LLMAgent("Alex", client, system_prompt("Alex", "Sam"))
    assert agent.respond("Turn 1.") == Choose(3)
    agent.finish("")
    with pytest.raises(InvalidReply):
        agent.respond("Turn 2.")


def test_budget_cap_refuses_calls_that_could_pass_it(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    client = ChatClient("gemini-3.1-flash-lite", ledger=ledger)
    client.check_budget([{"role": "user", "content": "hi"}])  # nothing spent yet: allowed
    ledger.add(client.config.key_env, client.config.budget - 0.001)
    with pytest.raises(BudgetExceeded):
        client.check_budget([{"role": "user", "content": "hi"}])
    assert Ledger(tmp_path / "ledger.json").total(client.config.key_env) == pytest.approx(1.999)


def test_cost_counts_cached_input_at_the_cached_price():
    client = ChatClient("gemini-3.1-flash-lite", ledger=Ledger.__new__(Ledger))
    usage = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
        "prompt_tokens_details": {"cached_tokens": 500_000},
    }
    config = client.config
    assert client.cost(usage) == pytest.approx(
        0.5 * config.price_in + 0.5 * config.price_cached + config.price_out
    )


def test_load_env_reads_values_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("pokerl_test_a=one\npokerl_test_b='two'\n# comment\n")
    monkeypatch.setenv("pokerl_test_a", "already set")
    monkeypatch.delenv("pokerl_test_b", raising=False)
    load_env(env)
    import os

    assert os.environ["pokerl_test_a"] == "already set"
    assert os.environ["pokerl_test_b"] == "two"


def test_thinking_is_sent_back_under_the_name_qwens_template_reads():
    from pokerl.models import clean_message

    returned = {"role": "assistant", "content": None, "reasoning": "Mewtwo is faster...", "tool_calls": []}
    assert clean_message(returned) == {"role": "assistant", "reasoning_content": "Mewtwo is faster..."}


def test_output_tokens_include_thinking_that_gemini_leaves_out_of_completion_tokens():
    from pokerl.models import output_tokens

    assert output_tokens({"prompt_tokens": 2434, "completion_tokens": 40, "total_tokens": 3113}) == 679
    assert output_tokens({"prompt_tokens": 10, "completion_tokens": 5}) == 5


def test_two_clients_share_one_ledger_total(tmp_path):
    first, second = Ledger(tmp_path / "ledger.json"), Ledger(tmp_path / "ledger.json")
    first.add("gemini_key", 0.25)
    second.add("gemini_key", 0.50)
    first.add("gemini_key", 0.25)
    assert second.total("gemini_key") == pytest.approx(1.0)


def test_talk_guidance_is_on_by_default_and_can_be_turned_off():
    from pokerl.prompts import system_prompt

    assert "Talking is how your team plans together." in system_prompt("Blue 1", "Blue 2")
    plain = system_prompt("Blue 1", "Blue 2", talk_guidance=False)
    assert "Talking is how your team plans together." not in plain
    assert "TALKING AND CHOOSING\n- Before acting" in plain


def test_sides_for_each_condition():
    from pokerl.llm_agent import CHOOSE_ONLY
    from pokerl.llm_batch import make_side
    from pokerl.talk import SoloTeam, TalkingTeam

    clients = {
        name: FakeClient() for name in ("gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite")
    }
    mixed = make_side("p1p3", "gemini-3.8-flash+gemini-3.1-flash-lite", "free", clients, dex=None)
    assert mixed.names() == {"p1": "Blue 1 - gemini-3.8-flash", "p3": "Blue 2 - gemini-3.1-flash-lite"}
    silent = make_side("p2p4", "gemini-3.5-flash-lite", "no-talk", clients, dex=None)
    assert isinstance(silent, TalkingTeam) and silent.mode == "no-talk"
    assert silent.agents["p2"].tools == CHOOSE_ONLY
    assert "cannot talk to each other" in silent.agents["p2"].messages[0]["content"]
    solo = make_side("p1p3", "gemini-3.8-flash", "solo", clients, dex=None)
    assert isinstance(solo, SoloTeam) and solo.names() == {
        "p1": "Blue 1 - gemini-3.8-flash",
        "p3": "Blue 2 - gemini-3.8-flash",
    }
    assert "You control both players" in solo.agent.messages[0]["content"]


def test_prompt_states_the_chosen_talk_rule():
    from pokerl.prompts import system_prompt

    assert "You can keep talking until Blue 2 has chosen too" in system_prompt("Blue 1", "Blue 2")
    first_rule = system_prompt("Blue 1", "Blue 2", keep_talking=False)
    assert "Once Blue 2 has chosen, you cannot talk any more this turn" in first_rule
