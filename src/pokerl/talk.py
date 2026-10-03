"""Two teammates talk, then each chooses its own action.

When it is an agent's turn it either says something to its partner or chooses an option.
Choosing ends that agent's talking, and its partner is told exactly what it chose.
A round ends once every seat that has to act has chosen. Who speaks first alternates each turn.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from itertools import cycle
from typing import Protocol

from pokerl import ALLY
from pokerl.dex import Dex
from pokerl.env import MultiBattleEnv, SeatView
from pokerl.observation import Observer

MAX_MESSAGES = 20  # per round, both agents together; only a guard against runaway talk
MAX_RETRIES = 2  # extra attempts after an unusable reply


@dataclass
class Say:
    message: str


@dataclass
class Choose:
    option: int  # 1-based number from the agent's option list


Reply = Say | Choose


class InvalidReply(ValueError):
    """The agent's output could not be read as a say or a choose call."""


class Agent(Protocol):
    name: str

    def respond(self, text: str) -> Reply:
        """Read everything new since the agent last acted and return its next call."""

    def finish(self, text: str) -> None:
        """End the round: deliver anything left over without asking for a reply."""


class ScriptedAgent:
    """An agent driven by a function; used to test the talk loop without a model."""

    def __init__(self, name: str, script: Callable[[str], Reply]):
        self.name = name
        self.script = script
        self.inputs: list[str] = []

    def respond(self, text: str) -> Reply:
        self.inputs.append(text)
        return self.script(text)

    def finish(self, text: str) -> None:
        if text:
            self.inputs.append(text)


def still_playing(view: SeatView) -> bool:
    """A player is out once all of their Pokémon have fainted."""
    if not view.request:
        return False
    return any(not mon["condition"].endswith(" fnt") for mon in view.request["side"]["pokemon"])


def add(inbox: dict[str, str], seat: str, text: str) -> None:
    inbox[seat] = f"{inbox[seat]}\n\n{text}" if inbox.get(seat) else text


class TalkingTeam:
    """Controls one side with two agents that talk before choosing."""

    def __init__(self, agents: dict[str, Agent], dex: Dex, max_messages: int = MAX_MESSAGES):
        self.agents = agents
        self.max_messages = max_messages
        self.observers = {seat: Observer(seat, agents[ALLY[seat]].name, dex) for seat in agents}
        self.transcript: list[dict] = []

    def names(self) -> dict[str, str]:
        return {seat: agent.name for seat, agent in self.agents.items()}

    def record(self) -> dict:
        return {"kind": "talking", "agents": self.names(), "transcript": self.transcript}

    def decide(
        self, env: MultiBattleEnv, seats: list[str], step: int, rejected: dict[str, str]
    ) -> dict[str, str]:
        chosen = {seat: "pass" for seat in seats if env.views[seat].choices == ["pass"]}
        choosers = set(seats)
        talkers = [seat for seat in sorted(self.agents) if still_playing(env.views[seat])]
        if len(talkers) == 2 and env.turn % 2 == 1:
            talkers.reverse()
        if not choosers - chosen.keys() <= set(talkers):
            raise RuntimeError(f"Seats {choosers} must act but only {talkers} are still playing")

        inbox: dict[str, str] = {}
        for seat in talkers:
            header = self.header(env, seat, talkers, choosers, rejected)
            inbox[seat], _ = self.observers[seat].observe(env.views[seat], header)
            self.log(env, step, seat, "observation", text=inbox[seat])

        messages = 0
        for seat in cycle(talkers):
            if choosers <= chosen.keys():
                break
            partner = ALLY[seat] if ALLY[seat] in talkers else None
            can_talk = partner is not None and partner not in chosen and messages < self.max_messages
            if seat in chosen or (seat not in choosers and not can_talk):
                continue
            reply = self.ask(env, step, seat, inbox.pop(seat, ""), can_talk, seat in choosers)
            if isinstance(reply, Say):
                messages += 1
                self.log(env, step, seat, "say", text=reply.message)
                add(inbox, partner, f'{self.agents[seat].name} says: "{reply.message}"')
                if messages == self.max_messages:
                    for talker in talkers:
                        add(inbox, talker, "Talking time is over. Choose your action now.")
            elif isinstance(reply, Choose):
                action = env.views[seat].legal[reply.option - 1]
                chosen[seat] = action["choice"]
                self.log(env, step, seat, "choose", option=reply.option, choice=action["choice"])
                if partner and partner not in chosen:
                    label = self.observers[partner].option_label(action)
                    add(inbox, partner, f"{self.agents[seat].name} chose: {label}")

        for seat in talkers:
            self.agents[seat].finish(inbox.pop(seat, ""))
        return {seat: chosen[seat] for seat in seats}

    def ask(
        self, env: MultiBattleEnv, step: int, seat: str, text: str, can_talk: bool, must_choose: bool
    ) -> Reply | None:
        """Get a usable reply, retrying with feedback. Returns None if a non-chooser never manages one."""
        options = len(env.views[seat].legal)
        for _ in range(MAX_RETRIES + 1):
            try:
                reply = self.agents[seat].respond(text)
                problem = self.problem(seat, reply, can_talk, must_choose, options)
            except InvalidReply as error:
                problem = str(error)
            if not problem:
                return reply
            self.log(env, step, seat, "invalid", problem=problem)
            text = f"That did not work: {problem}"
        if not must_choose:
            return None
        option = random.Random(f"{env.seed}:{seat}:{step}:fallback").randint(1, options)
        self.log(env, step, seat, "fallback", option=option)
        return Choose(option)

    def problem(self, seat: str, reply: Reply, can_talk: bool, must_choose: bool, options: int) -> str:
        """Why a well-formed reply is not allowed right now, or '' if it is."""
        partner = self.agents[ALLY[seat]].name
        if isinstance(reply, Say):
            if not reply.message.strip():
                return "your message is empty."
            if not can_talk:
                return "you cannot talk right now. Call choose."
        elif isinstance(reply, Choose):
            if not must_choose:
                return f"you have nothing to choose right now; {partner} is choosing. You can talk with say."
            if not 1 <= reply.option <= options:
                return f"the option must be a number from 1 to {options}."
        return ""

    def header(
        self, env: MultiBattleEnv, seat: str, talkers: list[str], choosers: set[str], rejected: dict[str, str]
    ) -> str:
        partner = self.agents[ALLY[seat]].name
        lines = [f"Turn {env.turn}."]
        if env.request_state == "switch":
            if seat in choosers:
                lines.append("Your Pokémon fainted or must leave the field: choose a replacement.")
            else:
                lines.append(
                    f"{partner} must choose a replacement. You have nothing to choose, but you can talk."
                )
        if len(talkers) == 1:
            lines.append("Your partner is out of the battle, so you choose alone.")
        else:
            lines.append("You speak first." if talkers[0] == seat else f"{partner} speaks first.")
        if seat in rejected:
            lines.append(f"Your last choice was not possible ({rejected[seat]}). Choose again.")
        return " ".join(lines)

    def log(self, env: MultiBattleEnv, step: int, seat: str, event: str, **fields) -> None:
        self.transcript.append({"step": step, "turn": env.turn, "seat": seat, "event": event, **fields})
