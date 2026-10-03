"""Model-backed sides: two teammates who may talk before choosing, or one model controlling both.

Two-teammate modes (the baseline conditions, see prompts.py):
- "free": on its turn an agent either says something to its partner or chooses. Choosing locks its
  action and its partner is told exactly what it chose; talk continues until every seat that has to
  act has chosen. (keep_talking=False is the first rule: the first choice ends the talk.)
- "one-message": each agent sends exactly one message, in turn, then both choose privately.
- "no-talk": both choose privately without talking.
Who speaks first alternates each turn.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from itertools import cycle
from typing import Protocol

from pokerl import ALLY, PLAYER
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

    def record(self) -> dict:
        """What to keep in the battle log (conversation, model calls, ...)."""


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

    def record(self) -> dict:
        return {"inputs": self.inputs}


def still_playing(view: SeatView) -> bool:
    """A player is out once all of their Pokémon have fainted."""
    if not view.request:
        return False
    return any(not mon["condition"].endswith(" fnt") for mon in view.request["side"]["pokemon"])


def add(inbox: dict[str, str], seat: str, text: str) -> None:
    inbox[seat] = f"{inbox[seat]}\n\n{text}" if inbox.get(seat) else text


def problem(reply: Reply, can_talk: bool, must_choose: bool, options: int) -> str:
    """Why a well-formed reply is not allowed right now, or '' if it is."""
    if isinstance(reply, Say):
        if not reply.message.strip():
            return "your message is empty."
        if not can_talk:
            return "you cannot talk right now. Call choose."
    elif isinstance(reply, Choose):
        if not must_choose:
            return "you have nothing to choose right now (your action is set or not needed). Reply with say."
        if not 1 <= reply.option <= options:
            return f"the option must be a number from 1 to {options}."
    return ""


class ModelSide:
    """What every model-backed side shares: asking agents for usable replies, and the transcript."""

    transcript: list[dict]

    def ask(
        self,
        env: MultiBattleEnv,
        step: int,
        seat: str,
        agent: Agent,
        text: str,
        can_talk: bool,
        must_choose: bool,
    ) -> Reply | None:
        """Get a usable reply, retrying with feedback. Returns None if a non-chooser never manages one."""
        options = len(env.views[seat].legal)
        for _ in range(MAX_RETRIES + 1):
            try:
                reply = agent.respond(text)
                issue = problem(reply, can_talk, must_choose, options)
            except InvalidReply as error:
                issue = str(error)
            if not issue:
                return reply
            self.log(env, step, seat, "invalid", problem=issue)
            text = f"That did not work: {issue}"
        if not must_choose:
            return None
        option = random.Random(f"{env.seed}:{seat}:{step}:fallback").randint(1, options)
        self.log(env, step, seat, "fallback", option=option)
        return Choose(option)

    def log(self, env: MultiBattleEnv, step: int, seat: str, event: str, **fields) -> None:
        self.transcript.append({"step": step, "turn": env.turn, "seat": seat, "event": event, **fields})


class TalkingTeam(ModelSide):
    """Controls one side with two agents, each choosing for its own Pokémon."""

    def __init__(
        self,
        agents: dict[str, Agent],
        dex: Dex,
        labels: dict[str, str] | None = None,
        mode: str = "free",
        keep_talking: bool = True,
        max_messages: int = MAX_MESSAGES,
    ):
        self.agents = agents
        self.labels = labels or {}  # per seat, shown after the player's name in the battle (the model)
        self.mode = mode
        self.keep_talking = keep_talking
        self.max_messages = max_messages
        self.observers = {seat: Observer(seat, agents[ALLY[seat]].name, dex) for seat in agents}
        self.transcript: list[dict] = []

    def names(self) -> dict[str, str]:
        return {
            seat: f"{agent.name} - {self.labels[seat]}" if seat in self.labels else agent.name
            for seat, agent in self.agents.items()
        }

    def record(self) -> dict:
        return {
            "kind": "talking",
            "mode": self.mode,
            "agents": self.names(),
            "transcript": self.transcript,
            "agent_logs": {seat: agent.record() for seat, agent in self.agents.items()},
        }

    def decide(
        self, env: MultiBattleEnv, seats: list[str], step: int, rejected: dict[str, str]
    ) -> dict[str, str]:
        chosen = {seat: "pass" for seat in seats if env.views[seat].choices == ["pass"]}
        choosers = set(seats)
        talkers = [seat for seat in sorted(self.agents) if still_playing(env.views[seat])]
        if self.mode == "no-talk":
            talkers = [seat for seat in talkers if seat in choosers]
        if len(talkers) == 2 and env.turn % 2 == 1:
            talkers.reverse()
        if not choosers - chosen.keys() <= set(talkers):
            raise RuntimeError(f"Seats {choosers} must act but only {talkers} are still playing")

        inbox: dict[str, str] = {}
        for seat in talkers:
            header = self.header(env, seat, talkers, choosers, rejected)
            inbox[seat], _ = self.observers[seat].observe(env.views[seat], header)
            self.log(env, step, seat, "observation", text=inbox[seat])

        if self.mode == "free":
            self.free_talk(env, step, talkers, choosers, chosen, inbox)
        else:
            if self.mode == "one-message":
                self.one_message_each(env, step, talkers, inbox)
            self.choose_privately(env, step, talkers, choosers, chosen, inbox)

        for seat in talkers:
            self.agents[seat].finish(inbox.pop(seat, ""))
        return {seat: chosen[seat] for seat in seats}

    def free_talk(
        self,
        env: MultiBattleEnv,
        step: int,
        talkers: list[str],
        choosers: set[str],
        chosen: dict[str, str],
        inbox: dict[str, str],
    ) -> None:
        messages = 0
        for seat in cycle(talkers):
            if choosers <= chosen.keys():
                break
            partner = ALLY[seat] if ALLY[seat] in talkers else None
            can_talk = partner is not None and messages < self.max_messages
            if not self.keep_talking:
                can_talk = can_talk and partner not in chosen
            if seat in chosen:
                # A player who has chosen only speaks again to answer something new from its partner.
                if not (self.keep_talking and can_talk and inbox.get(seat)):
                    continue
            elif seat not in choosers and not can_talk:
                continue
            must_choose = seat in choosers and seat not in chosen
            reply = self.ask(env, step, seat, self.agents[seat], inbox.pop(seat, ""), can_talk, must_choose)
            if isinstance(reply, Say):
                messages += 1
                self.send(env, step, seat, partner, reply.message, inbox)
                if messages == self.max_messages:
                    for talker in talkers:
                        if talker not in chosen:
                            add(inbox, talker, "Talking time is over. Choose your action now.")
            elif isinstance(reply, Choose):
                self.lock(env, step, seat, reply, chosen)
                if partner and partner not in chosen:
                    as_partner_sees_it = self.observers[partner].option_label(
                        env.views[seat].legal[reply.option - 1]
                    )
                    add(inbox, partner, f"{self.agents[seat].name} chose: {as_partner_sees_it}")

    def one_message_each(
        self, env: MultiBattleEnv, step: int, talkers: list[str], inbox: dict[str, str]
    ) -> None:
        if len(talkers) < 2:
            return
        for seat in talkers:
            reply = self.ask(
                env, step, seat, self.agents[seat], inbox.pop(seat, ""), can_talk=True, must_choose=False
            )
            if isinstance(reply, Say):
                self.send(env, step, seat, ALLY[seat], reply.message, inbox)

    def choose_privately(
        self,
        env: MultiBattleEnv,
        step: int,
        talkers: list[str],
        choosers: set[str],
        chosen: dict[str, str],
        inbox: dict[str, str],
    ) -> None:
        """Each remaining chooser picks without seeing its partner's choice."""
        for seat in talkers:
            if seat in choosers and seat not in chosen:
                if self.mode == "one-message":
                    add(inbox, seat, "Now choose your action.")
                reply = self.ask(
                    env, step, seat, self.agents[seat], inbox.pop(seat, ""), can_talk=False, must_choose=True
                )
                self.lock(env, step, seat, reply, chosen)

    def send(
        self, env: MultiBattleEnv, step: int, seat: str, partner: str, message: str, inbox: dict[str, str]
    ) -> None:
        self.log(env, step, seat, "say", text=message)
        add(inbox, partner, f'{self.agents[seat].name} says: "{message}"')

    def lock(self, env: MultiBattleEnv, step: int, seat: str, reply: Choose, chosen: dict[str, str]) -> None:
        action = env.views[seat].legal[reply.option - 1]
        chosen[seat] = action["choice"]
        label = self.observers[seat].option_label(action)
        self.log(env, step, seat, "choose", option=reply.option, choice=action["choice"], label=label)

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
        if len(talkers) == 1 and self.mode != "no-talk":
            lines.append("Your partner is out of the battle, so you choose alone.")
        elif len(talkers) == 2 and self.mode != "no-talk":
            lines.append("You speak first." if talkers[0] == seat else f"{partner} speaks first.")
        if seat in rejected:
            lines.append(f"Your last choice was not possible ({rejected[seat]}). Choose again.")
        return " ".join(lines)


class SoloTeam(ModelSide):
    """One model controls both players on a side: the reference for perfect coordination.

    It reads the state from the first still-playing player's point of view, then picks for each
    player that has to act, one after the other, in the same conversation.
    """

    def __init__(self, agent: Agent, seats: tuple[str, str], dex: Dex, label: str = ""):
        self.agent = agent
        self.seats = seats
        self.label = label
        self.observers = {seat: Observer(seat, PLAYER[ALLY[seat]], dex) for seat in seats}
        self.transcript: list[dict] = []

    def names(self) -> dict[str, str]:
        return {seat: f"{PLAYER[seat]} - {self.label}" if self.label else PLAYER[seat] for seat in self.seats}

    def record(self) -> dict:
        return {
            "kind": "solo",
            "agents": self.names(),
            "transcript": self.transcript,
            "agent_logs": {"solo": self.agent.record()},
        }

    def decide(
        self, env: MultiBattleEnv, seats: list[str], step: int, rejected: dict[str, str]
    ) -> dict[str, str]:
        chosen = {seat: "pass" for seat in seats if env.views[seat].choices == ["pass"]}
        viewer = next(seat for seat in self.seats if still_playing(env.views[seat]))
        for seat in self.seats:
            if seat != viewer:
                self.observers[seat].catch_up(env.views[seat])
        header = " ".join(
            [f"Turn {env.turn}."]
            + [
                f"{PLAYER[seat]}'s last choice was not possible ({rejected[seat]})."
                for seat in seats
                if seat in rejected
            ]
        )
        text, _ = self.observers[viewer].observe(env.views[viewer], header, f"Options for {PLAYER[viewer]}")
        self.log(env, step, viewer, "observation", text=text)
        for seat in self.seats:
            if seat not in seats or seat in chosen:
                continue
            if seat != viewer:
                options = self.observers[seat].options_section(env.views[seat], f"Options for {PLAYER[seat]}")
                text = f"{text}\n\n{options}" if text else options
            text = f"{text}\n\nChoose the action for {PLAYER[seat]}."
            reply = self.ask(env, step, seat, self.agent, text, can_talk=False, must_choose=True)
            action = env.views[seat].legal[reply.option - 1]
            chosen[seat] = action["choice"]
            label = self.observers[seat].option_label(action)
            self.log(env, step, seat, "choose", option=reply.option, choice=action["choice"], label=label)
            text = ""
        self.agent.finish("")
        return {seat: chosen[seat] for seat in seats}
