"""An agent backed by a chat model, playing through two tools: say and choose.

Each agent keeps one conversation for the whole battle. A new round starts with a user message
(the observation); within a round, whatever the partner does next comes back as the result of the
agent's last tool call. With Qwen's chat template this keeps thinking within a round and drops it
after, because reasoning is only kept for messages after the latest user message.
"""

from __future__ import annotations

import json

from pokerl.models import ChatClient
from pokerl.prompts import system_prompt
from pokerl.talk import Choose, InvalidReply, Reply, Say

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "say",
            "description": "Send a private message to your partner.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string", "description": "Your message."}},
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "choose",
            "description": "Lock in your action. Your partner is told what you chose.",
            "parameters": {
                "type": "object",
                "properties": {
                    "option": {"type": "integer", "description": "The number of one of your options."}
                },
                "required": ["option"],
            },
        },
    },
]


def parse_call(call: dict) -> Reply:
    """One tool call from the model -> Say or Choose."""
    name = call["function"]["name"]
    raw = call["function"].get("arguments") or "{}"
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        raise InvalidReply(f"could not read the arguments of {name}: {raw[:200]}") from None
    if name == "say":
        if not isinstance(args.get("message"), str):
            raise InvalidReply("say needs a text message.")
        return Say(args["message"])
    if name == "choose":
        option = args.get("option")
        if isinstance(option, str) and option.strip().isdigit():
            option = int(option)
        if not isinstance(option, int):
            raise InvalidReply("choose needs the number of one of your options.")
        return Choose(option)
    raise InvalidReply(f"there is no tool called {name}; use say or choose.")


class LLMAgent:
    def __init__(self, name: str, partner_name: str, client: ChatClient, keep_talking: bool = True):
        self.name = name
        self.client = client
        prompt = system_prompt(name, partner_name, keep_talking=keep_talking)
        self.messages: list[dict] = [{"role": "system", "content": prompt}]
        self.waiting_calls: list[str] = []  # ids of tool calls whose result has not been sent yet
        self.calls: list[dict] = []  # one entry per model call, for the battle log

    def respond(self, text: str) -> Reply:
        self.deliver(text)
        completion = self.client.complete(self.messages, TOOLS)
        message = completion.message
        self.messages.append(message)
        tool_calls = message.get("tool_calls", [])
        self.waiting_calls = [call["id"] for call in tool_calls]
        self.calls.append(
            {
                "usage": completion.usage,
                "cost": completion.cost,
                "seconds": round(completion.seconds, 3),
                "finish_reason": completion.finish_reason,
            }
        )
        if completion.finish_reason == "length":
            raise InvalidReply("your reply was cut off. Reply with a single say or choose call.")
        if len(tool_calls) != 1:
            raise InvalidReply("reply by calling exactly one tool: say or choose.")
        return parse_call(tool_calls[0])

    def finish(self, text: str) -> None:
        """End the round: answer any open tool call, or keep leftover news for the next round."""
        if self.waiting_calls or text:
            self.deliver(text or "OK.")

    def deliver(self, text: str) -> None:
        """Send text as the result of the agent's open tool calls, or as a new user message."""
        text = text or "(nothing new)"
        if self.waiting_calls:
            for call_id in self.waiting_calls:
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            self.waiting_calls = []
        else:
            self.messages.append({"role": "user", "content": text})

    def record(self) -> dict:
        return {"model": self.client.name, "messages": self.messages, "calls": self.calls}
