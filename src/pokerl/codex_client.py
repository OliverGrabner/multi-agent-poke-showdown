"""OpenAI models through the Codex CLI, signed in with a ChatGPT plan (no API key; plan usage).

Codex keeps each conversation itself, so a client serves one agent: the first call starts a
Codex session, later calls resume it and send only what is new. The say and choose tools become a
JSON reply shape that Codex enforces, turned back into one tool call for the agent. Codex adds its
own coding-agent instructions to every call; both teams in a comparison get the same ones.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from pokerl.models import Completion, ModelError

CODEX = shutil.which("codex") or "codex"

# Name used on the command line -> (Codex model, reasoning effort).
CODEX_MODELS = {
    "gpt-6-astra-low": ("gpt-6-astra", "low"),
    "gpt-5.6-luna-low": ("gpt-5.6-luna", "low"),
}

FORMAT = """\
HOW TO REPLY
You are not doing a coding task: do not run commands, read files or use any tools. Every reply is \
one JSON object, which the game turns into one action:
- {"action": "say", "message": "<text>", "option": 0} sends a private message to your partner.
- {"action": "choose", "message": "", "option": <number>} locks in one of your numbered options.
"""

CHOOSE_ONLY_FORMAT = """\
HOW TO REPLY
You are not doing a coding task: do not run commands, read files or use any tools. Every reply is \
one JSON object: {"action": "choose", "message": "", "option": <number>} locks in one of your \
numbered options.
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "message", "option"],
    "properties": {
        "action": {"type": "string", "enum": ["say", "choose"]},
        "message": {"type": "string"},
        "option": {"type": "integer"},
    },
}


class CodexClient:
    """One agent's conversation with a Codex model. Use `for_agent` to get a fresh one per agent."""

    def __init__(self, name: str, timeout: int = 600, retries: int = 2):
        self.name = name
        self.model, self.effort = CODEX_MODELS[name]
        self.timeout = timeout
        self.retries = retries
        self.session: str | None = None
        self.sent = 0  # messages already sent to Codex
        self.workdir = Path(tempfile.mkdtemp(prefix="pokerl-codex-"))  # empty: nothing to read
        self.schema = self.workdir / "reply.schema.json"
        self.schema.write_text(json.dumps(SCHEMA))

    def for_agent(self) -> CodexClient:
        return CodexClient(self.name, self.timeout, self.retries)

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        text = self.render(messages[self.sent :], tools)
        self.sent = len(messages) + 1  # the reply we return is appended next
        start = time.perf_counter()
        events = self.run(text)
        seconds = time.perf_counter() - start
        reply, usage = None, {}
        for event in events:
            if event.get("type") == "thread.started":
                self.session = event["thread_id"]
            elif event.get("type") == "item.completed" and event["item"].get("type") == "agent_message":
                reply = event["item"]["text"]
            elif event.get("type") == "turn.completed":
                usage = event["usage"]
        usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "cached_tokens": usage.get("cached_input_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_output_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        }
        return Completion(self.as_tool_call(reply), usage, 0.0, seconds, "stop")

    def render(self, new: list[dict], tools: list[dict]) -> str:
        """The new messages as one prompt; the system prompt and reply format come first, once."""
        parts = []
        for message in new:
            if message["role"] == "system":
                names = {tool["function"]["name"] for tool in tools}
                parts.append(message["content"] + "\n" + (FORMAT if "say" in names else CHOOSE_ONLY_FORMAT))
            elif message["role"] == "user":
                parts.append(message["content"])
            elif message["role"] == "tool":
                parts.append(message["content"])
        return "\n\n".join(parts) or "(nothing new)"

    def as_tool_call(self, reply: str | None) -> dict:
        """The JSON reply as an assistant message with one say or choose call (none if unreadable)."""
        try:
            data = json.loads(reply or "")
            action = data["action"]
            args = {"message": data["message"]} if action == "say" else {"option": data["option"]}
        except (json.JSONDecodeError, KeyError, TypeError):
            return {"role": "assistant", "content": reply or ""}
        call = {
            "id": f"call_{self.sent}",
            "type": "function",
            "function": {"name": action, "arguments": json.dumps(args)},
        }
        return {"role": "assistant", "content": None, "tool_calls": [call]}

    def run(self, prompt: str) -> list[dict]:
        common = [
            "--skip-git-repo-check", "--ignore-user-config", "--json",
            "-m", self.model, "-c", f"model_reasoning_effort={self.effort}",
            "--output-schema", str(self.schema),
        ]  # fmt: skip
        if self.session is None:
            command = [CODEX, "exec", "--sandbox", "read-only", "-C", str(self.workdir), *common, "-"]
        else:
            command = [CODEX, "exec", "resume", *common, self.session, "-"]
        problem = ""
        for attempt in range(self.retries + 1):
            try:
                result = subprocess.run(
                    command, input=prompt, capture_output=True, text=True, encoding="utf-8",
                    timeout=self.timeout, cwd=self.workdir, shell=False,
                )  # fmt: skip
            except subprocess.TimeoutExpired:
                problem = f"no reply within {self.timeout}s"
            else:
                events = []
                for line in result.stdout.splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                if any(e.get("type") == "turn.completed" for e in events):
                    return events
                errors = [e for e in events if e.get("type") in ("error", "turn.failed")]
                problem = json.dumps(errors)[:500] if errors else result.stderr[-500:]
            if attempt < self.retries:
                time.sleep(5 * (attempt + 1))
        raise ModelError(f"codex {self.model}: {problem}")
