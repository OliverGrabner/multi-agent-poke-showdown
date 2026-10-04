"""Chat models behind one OpenAI-compatible client: Gemini for the smoke test, vLLM (Qwen) on HPRC.

API keys are read from environment variables at call time and never stored, logged or printed.
Paid models record every call's cost in a ledger on disk and refuse calls that could pass the cap.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "spend-ledger.json"


@dataclass(frozen=True)
class ModelConfig:
    model: str  # the provider's model id
    base_url: str  # a URL, or "$NAME" to read it from an environment variable
    key_env: str  # environment variable holding the API key
    price_in: float = 0.0  # USD per 1M input tokens
    price_cached: float = 0.0  # USD per 1M cached input tokens
    price_out: float = 0.0  # USD per 1M output tokens, thinking included
    budget: float | None = None  # hard cap in USD across all runs, for paid models
    max_tokens: int = 32768  # only stops runaway output
    params: dict = field(default_factory=dict)  # sampling settings from the model card


MODELS = {
    # Smoke test only (2.5-flash-lite is closed to new users). Prices: ai.google.dev/gemini-api/docs/pricing, 2026-10-03.
    "gemini-3.1-flash-lite": ModelConfig(
        model="gemini-3.1-flash-lite",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_env="gemini_key",
        price_in=0.25,
        price_cached=0.025,
        price_out=1.50,
        budget=2.00,
    ),
    # A middle-tier Gemini for demos.
    "gemini-3.5-flash-lite": ModelConfig(
        model="gemini-3.5-flash-lite",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_env="gemini_key",
        price_in=0.30,
        price_cached=0.03,
        price_out=2.50,
        budget=2.00,
    ),
    # The stronger Gemini for the strong-vs-weak demo. Promotional prices through 2026-12-31.
    "gemini-3.8-flash": ModelConfig(
        model="gemini-3.8-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_env="gemini_key",
        price_in=0.75,
        price_cached=0.075,
        price_out=3.75,
        budget=2.00,
    ),
    # Main model, served by vLLM on HPRC. Sampling: model card, thinking mode (checked 2026-10-03).
    "qwen3.8-27b": ModelConfig(
        model="Qwen/Qwen3.8-27B",
        base_url="$QWEN_BASE_URL",
        key_env="VLLM_API_KEY",
        params={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            # Thinking on, at the card's "medium" effort (the default "xhigh" ran 6-20k tokens per
            # decision). preserve_thinking stays at the card's default (on): earlier thinking
            # stays in the conversation.
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
        },
    ),
    # Second model family, served by vLLM on HPRC; used only if it passes the tool-call check.
    # Sampling: model card (checked 2026-10-03).
    "gemma-4-31b": ModelConfig(
        model="google/gemma-4-31B-it",
        base_url="$GEMMA_BASE_URL",
        key_env="VLLM_API_KEY",
        params={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    ),
}


class BudgetExceeded(RuntimeError):
    """A paid call was refused because it could take total spend past the cap."""


class ModelError(RuntimeError):
    """The model API failed after retries."""


def load_env(path: Path = REPO / ".env") -> None:
    """Read KEY=VALUE lines into the environment without overriding variables already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class Ledger:
    """Total paid spend per API key, kept on disk so the cap holds across runs.

    Every read and update goes to the file, so several clients (one per model) share one total.
    """

    def __init__(self, path: Path = LEDGER):
        self.path = path

    def read(self) -> dict[str, float]:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def total(self, key_env: str) -> float:
        return self.read().get(key_env, 0.0)

    def add(self, key_env: str, cost: float) -> None:
        spent = self.read()
        spent[key_env] = spent.get(key_env, 0.0) + cost
        self.path.write_text(json.dumps(spent, indent=1))


def output_tokens(usage: dict) -> int:
    """Billed output tokens, thinking included.

    Gemini leaves thinking out of `completion_tokens` but counts it in `total_tokens`;
    vLLM counts it in both. Total minus prompt is right for either.
    """
    if "total_tokens" in usage:
        return usage["total_tokens"] - usage.get("prompt_tokens", 0)
    return usage.get("completion_tokens", 0)


def clean_message(message: dict) -> dict:
    """The assistant message as it should be sent back: empty fields dropped, thinking under one name.

    vLLM may return Qwen's thinking as `reasoning`; Qwen's chat template reads `reasoning_content`.
    Provider extras such as Gemini's thought signatures are kept untouched.
    """
    message = {k: v for k, v in message.items() if v is not None and v != []}
    if "reasoning" in message:
        message.setdefault("reasoning_content", message.pop("reasoning"))
    return message


@dataclass
class Completion:
    message: dict  # the assistant message exactly as returned, to send back in later requests
    usage: dict
    cost: float
    seconds: float
    finish_reason: str | None


class ChatClient:
    def __init__(self, name: str, ledger: Ledger | None = None, retries: int = 4):
        self.name = name
        self.config = MODELS[name]
        self.ledger = ledger or Ledger()
        self.retries = retries
        base_url = self.config.base_url
        if base_url.startswith("$"):
            base_url = os.environ[base_url[1:]]
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=600)

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        self.check_budget(messages)
        body = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self.config.max_tokens,
            **self.config.params,
        }
        start = time.perf_counter()
        data = self.post(body)
        seconds = time.perf_counter() - start
        usage = data.get("usage") or {}
        cost = self.cost(usage)
        if self.config.budget is not None:
            self.ledger.add(self.config.key_env, cost)
        choice = data["choices"][0]
        return Completion(clean_message(choice["message"]), usage, cost, seconds, choice.get("finish_reason"))

    def post(self, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {os.environ[self.config.key_env]}"}
        for attempt in range(self.retries + 1):
            try:
                response = self.http.post("/chat/completions", json=body, headers=headers)
            except httpx.TransportError as error:
                problem = f"{type(error).__name__}"
            else:
                if response.status_code == 200:
                    return response.json()
                # Only the status and response body; never the request headers.
                problem = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in (429, 500, 502, 503, 504):
                    raise ModelError(problem)
            if attempt < self.retries:
                time.sleep(2 ** (attempt + 1))
        raise ModelError(f"Gave up after {self.retries + 1} attempts: {problem}")

    def cost(self, usage: dict) -> float:
        config = self.config
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        fresh = usage.get("prompt_tokens", 0) - cached
        return (
            fresh * config.price_in + cached * config.price_cached + output_tokens(usage) * config.price_out
        ) / 1e6

    def check_budget(self, messages: list[dict]) -> None:
        """Refuse a paid call if even its worst case (long input, maximum output) could pass the cap."""
        config = self.config
        if config.budget is None:
            return
        input_tokens = len(json.dumps(messages)) / 3  # generous; real tokens are about 4 characters
        worst = (input_tokens * config.price_in + config.max_tokens * config.price_out) / 1e6
        spent = self.ledger.total(config.key_env)
        if spent + worst > config.budget:
            raise BudgetExceeded(
                f"{self.name}: ${spent:.4f} spent of ${config.budget:.2f}; next call could cost ${worst:.4f}"
            )
