"""The engine slot -- the one part of the harness that is meant to be replaced.

Everything else in this package is engine-agnostic on purpose. Argus, and later
Metis/Talos/Oracle, do not know or care what is generating tokens. That is the
whole point of the engine-swap design: a loaner engine buys capability now, the
from-scratch Daedalus core takes the slot later, and nothing around it changes.

The contract is deliberately tiny -- one method, streaming, cancellable:

    class MyEngine:
        name = "my-engine"
        def generate(self, prompt, context, cancelled):
            yield "some text"

`context` is what Argus retrieved, already packed and annotated with provenance.
`cancelled()` is polled between chunks; return promptly when it goes True, since
the editor is waiting to report the turn as cancelled.

Two engines ship here. `RetrievalOnlyEngine` needs no model at all and is what
makes the ACP server useful today -- it reports what Argus found, which is a real
answer to "where is this handled?". `TransformersEngine` is the Track B slot for
a QLoRA'd Qwen2.5-Coder.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import (Any, Callable, Dict, Iterator, List, Optional, Protocol,
                    Sequence, runtime_checkable)

from .jsonrpc import log

__all__ = ["Engine", "RetrievalOnlyEngine", "TransformersEngine", "StaticEngine",
           "OpenAICompatEngine", "Provider", "PROVIDERS", "Thought", "ThinkSplitter",
           "Usage"]

Cancelled = Callable[[], bool]


def _int_env(name: str) -> Optional[int]:
    """A positive integer from the environment, or None. Never raises.

    A malformed override is a configuration mistake, not a reason to refuse to
    start -- so it is logged and ignored rather than propagated.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        log(f"[engine] ignoring ${name}={raw!r}: not an integer")
        return None
    if value <= 0:
        log(f"[engine] ignoring ${name}={value}: must be positive")
        return None
    return value


class Thought(str):
    """Reasoning text, to be shown as thinking rather than as the answer.

    Reasoning models emit their scratchpad in the same stream as the reply. ACP
    has a separate channel for it -- `agent_thought_chunk` -- which editors
    render collapsed. Yielding a `Thought` instead of a `str` routes a chunk
    there.

    It subclasses `str` so an engine that never yields one, or a caller that
    does not care, is unaffected: everything still behaves like text.
    """


_OPEN, _CLOSE = "<think>", "</think>"


def _partial_tail(text: str, tag: str) -> int:
    """Length of the longest suffix of `text` that could start `tag`."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


class ThinkSplitter:
    """Separates `<think>...</think>` from reply text across a chunked stream.

    The tags do not respect chunk boundaries -- `<th` can arrive in one delta and
    `ink>` in the next -- so a naive `str.replace` per chunk both misses tags and
    leaks their fragments into the reply. This holds back any suffix that could
    still turn out to be the start of a tag.
    """

    def __init__(self) -> None:
        self.buf = ""
        self.in_think = False

    def feed(self, text: str) -> List[tuple[bool, str]]:
        """Return `(is_thought, text)` pairs for everything unambiguous so far."""
        self.buf += text
        out: List[tuple[bool, str]] = []
        while self.buf:
            tag = _CLOSE if self.in_think else _OPEN
            idx = self.buf.find(tag)
            if idx == -1:
                hold = _partial_tail(self.buf, tag)
                if len(self.buf) > hold:
                    out.append((self.in_think, self.buf[:len(self.buf) - hold]))
                    self.buf = self.buf[len(self.buf) - hold:]
                break
            if idx:
                out.append((self.in_think, self.buf[:idx]))
            self.buf = self.buf[idx + len(tag):]
            self.in_think = not self.in_think
        return [(t, s) for t, s in out if s]

    def flush(self) -> List[tuple[bool, str]]:
        """Emit whatever is held back. A partial tag at EOF was never a tag."""
        rest, self.buf = self.buf, ""
        return [(self.in_think, rest)] if rest else []


@dataclass
class Usage:
    """What one request, or a run of them, cost.

    The harness could measure everything about a run except what it spent. That
    is the gap that makes "this harness change is an improvement" unfalsifiable:
    a change that raises the solve rate by burning three times the tokens is not
    obviously an improvement, and without a number nobody can say which happened.
    `scripts/seeds.py` already refuses a score without its `n`; this is the same
    rule applied to the other axis.

    `cached` is reported separately rather than folded into `prompt`, because a
    cached prefix token is billed at a fraction of a fresh one and is the whole
    return on keeping the system prompt stable. It is a *subset* of `prompt`, not
    an addition to it -- providers count it that way and so does this.

    Every field stays 0 when the provider says nothing, which is the honest
    reading: absent usage means unknown, not free.
    """

    prompt: int = 0
    completion: int = 0
    #: Prompt tokens served from the provider's cache. Subset of `prompt`.
    cached: int = 0
    #: How many requests this covers, so an average is available downstream.
    requests: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    @property
    def known(self) -> bool:
        """False when no provider ever reported anything.

        Callers must be able to tell "measured zero" from "never measured", and
        printing 0 tokens for a run that plainly cost something is the sort of
        confident wrong number this project exists to avoid.
        """
        return self.requests > 0 and self.total > 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt + other.prompt,
                     self.completion + other.completion,
                     self.cached + other.cached,
                     self.requests + other.requests)

    def report(self) -> str:
        if not self.known:
            return "tokens: not reported by this provider"
        hit = f", {self.cached} cached" if self.cached else ""
        return (f"tokens: {self.total} ({self.prompt} in{hit}, "
                f"{self.completion} out) over {self.requests} request(s)")

    @classmethod
    def from_payload(cls, raw: Any) -> "Usage":
        """Read a provider's `usage` object. Unknown shapes yield zeros.

        Providers disagree on where a cache hit is reported: OpenAI nests it at
        `prompt_tokens_details.cached_tokens`, DeepSeek puts it flat at
        `prompt_cache_hit_tokens`. Both are read, neither is required.
        """
        if not isinstance(raw, dict):
            return cls()
        details = raw.get("prompt_tokens_details")
        cached = 0
        if isinstance(details, dict):
            cached = _as_int(details.get("cached_tokens"))
        if not cached:
            cached = _as_int(raw.get("prompt_cache_hit_tokens"))
        return cls(prompt=_as_int(raw.get("prompt_tokens")),
                   completion=_as_int(raw.get("completion_tokens")),
                   cached=cached,
                   requests=1)


def _as_int(value: Any) -> int:
    """Coerce a provider's number to int. Anything unusable becomes 0."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _flatten_tool_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Render native `tool_calls` / `tool` messages back into plain text.

    For providers whose OpenAI-compatible surface does not accept them. The
    conversation is the same conversation; only its shape changes, back to the
    fenced convention every model here already understands.

    Adjacent `tool` messages merge into one user turn, because that is what the
    non-native path produces and a run of single-result user messages would be a
    third shape neither side was built for.
    """
    out: List[Dict[str, Any]] = []
    pending: List[str] = []

    def flush() -> None:
        if pending:
            out.append({"role": "user", "content": "\n".join(pending)})
            pending.clear()

    for message in messages:
        role = message.get("role")
        if role == "tool":
            if not pending:
                pending.append("## Tool results")
            pending.append(f"\n### {message.get('name', 'tool')}\n"
                           f"{message.get('content', '')}")
            continue
        flush()
        if role == "assistant" and message.get("tool_calls"):
            parts = [str(message.get("content") or "")]
            for call in message["tool_calls"]:
                function = call.get("function") or {}
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                parts.append("\n```json\n"
                             + json.dumps({"tool": function.get("name", ""),
                                           "args": args})
                             + "\n```\n")
            out.append({"role": "assistant", "content": "".join(parts).strip()})
            continue
        out.append(dict(message))
    flush()
    return out


def _fold_system(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge a system message into the first user message.

    For the providers that reject a `system` role outright. Folding rather than
    dropping, because the system message carries the tool schema and the
    constitution -- discarding it would silently change what the agent is
    allowed to do.
    """
    out: List[Dict[str, Any]] = []
    carried = ""
    for message in messages:
        if message.get("role") == "system":
            carried += ("\n\n" if carried else "") + str(message.get("content", ""))
            continue
        if carried and message.get("role") == "user":
            out.append({"role": "user",
                        "content": f"{carried}\n\n{message.get('content', '')}"})
            carried = ""
            continue
        out.append(dict(message))
    if carried:
        out.insert(0, {"role": "user", "content": carried})
    return out


@runtime_checkable
class Engine(Protocol):
    """Anything that can turn a prompt plus context into a stream of text.

    `generate` is the whole required surface, and deliberately so: the slot must
    one day hold a model that only maps bytes to bytes.

    An engine *may* also offer `generate_messages(messages, cancelled)`, taking
    an OpenAI-style array instead of a flattened string. Talos prefers it when
    present and falls back to `generate` otherwise, so this stays an optional
    capability rather than a second required method -- the same shape as
    `prepare` on `Verifier`.
    """

    name: str

    def generate(self, prompt: str, context: str, cancelled: Cancelled) -> Iterator[str]:
        ...


class RetrievalOnlyEngine:
    """No model. Reports what Argus retrieved, and says so plainly.

    This exists so the ACP server is honest rather than empty before Track B
    lands: it will not answer a question, but "these are the twelve places this
    identifier is defined, ranked, with reasons" is genuinely the answer to a
    large share of the questions asked of a coding agent -- and it is the half
    that does not require a model to be correct.
    """

    name = "retrieval-only"

    def generate(self, prompt: str, context: str, cancelled: Cancelled) -> Iterator[str]:
        hits = getattr(context, "hits", None)          # an argus.Context, ideally
        if not hits:
            if context.strip():                        # a bare string: nothing to itemise
                yield "No engine is loaded. Retrieved context:\n\n"
                yield str(context)
                return
            yield ("Nothing in the index matched that. Argus is structural, not "
                   "semantic -- try naming a symbol, file, or identifier.")
            return

        yield ("No engine is loaded, so I can't reason about this yet. "
               "Here is what Argus pulled for it:\n\n")
        for hit in hits:
            if cancelled():
                return
            label = f"  ({hit.symbol})" if hit.symbol else ""
            yield f"{hit.ref}-{hit.end_line}{label}  [{hit.reason}]\n"


class StaticEngine:
    """Returns a fixed script of chunks. For tests, and for exercising a client."""

    name = "static"

    def __init__(self, chunks: Optional[List[str]] = None) -> None:
        self.chunks = chunks if chunks is not None else ["ok"]
        self.calls: List[tuple[str, str]] = []

    def generate(self, prompt: str, context: str, cancelled: Cancelled) -> Iterator[str]:
        self.calls.append((prompt, context))
        for chunk in self.chunks:
            if cancelled():
                return
            yield chunk


def _human_bytes(count: int) -> str:
    """A size a person can act on. `0.0 MB` for a 5 KB body helps nobody."""
    for unit, scale in (("MB", 1_048_576), ("KB", 1024)):
        if count >= scale:
            return f"{count / scale:.1f} {unit}"
    return f"{count} B"


class RequestTooLarge(urllib.error.HTTPError):
    """Refused by the harness before sending, reported as the 413 it would be.

    Subclassing `HTTPError` rather than inventing a new type is the whole point:
    every retry guard, every `except HTTPError` and `_explain` itself already
    know what an HTTP failure means, and a locally-detected overflow is the same
    event as a remotely-detected one -- caught earlier and with the exact size
    in hand. Nothing downstream needs to learn a new exception.
    """

    def __init__(self, size: int, limit: int) -> None:
        detail = (f"request body is {size} bytes, over the configured "
                  f"max_request_bytes of {limit}")
        super().__init__(
            url="", code=413, msg=detail, hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(json.dumps({"error": {"message": detail}}).encode()))
        self.size = size
        self.limit = limit


@dataclass(frozen=True)
class EngineProfile:
    """What the thing in the slot can actually take.

    The harness has one set of policies -- prompt budget, step ceiling, how tool
    calls are expressed -- and until now they were global constants tuned for an
    imagined engine. They serve neither of the two engines this project targets.
    A frontier model wants a long horizon, a large stable prefix that caches, and
    native tool calls; a 3B-active local model wants short steps, a small prompt,
    and a fenced-JSON fallback because it has no tool API at all.

    This is the descriptor that lets one harness do both. It says what the engine
    *is*, not what the loop should do about it -- that mapping belongs to the
    caller, so a policy change does not require editing a capability table.

    `None` means unknown, which is deliberately distinct from unlimited. An
    unknown limit is not checked; a known one is enforced before the request
    leaves the process.
    """

    #: Bytes the *endpoint* will accept in one HTTP body, which is a different
    #: and much lower limit than the context window on several hosted providers.
    #: Exceeding it is an HTTP 413 that no amount of retrying fixes, and which
    #: reads in a trace exactly like a model that could not do the task.
    max_request_bytes: Optional[int] = None

    # Deliberately the only field.
    #
    # `context_window`, `native_tools` and `prompt_cache` all belong here
    # eventually, and were written and then removed before this shipped, because
    # nothing consults them yet. Native tool calling is already decided by
    # `Talos._advertise_tools`, which duck-types on `engine.tool_schema`; a
    # second, unread source of truth for the same fact is how a descriptor
    # starts lying. This codebase has five mechanisms that were built, tested,
    # and never connected, and each one made a claim the code did not keep.
    #
    # Add a field in the commit that reads it.


@dataclass(frozen=True)
class Provider:
    """A hosted endpoint that speaks the OpenAI chat-completions shape."""
    base_url: str
    key_env: Optional[str]          # None for local servers that need no key
    default_model: str
    note: str = ""
    #: What this endpoint can take. Conservative by construction: a limit that
    #: is set and slightly too low costs one avoidable compaction, while one
    #: that is unset costs a wasted request and an uninterpretable trace.
    profile: EngineProfile = field(default_factory=EngineProfile)


#: Base URLs are stable; **model ids drift constantly**, so treat `default_model`
#: as a starting guess and pass `--model` when it 404s. `OpenAICompatEngine.models()`
#: asks the provider what it actually serves today, which beats trusting this table.
PROVIDERS: Dict[str, Provider] = {
    # Anthropic's OpenAI-compatibility layer, not their native API. That is what
    # lets this engine reach Claude at all without a second backend -- the
    # native message format is a different shape, and `knossos-rs` has a
    # dedicated `anthropic.rs` for it precisely because of that.
    #
    # **Not exercised by any test or run in this repository**, because no
    # `ANTHROPIC_API_KEY` has been available here. The base URL and auth scheme
    # are from Anthropic's documented compatibility endpoint; the model id will
    # drift like every other entry in this table. Treat the first run as the
    # test, and expect the compatibility layer to ignore some parameters it does
    # not implement -- `stream_options` is the one that matters here, and
    # `_NO_STREAM_OPTIONS` already degrades to "no token counts" rather than
    # failing the run if it is refused.
    "anthropic": Provider("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY",
                          "claude-sonnet-5",
                          "Claude via the OpenAI-compatible endpoint; paid, no "
                          "free tier. The compatibility layer ignores what it "
                          "does not implement, so effort, thinking display and "
                          "prompt caching are unavailable here -- knossos-rs "
                          "has a native backend for those."),
    "groq": Provider("https://api.groq.com/openai/v1", "GROQ_API_KEY",
                     "qwen/qwen3.6-27b",
                     "fastest free tier; no card. Also serves "
                     "llama-3.3-70b-versatile and openai/gpt-oss-120b"),
    "nvidia": Provider("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
                       "qwen/qwen3-coder-480b-a35b-instruct",
                       "serves Qwen coder models; low daily cap"),
    # `gemini-2.5-flash` was the default here and 404s: it is still *listed* by
    # `/models` but answers "no longer available to new users", so the list is
    # not a statement about what this key may call. `gemini-3.6-flash` is
    # verified working from this repository; expect it to drift too.
    "gemini": Provider("https://generativelanguage.googleapis.com/v1beta/openai",
                       "GEMINI_API_KEY", "gemini-3.6-flash",
                       "largest free daily cap; 1M context. Free tier is "
                       "requests-per-minute bound, so expect heavy throttling"),
    "cerebras": Provider("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY",
                         "gpt-oss-120b", "high token cap; shrinking model list"),
    "openrouter": Provider("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                           "meta-llama/llama-3.3-70b-instruct:free",
                           "many models, one key; tight free limits"),
    # Alibaba's own endpoint, not a reseller. Note `/compatible-mode/v1`: the
    # plain `/api/v1` on the same host is the native DashScope protocol and will
    # not answer OpenAI-shaped requests.
    #
    # `dashscope-intl` serves keys issued from the international console. A key
    # from the mainland-China console needs `dashscope.aliyuncs.com` instead and
    # will be rejected here -- pass --base-url rather than editing this entry,
    # since which one is correct is a property of the account, not the provider.
    # Default is a *coder* model, not the general one: everything this
    # repository evaluates is a coding task.
    #
    # `qwen3.7-flash` was the obvious pick on price, and is the same trap the
    # Gemini entry above describes. It *is* listed by `/models` -- and asking
    # for it still returned `403 AccessDenied.Unpurchased` on 29 Jul 2026,
    # because entitlement is granted per model in the console and the listing
    # says nothing about it. A `/models` response is a catalogue, not a
    # permission. The only reliable test is a one-token request per candidate.
    #
    # This endpoint also serves other vendors' models -- deepseek-v4-*,
    # kimi-k2.7-code, glm-5.* -- which are reachable through this same entry by
    # passing --model.
    "qwen": Provider("https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                     "DASHSCOPE_API_KEY", "qwen3-coder-flash",
                     "Qwen direct. Coder line: qwen3-coder-flash is the cheap "
                     "one, qwen3-coder-plus and qwen3-coder-480b-a35b-instruct "
                     "are stronger. Prefer a dated id such as "
                     "qwen3-coder-plus-2025-09-23 for a number you intend to "
                     "compare against later -- the undated names move"),
    # Model ids here are paths, not names: `accounts/fireworks/models/<name>`.
    # Passing the bare name is the standard first mistake and returns a 404
    # that reads like the model does not exist.
    #
    # That shape also means `Engine.name` contains slashes. Checked: it reaches
    # display strings only (`acp.py`, `eval.py`) and never a path, so nothing
    # needs to sanitise it. Worth re-checking if a trace file is ever named
    # after the engine.
    # Decimals are spelled with `p`: qwen3p7-plus is Qwen 3.7 Plus,
    # deepseek-v3p1 is v3.1. Searching the catalogue for "3.7" finds nothing.
    #
    # `accounts/fireworks/routers/...` is a separate resource path from
    # `.../models/...` and is not interchangeable with it.
    "fireworks": Provider("https://api.fireworks.ai/inference/v1",
                          "FIREWORKS_API_KEY",
                          "accounts/fireworks/models/deepseek-v4-flash",
                          "frontier models, per-token, no free tier. "
                          "deepseek-v4-flash is the cheap frontier tier; "
                          "kimi-k2p7-code and deepseek-v4-pro are the "
                          "alternatives worth running against it. None of "
                          "these ids carry a date, so none of them are pinned "
                          "-- record the id and the run date with any number "
                          "you intend to compare later"),
    "mistral": Provider("https://api.mistral.ai/v1", "MISTRAL_API_KEY",
                        "mistral-small-latest"),
    "together": Provider("https://api.together.xyz/v1", "TOGETHER_API_KEY",
                         "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "ollama": Provider("http://localhost:11434/v1", None, "qwen2.5-coder:7b",
                       "local; no key, no limits, no data leaves the network. "
                       "Set OLLAMA_HOST to reach another machine"),
    "openwebui": Provider("http://localhost:3000/api", "OPENWEBUI_API_KEY", "",
                          "Open WebUI. Note the /api path, not /v1, and it "
                          "requires a key even on your own LAN. "
                          "Set OPENWEBUI_HOST"),
    "local": Provider("http://localhost:8000/v1", None, "local-model",
                      "llama.cpp / vLLM / LM Studio. Set LOCAL_LLM_HOST"),
}


#: Self-hosted servers live on a homelab box, not localhost. Each reads its host
#: from an environment variable: (variable, default port, API path).
LOCAL_HOSTS = {
    "ollama": ("OLLAMA_HOST", 11434, "/v1"),
    "openwebui": ("OPENWEBUI_HOST", 3000, "/api"),
    "local": ("LOCAL_LLM_HOST", 8000, "/v1"),
}


def _local_base_url(provider: str) -> Optional[str]:
    """Resolve a host variable into a full base URL.

    People write the same address several ways -- `10.0.0.5`, `10.0.0.5:3000`,
    `http://10.0.0.5:3000` -- and all of them mean one thing. Normalising here
    beats letting a missing scheme surface later as an opaque URLError.
    """
    entry = LOCAL_HOSTS.get(provider)
    if not entry:
        return None
    var, port, suffix = entry
    raw = os.environ.get(var, "").strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = "http://" + raw
    raw = raw.rstrip("/")
    if raw.count(":") < 2:                      # scheme colon only: no port given
        raw += f":{port}"
    return raw if raw.endswith(suffix) else raw + suffix


class OpenAICompatEngine:
    """Any endpoint speaking OpenAI `/chat/completions`, which is nearly all of them.

    Groq, Cerebras, OpenRouter, NVIDIA NIM, Together, Mistral and Gemini all
    expose this shape, and so do Ollama, llama.cpp and vLLM locally. One adapter
    covers the lot, so switching provider is a flag rather than a code change --
    which is the same engine-slot argument one level down.

    Deliberately stdlib-only (`urllib`, not `requests` or an SDK): the harness has
    no third-party dependencies and this is not a good enough reason to start.

    The API key is read from the environment, never passed as an argument and
    never logged -- so it stays out of shell history, tracebacks and this repo.
    """

    name = "openai-compat"

    #: Used when retrieval supplied context.
    SYSTEM = (
        "You are Knossos, a coding assistant. You are given excerpts retrieved "
        "from the user's repository, each labelled with its file, line range, and "
        "the reason it was retrieved. Cite file:line when you refer to code. If "
        "the excerpts do not contain the answer, say so instead of guessing."
    )

    #: Used when there is no context. Promising excerpts and supplying none makes
    #: the model ask for them instead of answering -- and in an A/B against the
    #: retrieval condition that handicaps the control and inflates the measured
    #: benefit of retrieval. The two prompts differ only in what they claim is
    #: available.
    SYSTEM_NO_CONTEXT = (
        "You are Knossos, a coding assistant. Answer the question as directly as "
        "you can from what you already know. Cite file:line if you are confident "
        "of a location. If you do not know, say so instead of guessing."
    )

    def __init__(self, provider: str = "groq", model: Optional[str] = None,
                 base_url: Optional[str] = None, key_env: Optional[str] = None,
                 max_tokens: int = 32_000, temperature: float = 0.2,
                 timeout: float = 120.0, max_retries: int = 4,
                 max_backoff: float = 30.0, max_advised_wait: float = 120.0,
                 give_up_after: int = 3, fold_system: bool = False) -> None:
        # 32k, and the reasoning is the same one that took it from 1024 to 4096
        # to 8192: on a reasoning model the `<think>` block is charged against
        # this same budget, so a small cap is spent entirely on thinking and the
        # turn ends truncated and empty -- `EMPTY_REPLY_NOTE` in Talos exists
        # because that was observed. The stronger the model, the longer it
        # reasons, so the cap that was merely tight for a 7B is a hard ceiling
        # for a frontier one.
        #
        # 8192 was still that ceiling for the current frontier line: Claude
        # Sonnet 5 and Opus 5 think *by default* -- omitting the thinking
        # parameter enables it rather than disabling it -- so a budget sized
        # around the answer alone is spent on reasoning before the answer
        # starts. It stays a constructor argument because some providers cap it
        # lower, and `_shrink_to_token_limit` already retries downward against
        # the ones that reject the request outright.
        preset = PROVIDERS.get(provider)
        if preset is None and not base_url:
            raise ValueError(
                f"unknown provider {provider!r}; pass --base-url, or pick one of: "
                f"{', '.join(sorted(PROVIDERS))}")

        self.provider = provider
        # A homelab box is the normal case for a local server, not localhost.
        # OLLAMA_HOST is Ollama's own convention, so honour it rather than
        # inventing a second way to say the same thing.
        if base_url is None:
            base_url = _local_base_url(provider)
        self.base_url = (base_url or preset.base_url).rstrip("/")   # type: ignore[union-attr]
        self.model = model or (preset.default_model if preset else "")
        self.key_env = key_env or (preset.key_env if preset else None)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        #: Ceiling on a *guessed* exponential backoff. Deliberately tight: a
        #: guess that sleeps a minute is worse than failing, because nobody
        #: asked for it.
        self.max_backoff = max_backoff
        #: Ceiling on a wait the provider explicitly asked for, which is a
        #: different thing and deserves more patience. Free tiers routinely say
        #: "retry in 59s" -- measured on Gemini's 20-requests-per-minute tier --
        #: and refusing to wait longer than a guess would means the harness
        #: gives up on a limit that was about to clear. That is how the provider
        #: with the largest free daily allowance ended up unusable.
        self.max_advised_wait = max(max_advised_wait, max_backoff)
        #: Requests that exhausted every retry, back to back. Patience without a
        #: way to notice futility is its own failure: measured on an exhausted
        #: Gemini free tier, the harness spent 480s per case waiting out a
        #: window that was never going to clear, and took 82.7 minutes to
        #: produce ten zeros. The retries were individually correct and the
        #: aggregate was absurd.
        self._consecutive_exhaustions = 0
        #: After this many, stop paying the retry cost and fail immediately.
        #: Three, because one is noise and two is bad luck.
        self.give_up_after = give_up_after
        #: Flipped automatically the first time a model rejects a system role.
        self.fold_system = fold_system
        self.name = f"{provider}:{self.model}"
        #: Set at the end of `generate`. The ACP layer maps it onto a stopReason,
        #: so a reply truncated at the token limit is not reported as a completed
        #: turn. Reset per call.
        self.stop_reason: Optional[str] = None
        #: Native `tool_calls` converted to fenced blocks during the last call.
        #: Reset per call.
        self.native_tool_calls: int = 0
        #: What the last request cost. Reset per call.
        self.usage = Usage()
        #: What every request through this engine has cost. Never reset, so a
        #: caller can read a whole run's spend off the engine it was run with.
        self.total_usage = Usage()
        #: How often, and for how long, this engine sat waiting out a rate limit.
        #:
        #: Cumulative and never reset, because the question they answer is about
        #: a whole run rather than a turn: **was this a measurement of the model,
        #: or of the quota?** A 429 that eventually succeeds leaves no trace in
        #: the reply, so a starved run and a weak model produce the same score
        #: with nothing to tell them apart. Measured on Groq's free tier: one
        #: eval case spent 146 of its 224 seconds asleep in backoff.
        self.throttle_waits = 0
        self.throttled_seconds = 0.0
        #: How often a provider's token ceiling forced the reply allowance down.
        #: A model answering under a shrunken `max_tokens` is being asked a
        #: different question than one that is not.
        self.output_shrinks = 0
        #: What the caller actually asked for, kept so a *transient* shrink can
        #: be undone. Every shrink path below is one-way, and one engine serves
        #: a whole eval suite, so a single rate limit in case 1 silently
        #: constrained every case after it: the score became a function of case
        #: order and of a provider's mood ten minutes earlier.
        self._configured_max_tokens = max_tokens
        #: A **permanent** per-model reply cap learned from a 400, as distinct
        #: from a per-minute rate ceiling learned from a 413. The difference is
        #: the whole reason restoring is safe: a rate window reopens, a model's
        #: maximum never does. Restoring past this would 400 on every request
        #: for the rest of the run, which is strictly worse than staying small.
        self._hard_output_cap: Optional[int] = None
        #: How often `restore_limits` undid one. A run that restores on every
        #: case is a run being throttled on every case, which is worth seeing
        #: rather than quietly smoothing over.
        self.limit_restores = 0
        #: Streaming responses carry no usage block unless it is asked for.
        #: Cleared permanently the first time a provider rejects the parameter,
        #: exactly like `fold_system` -- one wasted request, then never again.
        self._send_stream_options = True
        #: Sticky for the engine's life once a model rejects sampling params.
        self._send_sampling = True
        #: Whether this provider accepts a `tool` role and assistant
        #: `tool_calls`. Assumed yes, since the endpoint it is imitating defines
        #: them; cleared permanently on the first refusal, after which the same
        #: conversation is sent as text.
        self._native_history = True
        #: OpenAI `tools` array, set by whoever owns the tool registry (Talos).
        #: `None` means "do not send one", which is right for a bare question --
        #: the retrieval path has no tools to offer.
        self.tool_schema: Optional[List[Dict[str, Any]]] = None
        #: How many tokens the *server* will accept, when that can be known.
        #: `None` means unknown, and callers must then fall back to a guess.
        #:
        #: This is not the model's advertised maximum. A model card saying 256K
        #: is irrelevant if the server was started with a 16K window, and it is
        #: the server that truncates. Measured on a real deployment: an Ollama
        #: host serving models advertising 128K-256K handed out 16384, while the
        #: harness was budgeting 24000 -- so the front of every long transcript,
        #: including the task statement, was being dropped by the *server*,
        #: silently and after Lethe had already decided the prompt fitted.
        #: Probed on first read, not in `__init__`. Constructing an engine must
        #: not touch the network: the constructor runs in tests, in `--help`,
        #: and before anyone has decided to use it.
        self._context_window: Optional[int] = _int_env("KNOSSOS_CONTEXT_WINDOW")
        self._context_probed = self._context_window is not None

    @property
    def context_window(self) -> Optional[int]:
        """The server's window, discovered lazily. None if it cannot be known.

        Retried while the answer is unknown-but-knowable, because the only
        reliable source needs the model *loaded* and it will not be on the first
        turn. That ordering is benign: the risk this exists to manage is a long
        transcript overflowing the window, which cannot happen on turn one.
        """
        if not self._context_probed:
            self._context_window, retryable = self._discover_context_window()
            self._context_probed = not retryable
        return self._context_window

    @context_window.setter
    def context_window(self, value: Optional[int]) -> None:
        self._context_window = value
        self._context_probed = True

    #: Smallest reply worth asking for. Below this the model cannot finish a
    #: tool call, so shrinking further trades one failure for a worse one.
    MIN_OUTPUT_TOKENS = 512

    #: Headroom left under a provider's stated limit. The limit is enforced on
    #: the provider's tokenizer, not ours, and a request rejected for being
    #: eight tokens over costs a whole round trip.
    _LIMIT_MARGIN = 256

    _LIMIT_RE = re.compile(r"limit\s+(\d+)", re.IGNORECASE)
    _REQUESTED_RE = re.compile(r"requested\s+(\d+)", re.IGNORECASE)

    #: A hard per-*model* cap on the reply allowance, which is a different thing
    #: from the context window and is reported differently: Groq answers HTTP
    #: 400 with "`max_tokens` must be less than or equal to `16384`" while the
    #: same model's window is far larger. Matched tightly on purpose -- 400 is
    #: also how a provider rejects a tool schema, and that path degrades
    #: `_native_history` instead (`:1119`). Widening this pattern would send
    #: those requests down the retry branch and hide the real failure.
    _OUTPUT_CAP_RE = re.compile(
        r"max_tokens.{0,120}?less than or equal to\D{0,8}(\d+)",
        re.IGNORECASE | re.DOTALL)

    def _shrink_to_token_limit(self, code: int, body: str) -> bool:
        """Adapt to a provider's token ceiling. True if the request may be retried.

        Providers enforce a per-request or per-minute token allowance, and a
        request over it is refused outright rather than truncated. Groq answers
        HTTP 413 with the arithmetic spelled out -- `Limit 8000, Requested 8263`
        -- which is enough to fix the request rather than merely report it.

        Two things get adjusted, and the second matters more:

        **`max_tokens`**, so the reply fits in what is left after the prompt.

        **`context_window`**, because the same ceiling bounds the *input* too.
        Talos sizes Lethe's transcript budget from that attribute (§5), so
        telling it here means the next request is built to fit rather than
        failing the same way. Without this the harness would shrink its reply
        allowance and then send a 24 000-token transcript into an 8 000-token
        allowance forever.

        Found by running the eval against Groq's free tier and getting 0/10 with
        every case dying in two steps: raising the default `max_tokens` to 8192
        had put it over an 8 000 limit on its own, before a single prompt token.
        """
        # A per-model reply cap, reported as 400 rather than 413 because the
        # request is malformed rather than too large: nothing about the prompt
        # is wrong, the *allowance asked for* exceeds what this model will ever
        # grant. Clamping is exact -- the provider states the number -- so this
        # needs no margin and no second guess.
        cap = self._OUTPUT_CAP_RE.search(body) if code in (400, 422) else None
        if cap:
            allowed = int(cap.group(1))
            if allowed < self.MIN_OUTPUT_TOKENS or allowed >= self.max_tokens:
                # Either the cap is too small to be worth asking under, or we
                # were already inside it and the 400 is about something else.
                # Retrying an unchanged request is how a bad match becomes a
                # loop, so decline and let the caller report the real error.
                return False
            log(f"[engine] {self.model}: reply capped at {allowed} tokens; "
                f"max_tokens {self.max_tokens} -> {allowed}")
            self.output_shrinks += 1
            # Remembered as permanent. This is a property of the model, not of
            # the minute, so `restore_limits` must never raise back above it --
            # doing so would re-issue the same rejected request forever.
            self._hard_output_cap = (allowed if self._hard_output_cap is None
                                     else min(self._hard_output_cap, allowed))
            self.max_tokens = allowed
            return True

        # 413 only, below. A 429 is the *rate* path, which `_post_with_retry`
        # already handles and which `_is_long_quota` deliberately refuses to
        # retry -- intercepting it here would turn a daily quota into a retry
        # loop.
        if code != 413 or "token" not in body:
            return False
        limit = self._LIMIT_RE.search(body)
        if not limit:
            return False
        allowance = int(limit.group(1))

        if self.context_window is None or self.context_window > allowance:
            self.context_window = allowance

        requested = self._REQUESTED_RE.search(body)
        if requested:
            # What the prompt itself cost, by subtraction -- the only figure
            # here measured with the provider's own tokenizer.
            prompt_tokens = max(0, int(requested.group(1)) - self.max_tokens)
            if prompt_tokens + self.MIN_OUTPUT_TOKENS >= allowance:
                # No reply worth asking for fits alongside this prompt. Retrying
                # with a smaller `max_tokens` would just fail again, more slowly.
                log(f"[engine] {self.model}: the prompt alone is {prompt_tokens} "
                    f"tokens against a {allowance} allowance; transcript budget "
                    f"reduced, request not retried")
                return False
            room = allowance - prompt_tokens - self._LIMIT_MARGIN
        else:
            room = self.max_tokens // 2

        new_max = max(self.MIN_OUTPUT_TOKENS, min(self.max_tokens, room))
        if new_max >= self.max_tokens:
            # The prompt alone does not fit. Shrinking the reply cannot help;
            # the transcript has to come down, which `context_window` above has
            # now told Talos to do.
            log(f"[engine] {self.model}: prompt exceeds the {allowance}-token "
                f"allowance on its own; transcript budget reduced")
            return False
        log(f"[engine] {self.model}: token allowance {allowance}; "
            f"max_tokens {self.max_tokens} -> {new_max}")
        self.output_shrinks += 1
        self.max_tokens = new_max
        return True

    def restore_limits(self) -> bool:
        """Undo transient shrinks. True if anything was actually restored.

        # Why this is needed at all

        Every path in `_shrink_to_token_limit` is one-way, and one engine serves
        a whole eval suite. So a single 413 in the first case left `max_tokens`
        reduced for all eleven after it, long after the per-minute window had
        reopened. The suite's score therefore depended on case order and on
        whatever the provider was doing at the start of the run, degrading
        monotonically as it went -- and nothing in the report said so.

        # Why it is safe

        Two shrinks look alike and are not. A 413 states a **per-minute** token
        allowance: it reopens, and holding the reduction afterwards measures the
        quota rather than the model. A 400 states a **per-model** reply cap:
        that never reopens, and asking above it again would be rejected on every
        request for the rest of the run. `_hard_output_cap` records the second
        kind, and this method treats it as a floor it may not cross.

        The configured value is likewise a ceiling: this restores what the caller
        asked for and never invents headroom above it.

        # Why it is explicit rather than automatic

        Called at a case boundary, never mid-run. Restoring inside a run would
        undo a shrink the very request that caused it needs, and the two would
        oscillate -- one wasted round trip per turn, forever. Once per case the
        cost is bounded at one rejected request, and `limit_restores` makes a
        run that pays it every time visible.
        """
        ceiling = self._configured_max_tokens
        if self._hard_output_cap is not None:
            ceiling = min(ceiling, self._hard_output_cap)
        if self.max_tokens >= ceiling:
            return False
        log(f"[engine] {self.model}: restoring max_tokens "
            f"{self.max_tokens} -> {ceiling}")
        self.max_tokens = ceiling
        self.limit_restores += 1
        return True

    def _discover_context_window(self) -> "tuple[Optional[int], bool]":
        """Ask the server what window it is *serving*. Returns (value, retryable).

        Only Ollama is probed: it is the only backend here that both exposes the
        number and routinely disagrees with the model card. Hosted providers do
        not publish a per-deployment window, so they stay unknown and the caller
        keeps its own default.

        **`/api/ps`, not `/api/show`.** This matters and it is the whole reason
        the method exists. `/api/show` reports what the *weights* support --
        262144 for one of the models measured here -- while the same server was
        serving that model with a 16384 window. Sizing a budget from the model
        card would reproduce exactly the over-promise this is meant to remove,
        only with a bigger number. `/api/ps` reports what the loaded instance
        actually has, which is the only figure that governs truncation.

        The cost is that it needs the model resident: before the first
        generation there is nothing to ask about. That is reported as
        *retryable* rather than as an answer, so the caller asks again next turn
        instead of caching a guess. A server that cannot be reached at all is
        not retryable -- there is no point paying a timeout every turn.
        """
        if "/v1" not in self.base_url:
            return None, False
        root = self.base_url.rsplit("/v1", 1)[0]
        try:
            with urllib.request.urlopen(f"{root}/api/ps", timeout=5) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None, False               # not Ollama, or not reachable

        for entry in body.get("models") or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("model") != self.model and entry.get("name") != self.model:
                continue
            value = entry.get("context_length")
            if isinstance(value, int) and value > 0:
                log(f"[engine] {self.model}: server is serving a {value}-token window")
                return value, False
        # Reachable, spoke Ollama, but this model is not loaded yet.
        return None, True

    # ------------------------------------------------------------------ wiring

    #: urllib defaults to `User-Agent: Python-urllib/3.x`, which Cloudflare-fronted
    #: providers reject outright -- Groq answers such requests with HTTP 403 and a
    #: bare `error code: 1010`, which looks exactly like a bad API key and is not
    #: one. Any ordinary UA gets through.
    USER_AGENT = "knossos/0.1.0"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json",
                   "User-Agent": self.USER_AGENT}
        if self.key_env:
            key = os.environ.get(self.key_env, "").strip()
            if not key:
                raise RuntimeError(
                    f"{self.key_env} is not set. Get a key from the provider and "
                    f"export it before starting the agent -- or set it in the `env` "
                    f"block of the editor's agent config.")
            headers["Authorization"] = f"Bearer {key}"
        return headers

    @property
    def profile(self) -> EngineProfile:
        """What this endpoint can take.

        Resolved in three layers, cheapest override last: the provider table,
        then anything assigned to `.profile`, then `KNOSSOS_MAX_REQUEST_BYTES`
        -- which exists because the limit that matters here is discoverable
        only empirically. A provider returns 413 without publishing a number,
        so the workflow is: hit it once, read the size out of the error, set
        the variable, and never hit it again.
        """
        override = getattr(self, "_profile", None)
        if override is None:
            preset = PROVIDERS.get(self.provider)
            override = preset.profile if preset is not None else EngineProfile()

        env_limit = _int_env("KNOSSOS_MAX_REQUEST_BYTES")
        if env_limit is not None:
            override = replace(override, max_request_bytes=env_limit)
        return override

    @profile.setter
    def profile(self, value: EngineProfile) -> None:
        self._profile = value

    def _post(self, path: str, payload: dict):
        body = json.dumps(payload).encode("utf-8")

        # Remembered so `_explain` can name the size that failed. A 413 whose
        # message is "see the agent log" tells you nothing you can act on; one
        # that says "you sent 4.2 MB" tells you both what happened and what to
        # set `max_request_bytes` to.
        self._last_request_bytes = len(body)

        limit = self.profile.max_request_bytes
        if limit is not None and len(body) > limit:
            # Refused here rather than by the provider. The round trip is
            # wasted either way, but a local refusal is attributable: it names
            # the harness as the thing that overflowed, where a 413 arrives as
            # an opaque provider failure that reads in a trace exactly like a
            # model that could not do the task.
            raise RequestTooLarge(len(body), limit)

        req = urllib.request.Request(
            f"{self.base_url}{path}", method="POST",
            data=body, headers=self._headers())
        return urllib.request.urlopen(req, timeout=self.timeout)

    @property
    def exhausted(self) -> bool:
        """Whether this engine has stopped being worth asking.

        Set once enough requests in a row have burned every retry. Cleared by
        any successful response, so a provider that recovers is used again.
        """
        return self._consecutive_exhaustions >= self.give_up_after

    def _post_with_retry(self, path: str, payload: dict, cancelled: Cancelled):
        """POST, backing off on 429.

        Free tiers are usually capped on *tokens* per minute rather than
        requests, so a couple of long-context calls can exhaust the window even
        at a modest request rate. That limit clears on its own in seconds, which
        makes failing the turn the wrong response -- waiting is.

        Honours `Retry-After` when the provider sends one, since a guess is
        strictly worse than being told.
        """
        if self.exhausted:
            # Every retry from here would wait the same window and fail the same
            # way. Say so at once instead of spending another few minutes
            # proving it -- see `_consecutive_exhaustions`.
            raise urllib.error.HTTPError(
                f"{self.base_url}{path}", 429, "Too Many Requests", {},   # type: ignore[arg-type]
                io.BytesIO(json.dumps({"error": {
                    "message": f"giving up: {self.give_up_after} consecutive requests "
                               f"exhausted every retry against {self.provider}. "
                               f"The limit is not clearing."}}).encode("utf-8")))

        for attempt in range(self.max_retries + 1):
            try:
                response = self._post(path, payload)
                self._consecutive_exhaustions = 0      # it is answering again
                return response
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == self.max_retries:
                    if exc.code == 429:
                        self._consecutive_exhaustions += 1
                        if self.exhausted:
                            log(f"[engine] {self.provider}: "
                                f"{self._consecutive_exhaustions} requests in a row "
                                f"exhausted every retry; giving up on this run")
                    raise
                # A per-minute window is worth waiting out. A daily quota is not:
                # backing off 4x30s against a limit that resets in 38 minutes
                # burns two minutes per call to fail anyway. The provider says
                # which it is -- believe it and fail fast.
                if self._is_long_quota(exc):
                    raise
                # `is not None`, not truthiness: a stated 0 means "now", and
                # treating that as "no advice" would sleep for nothing. The
                # delay may come from the header or from the body -- Gemini
                # states it only in the body, so header-only reading turned
                # every one of its 429s into a guess.
                advised = self._retry_delay(exc)
                # An explicit instruction gets the longer leash; a guess does
                # not. Capping both at `max_backoff` meant a stated 59-second
                # window -- which was about to clear -- was treated as a wait
                # too long to bother with.
                wait = (min(advised, self.max_advised_wait) if advised is not None
                        else min(2.0 ** attempt, self.max_backoff))
                log(f"[engine] 429 from {self.provider}; waiting {wait:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})")
                # Counted before the sleep, so a run cancelled mid-wait still
                # records that it was throttled.
                self.throttle_waits += 1
                self.throttled_seconds += wait
                deadline = time.monotonic() + wait
                while time.monotonic() < deadline:
                    if cancelled():
                        return None
                    time.sleep(0.1)
        return None

    def models(self) -> List[str]:
        """Ask the provider what it serves today. Beats trusting a hardcoded id."""
        req = urllib.request.Request(f"{self.base_url}/models", headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return sorted(m.get("id", "") for m in body.get("data", []))

    # -------------------------------------------------------------- generation

    #: Body text meaning "this model's template has no system role". Gemma is the
    #: common case: several serving stacks reject a system message outright
    #: rather than merging it, and the request fails with a 400.
    _NO_SYSTEM = ("system role not supported", "does not support system",
                  "system messages are not", "only user and assistant",
                  "system instruction")

    #: A provider whose OpenAI-compatible surface does not accept a `tool` role
    #: or an assistant `tool_calls` array. Rarer than the others -- most servers
    #: that advertise the endpoint accept the whole schema -- but a run must not
    #: fail over the conversation's *shape* when the same conversation renders
    #: perfectly well as text.
    _NO_TOOL_MESSAGES = ("tool_calls", "tool_call_id", "role 'tool'",
                         'role "tool"', "invalid role", "unsupported role",
                         "tool messages", "tool role",
                         # Gemini 3.x mints an opaque `thought_signature` on
                         # every function call and requires it echoed back when
                         # that call is replayed in history. The OpenAI-compat
                         # shape has nowhere to carry it, so a native
                         # tool-calling conversation is rejected from the second
                         # turn onward. Measured against gemini-3.1-pro-preview:
                         # 15 of ~30 engine calls in a 5-case run were 400s, and
                         # the provider's own message warns that the missing
                         # signature "may lead to degraded model performance" --
                         # so the run was not merely wasteful, it was handicapped.
                         # Falling back to a flattened text history is the same
                         # answer this engine gives every other provider quirk,
                         # and it costs one wasted request per engine rather
                         # than one per turn.
                         "thought_signature", "thought signature")

    #: A provider refusing `stream_options`. OpenAI-compatible servers vary in
    #: how strictly they validate unknown request fields: most ignore them,
    #: some reject the request outright, and the message names the field.
    _NO_STREAM_OPTIONS = ("stream_options", "stream_option",
                          "unknown field", "unrecognized field",
                          "extra fields not permitted",
                          "additional properties are not allowed")

    #: A model refusing sampling parameters outright.
    #:
    #: Claude Sonnet 5 and the Opus 4.7+ line removed `temperature`, `top_p` and
    #: `top_k`: a non-default value is a 400, not a silently ignored field. This
    #: harness sends `temperature=0.2` on every request, so without this every
    #: turn against those models fails before the model is ever reached -- and
    #: `codeval` would record the outcome as a capability score, which is the
    #: exact failure `CaseResult.unreachable` exists to prevent.
    #:
    #: Detected rather than tabulated by model id. Model ids drift constantly
    #: (`PROVIDERS` says so itself), and a hardcoded list would be wrong for the
    #: next model in the line. One wasted request per engine buys a fact that
    #: holds regardless.
    _NO_SAMPLING = ("temperature", "top_p", "top_k", "sampling")

    def _messages(self, user: str, fold_system: bool, system: str) -> List[dict]:
        if fold_system:
            return [{"role": "user", "content": f"{system}\n\n{user}"}]
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    def generate(self, prompt: str, context: str, cancelled: Cancelled) -> Iterator[str]:
        has_context = bool(str(context).strip())
        system = self.SYSTEM if has_context else self.SYSTEM_NO_CONTEXT
        user = (f"{prompt}\n\n<repository_excerpts>\n{context}\n</repository_excerpts>"
                if has_context else prompt)
        yield from self._run(self._messages(user, self.fold_system, system),
                             cancelled, refold=(user, system))

    def generate_messages(self, messages: Sequence[Dict[str, Any]],
                          cancelled: Cancelled) -> Iterator[str]:
        """Generate from a caller-built message array.

        The optional half of the engine slot. `generate` exists because the slot
        must one day hold a model that only maps bytes to bytes, and flattening
        a conversation into one string is what that costs. This is for the
        engines that do better, and Talos prefers it when present.

        Two things change for a model that gets this instead:

        **Its own prior turns arrive as `assistant` messages.** Flattened, they
        are a `## Assistant` heading inside one enormous user turn, which is not
        the shape any instruction-tuned model was trained on -- the model is
        being asked to infer the conversation structure from markdown.

        **The prefix stops changing.** The system message holds the role, the
        constitution and the tool schema, and is byte-identical on every step of
        a run, so a provider's prompt cache can hit it. Flattened, the system
        text is concatenated with a transcript that grows every turn, so nothing
        is ever cacheable and every step is billed as fresh input.
        """
        history = list(messages)
        # A provider already known to refuse them never gets sent them again.
        if not self._native_history:
            history = _flatten_tool_messages(history)
        yield from self._run(history, cancelled, refold=None)

    def _run(self, messages: List[Dict[str, Any]], cancelled: Cancelled,
             refold: Optional[tuple]) -> Iterator[str]:
        """POST a message array and stream the reply.

        `refold` carries the `(user, system)` pair needed to retry without a
        system role, for the providers that reject one; `None` means the caller
        built the array itself and a retry would have to rebuild it, so that
        path folds in place instead.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if self._send_sampling:
            payload["temperature"] = self.temperature
        # Only when the caller supplied one. A bare question needs no tools, and
        # sending an empty array is not the same request as sending none.
        if self.tool_schema:
            payload["tools"] = self.tool_schema
        # A streamed response omits `usage` unless this asks for it, so without
        # this the harness cannot cost a single turn. Dropped permanently on the
        # first provider that refuses it, below.
        if self._send_stream_options:
            payload["stream_options"] = {"include_usage": True}
        self.stop_reason = None
        self.native_tool_calls = 0
        self.usage = Usage()

        try:
            response = self._post_with_retry("/chat/completions", payload, cancelled)
            if response is None:                # cancelled while backing off
                self.stop_reason = "cancelled"
                return
        except urllib.error.HTTPError as exc:
            body = self._body(exc).lower()
            if self._shrink_to_token_limit(exc.code, body):
                payload["max_tokens"] = self.max_tokens
                try:
                    response = self._post_with_retry("/chat/completions", payload,
                                                     cancelled)
                    if response is None:
                        self.stop_reason = "cancelled"
                        return
                except urllib.error.HTTPError as retry_exc:
                    yield self._explain(retry_exc)
                    return
            elif exc.code in (400, 422) and self._native_history \
                    and any(m in body for m in self._NO_TOOL_MESSAGES) \
                    and any(m.get("role") == "tool" or m.get("tool_calls")
                            for m in messages):
                # Checked against the messages actually sent, not just the error
                # text: "tool_calls" appears in plenty of unrelated complaints,
                # and flattening a conversation that had none would be a wrong
                # diagnosis that hides the real one.
                log(f"[engine] {self.provider} rejects native tool messages; "
                    f"resending the conversation as text")
                self._native_history = False
                payload["messages"] = _flatten_tool_messages(messages)
                try:
                    response = self._post_with_retry("/chat/completions", payload,
                                                     cancelled)
                    if response is None:
                        self.stop_reason = "cancelled"
                        return
                except urllib.error.HTTPError as retry_exc:
                    yield self._explain(retry_exc)
                    return
            elif exc.code in (400, 422) and self._send_sampling \
                    and any(m in body for m in self._NO_SAMPLING):
                # Checked before the stream_options branch: both are 400s about
                # a rejected field, and "unknown field" matches either, so the
                # narrower diagnosis has to win or a sampling refusal is
                # misread as a stream_options one and retried unchanged.
                log(f"[engine] {self.model} rejects sampling parameters; "
                    f"resending without temperature")
                self._send_sampling = False
                payload.pop("temperature", None)
                try:
                    response = self._post_with_retry("/chat/completions", payload,
                                                     cancelled)
                    if response is None:
                        self.stop_reason = "cancelled"
                        return
                except urllib.error.HTTPError as retry_exc:
                    yield self._explain(retry_exc)
                    return
            elif exc.code in (400, 422) and self._send_stream_options \
                    and any(m in body for m in self._NO_STREAM_OPTIONS):
                # Costing a turn is worth one wasted request to discover, and
                # nothing beyond that: the flag is sticky for the engine's life.
                # Never worth failing a run over -- an unmeasured turn is a
                # smaller loss than a turn that did not happen.
                log(f"[engine] {self.provider} rejects stream_options; "
                    f"continuing without per-request token counts")
                self._send_stream_options = False
                payload.pop("stream_options", None)
                try:
                    response = self._post_with_retry("/chat/completions", payload,
                                                     cancelled)
                    if response is None:
                        self.stop_reason = "cancelled"
                        return
                except urllib.error.HTTPError as retry_exc:
                    yield self._explain(retry_exc)
                    return
            elif exc.code in (400, 422) and any(m in body for m in self._NO_SYSTEM) \
                    and not self.fold_system:
                log(f"[engine] {self.model} rejects a system role; "
                    f"folding it into the user message")
                self.fold_system = True         # remember, so this costs one request
                if refold is not None:
                    payload["messages"] = self._messages(refold[0], True, refold[1])
                else:
                    payload["messages"] = _fold_system(messages)
                try:
                    response = self._post_with_retry("/chat/completions", payload,
                                                     cancelled)
                    if response is None:
                        self.stop_reason = "cancelled"
                        return
                except urllib.error.HTTPError as retry_exc:
                    yield self._explain(retry_exc)
                    return
            elif exc.code in (400, 422) and self._send_stream_options \
                    and "stream_options" in payload:
                # Last resort, and it exists because of what this parameter is:
                # something *this harness* started adding to every request, for
                # a number nobody asked for. The marker list above is a guess at
                # how a provider phrases the refusal, and a guess that misses
                # turns a working setup into a failing one -- a bad trade for
                # token accounting.
                #
                # So any unexplained 400 gets one retry without it. If that
                # succeeds the parameter was the problem; if it fails the real
                # error is reported, one wasted request later, on a path that
                # was already failing.
                log(f"[engine] {self.provider} rejected the request; retrying "
                    f"once without stream_options in case that is why")
                self._send_stream_options = False
                payload.pop("stream_options", None)
                try:
                    response = self._post_with_retry("/chat/completions", payload,
                                                     cancelled)
                    if response is None:
                        self.stop_reason = "cancelled"
                        return
                except urllib.error.HTTPError:
                    # The original error is the one worth reporting: it is what
                    # the caller asked about, and it is not about this field.
                    yield self._explain(exc)
                    return
            else:
                yield self._explain(exc)
                return
        except urllib.error.URLError as exc:
            log(f"[engine] cannot reach {self.base_url}: {exc.reason}")
            yield (f"Could not reach {self.base_url} ({exc.reason}). "
                   f"If this is a local server, check that it is running.")
            return

        splitter = ThinkSplitter()
        #: index -> {"name": str, "arguments": str}, assembled across deltas.
        pending_calls: Dict[Any, Dict[str, str]] = {}
        with response:
            for raw in response:
                if cancelled():
                    self.stop_reason = "cancelled"
                    return
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue                        # SSE comments and blank keepalives
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # `include_usage` appends a final chunk carrying totals and an
                # empty `choices`. Some providers instead repeat a running total
                # on every chunk, so this assigns rather than accumulates -- the
                # last report is the whole request either way, and adding them up
                # would multiply the cost of every turn by its chunk count.
                if chunk.get("usage"):
                    self.usage = Usage.from_payload(chunk["usage"])
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}

                # Some providers expose reasoning on its own field rather than
                # inline in <think> tags. Both mean the same thing here.
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    yield Thought(reasoning)

                piece = delta.get("content")
                if piece:
                    for is_thought, text in splitter.feed(piece):
                        yield Thought(text) if is_thought else text

                # A model may answer with OpenAI's native tool-call format
                # instead of the fenced block Knossos asks for -- especially now
                # that the schema is sent in the request. The arguments arrive
                # in fragments across deltas, keyed by index.
                for part in delta.get("tool_calls") or []:
                    self._accumulate_tool_call(pending_calls, part)

                if choice.get("finish_reason"):
                    self.stop_reason = choice["finish_reason"]

        # Banked once the response is fully read, so a cancelled or failed
        # request contributes nothing rather than a partial figure.
        self.total_usage = self.total_usage + self.usage

        for is_thought, text in splitter.flush():
            yield Thought(text) if is_thought else text

        # Normalised to the fenced block the rest of the pipeline already
        # understands, so one convention reaches Talos however the model chose
        # to answer. Emitted after the content so a model that produced both
        # keeps its prose.
        for block in self._render_native_calls(pending_calls):
            yield block

    @staticmethod
    def _accumulate_tool_call(pending: Dict[Any, Dict[str, str]],
                              part: Dict[str, Any]) -> None:
        """Fold one streamed `tool_calls` fragment into the call it belongs to.

        Only `index` is reliably present on every fragment; `id` and the
        function name usually arrive once, on the first, and the arguments
        arrive as a string split at arbitrary points.
        """
        if not isinstance(part, dict):
            return
        slot = pending.setdefault(part.get("index", len(pending)),
                                  {"name": "", "arguments": "", "id": ""})
        # Older accumulations may predate the id slot; a plain `setdefault` on the
        # outer dict would not add it to one already present.
        slot.setdefault("id", "")
        if part.get("id"):
            slot["id"] = str(part["id"])
        function = part.get("function") or {}
        if function.get("name"):
            slot["name"] = str(function["name"])
        if function.get("arguments"):
            slot["arguments"] += str(function["arguments"])

    def _render_native_calls(self, pending: Dict[Any, Dict[str, str]]) -> List[str]:
        """Turn assembled native calls into the fenced blocks Talos parses.

        Fails closed, like `parse_calls`: a call whose arguments are not valid
        JSON is dropped with a log line rather than guessed at, because the
        alternative is running the wrong action with half an argument.
        """
        blocks: List[str] = []
        for _index, slot in sorted(pending.items(), key=lambda kv: str(kv[0])):
            if not slot["name"]:
                continue
            raw = slot["arguments"].strip() or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                log(f"[engine] {self.model} sent tool call {slot['name']} with "
                    f"unparseable arguments; dropping it: {raw[:200]!r}")
                continue
            if not isinstance(args, dict):
                log(f"[engine] {self.model} sent tool call {slot['name']} whose "
                    f"arguments are {type(args).__name__}, not an object; dropping it")
                continue
            self.native_tool_calls += 1
            block: Dict[str, Any] = {"tool": slot["name"], "args": args}
            # Carried through the text convention so the call can be paired with
            # its result when the conversation is sent back. Omitted rather than
            # invented when the provider did not supply one -- a fabricated id
            # matches nothing and is worse than none.
            if slot.get("id"):
                block["id"] = slot["id"]
            blocks.append("\n```json\n" + json.dumps(block) + "\n```\n")
        if blocks:
            log(f"[engine] {self.model} answered with {len(blocks)} native tool "
                f"call(s); converted to the fenced-block convention.")
        return blocks

    #: Markers for a quota that will not clear within a sensible backoff.
    _LONG_QUOTA = ("per day", "tpd", "daily", "per-day", "monthly", "quota exceeded",
                   "insufficient_quota", "credit")

    @staticmethod
    def _body(exc: urllib.error.HTTPError) -> str:
        """Read an error body once and cache it on the exception.

        `HTTPError.read()` is a one-shot stream: whoever calls it first gets the
        text and every later caller gets nothing. Both the quota check and the
        user-facing explanation need it, so it is read once and kept.
        """
        cached = getattr(exc, "_harness_body", None)
        if cached is None:
            try:
                cached = exc.read().decode("utf-8", errors="replace")
            except Exception:
                cached = ""
            try:
                exc._harness_body = cached          # type: ignore[attr-defined]
            except Exception:
                pass
        return cached

    #: A retry delay stated in the *body* rather than the `Retry-After` header.
    #: Gemini sends `"Please retry in 27.03s."` and a `RetryInfo` detail with
    #: `retryDelay`, and no header at all -- so a harness that only reads the
    #: header sees a 429 with no advice and falls back to guessing.
    _BODY_RETRY_RE = re.compile(
        r'(?:retry in|"?retrydelay"?\s*[:=]\s*")\s*([0-9]+(?:\.[0-9]+)?)\s*s',
        re.IGNORECASE)

    #: The provider saying this model is not available on this tier at all.
    #: Terminal in a way waiting cannot fix.
    _ZERO_LIMIT_RE = re.compile(r"limit:\s*0\b", re.IGNORECASE)

    def _retry_delay(self, exc: urllib.error.HTTPError) -> Optional[float]:
        """Seconds the provider asked us to wait, from the header or the body."""
        header = (exc.headers or {}).get("Retry-After")
        try:
            if header is not None:
                return float(header)
        except (TypeError, ValueError):
            pass
        found = self._BODY_RETRY_RE.search(self._body(exc))
        return float(found.group(1)) if found else None

    def _is_long_quota(self, exc: urllib.error.HTTPError) -> bool:
        """True when a 429 is worth giving up on rather than waiting out.

        Order matters, because the cheap substring test is the least reliable
        signal and used to run first.

        **A zero limit is terminal.** `limit: 0` means the model is not served
        on this tier -- measured against `gemini-3.1-pro-preview`, which is
        simply not a free-tier model. No amount of waiting changes that.

        **An advertised delay outranks any guess.** If the provider says when to
        come back, believe it: a stated 27 seconds is a short window whatever
        words surround it.

        **The substring test is the fallback**, and it is why this needed
        fixing. `_LONG_QUOTA` contains `"quota exceeded"`, and Gemini uses that
        exact phrase for *per-minute* limits -- so every Gemini 429 was
        classified as a daily cap and the run gave up rather than waiting out a
        27-second window. That made the provider with the largest free daily
        allowance unusable.
        """
        text = self._body(exc).lower()
        if self._ZERO_LIMIT_RE.search(text):
            return True

        delay = self._retry_delay(exc)
        if delay is not None:
            return delay > self.max_advised_wait

        return any(m in text for m in self._LONG_QUOTA)

    def _explain(self, exc: urllib.error.HTTPError) -> str:
        """Surface the failure in the editor *and* the log. Never swallow it."""
        detail = self._body(exc)[:500]
        log(f"[engine] {self.base_url} returned HTTP {exc.code}: {detail}")

        if exc.code == 429 and self._ZERO_LIMIT_RE.search(detail):
            hint = (f"{self.model!r} is not served on this account's tier "
                    f"(the provider reports a limit of 0) -- switch --model, or "
                    f"use --provider ollama")
        elif exc.code == 429 and any(m in detail.lower() for m in self._LONG_QUOTA):
            hint = ("daily or monthly quota exhausted -- retrying will not help; "
                    "wait for the window to reset, switch --model, or use "
                    "--provider ollama")
        elif exc.code in (401, 403):
            hint = f"the key in ${self.key_env} was rejected"
        elif exc.code == 404:
            hint = (f"model {self.model!r} was not found -- model ids change often; "
                    f"run `python -m knossos --list-models --provider {self.provider}`")
        elif exc.code == 429:
            hint = "rate limit or free-tier daily cap reached"
        elif exc.code == 413:
            # The prompt outgrew the endpoint's HTTP body limit, which on
            # several providers is far below the model's context window. This
            # is the harness's fault, not the model's, and retrying cannot fix
            # it -- but before this branch existed it surfaced as "see the agent
            # log", which is indistinguishable in a trace from a model that
            # failed the task. Naming the measured size makes the limit
            # discoverable in one run.
            # `exc.size` when the harness refused it locally, which is exact.
            # Otherwise the size of the last body actually put on the wire.
            sent = getattr(exc, "size", None) or getattr(
                self, "_last_request_bytes", 0)
            size = _human_bytes(sent) if sent else "an unknown size"
            hint = (f"the request body ({size}) exceeded what {self.provider} "
                    f"accepts -- this is a prompt-size problem, not a model "
                    f"one, and retrying will not help. Lower the context "
                    f"budget, or set the provider's `max_request_bytes` to "
                    f"just under {sent} so the harness compacts instead of "
                    f"discovering this per request")
        else:
            hint = "see the agent log for the full response"
        return f"Request failed: HTTP {exc.code} — {hint}."


class TransformersEngine:
    """A HuggingFace causal LM in the slot -- the Track B path (Qwen2.5-Coder + QLoRA).

    NOT exercised by the test suite: this machine has no torch, so the code below
    is unrun. Treat it as a starting point to verify on a GPU box, not as a
    working component. `RetrievalOnlyEngine` is the one that is tested.

    `adapter` points at a PEFT/LoRA directory to layer on the base weights.
    """

    name = "transformers"

    SYSTEM = (
        "You are Knossos, a coding assistant. You are given excerpts retrieved "
        "from the user's repository, each labelled with its file, line range, and "
        "the reason it was retrieved. Cite file:line when you refer to code. If "
        "the excerpts do not contain the answer, say so instead of guessing."
    )

    def __init__(self, model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
                 adapter: Optional[str] = None, max_new_tokens: int = 512,
                 device_map: str = "auto") -> None:
        self.model_id = model_id
        self.adapter = adapter
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: PLC0415

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map=self.device_map)
        if self.adapter:
            from peft import PeftModel                                  # noqa: PLC0415
            self._model = PeftModel.from_pretrained(self._model, self.adapter)
        self._model.eval()

    def generate(self, prompt: str, context: str, cancelled: Cancelled) -> Iterator[str]:
        from threading import Thread                                    # noqa: PLC0415
        from transformers import TextIteratorStreamer                   # noqa: PLC0415

        self._load()
        assert self._tokenizer is not None and self._model is not None

        user = f"{prompt}\n\n<repository_excerpts>\n{context}\n</repository_excerpts>" \
            if context.strip() else prompt
        text = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": self.SYSTEM},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        thread = Thread(target=self._model.generate,
                        kwargs=dict(**inputs, streamer=streamer,
                                    max_new_tokens=self.max_new_tokens))
        thread.start()
        for chunk in streamer:
            if cancelled():
                break
            if chunk:
                yield chunk
