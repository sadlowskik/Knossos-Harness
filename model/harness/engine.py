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

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Protocol, runtime_checkable

from .jsonrpc import log

__all__ = ["Engine", "RetrievalOnlyEngine", "TransformersEngine", "StaticEngine",
           "OpenAICompatEngine", "Provider", "PROVIDERS", "Thought", "ThinkSplitter"]

Cancelled = Callable[[], bool]


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


@runtime_checkable
class Engine(Protocol):
    """Anything that can turn a prompt plus context into a stream of text."""

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


@dataclass(frozen=True)
class Provider:
    """A hosted endpoint that speaks the OpenAI chat-completions shape."""
    base_url: str
    key_env: Optional[str]          # None for local servers that need no key
    default_model: str
    note: str = ""


#: Base URLs are stable; **model ids drift constantly**, so treat `default_model`
#: as a starting guess and pass `--model` when it 404s. `OpenAICompatEngine.models()`
#: asks the provider what it actually serves today, which beats trusting this table.
PROVIDERS: Dict[str, Provider] = {
    "groq": Provider("https://api.groq.com/openai/v1", "GROQ_API_KEY",
                     "qwen/qwen3.6-27b",
                     "fastest free tier; no card. Also serves "
                     "llama-3.3-70b-versatile and openai/gpt-oss-120b"),
    "nvidia": Provider("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
                       "qwen/qwen3-coder-480b-a35b-instruct",
                       "serves Qwen coder models; low daily cap"),
    "gemini": Provider("https://generativelanguage.googleapis.com/v1beta/openai",
                       "GEMINI_API_KEY", "gemini-2.5-flash",
                       "largest free daily cap; 1M context"),
    "cerebras": Provider("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY",
                         "gpt-oss-120b", "high token cap; shrinking model list"),
    "openrouter": Provider("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                           "meta-llama/llama-3.3-70b-instruct:free",
                           "many models, one key; tight free limits"),
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
        "You are Daedalus, a coding assistant. You are given excerpts retrieved "
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
        "You are Daedalus, a coding assistant. Answer the question as directly as "
        "you can from what you already know. Cite file:line if you are confident "
        "of a location. If you do not know, say so instead of guessing."
    )

    def __init__(self, provider: str = "groq", model: Optional[str] = None,
                 base_url: Optional[str] = None, key_env: Optional[str] = None,
                 max_tokens: int = 4096, temperature: float = 0.2,
                 timeout: float = 120.0, max_retries: int = 4,
                 max_backoff: float = 30.0, fold_system: bool = False) -> None:
        # 4096, not 1024: on a reasoning model the `<think>` block is charged
        # against the same budget, and a small cap gets spent entirely on
        # thinking -- the turn then ends with a truncated, empty reply.
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
        self.max_backoff = max_backoff
        #: Flipped automatically the first time a model rejects a system role.
        self.fold_system = fold_system
        self.name = f"{provider}:{self.model}"
        #: Set at the end of `generate`. The ACP layer maps it onto a stopReason,
        #: so a reply truncated at the token limit is not reported as a completed
        #: turn. Reset per call.
        self.stop_reason: Optional[str] = None

    # ------------------------------------------------------------------ wiring

    #: urllib defaults to `User-Agent: Python-urllib/3.x`, which Cloudflare-fronted
    #: providers reject outright -- Groq answers such requests with HTTP 403 and a
    #: bare `error code: 1010`, which looks exactly like a bad API key and is not
    #: one. Any ordinary UA gets through.
    USER_AGENT = "daedalus-harness/0.1.0"

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

    def _post(self, path: str, payload: dict):
        req = urllib.request.Request(
            f"{self.base_url}{path}", method="POST",
            data=json.dumps(payload).encode("utf-8"), headers=self._headers())
        return urllib.request.urlopen(req, timeout=self.timeout)

    def _post_with_retry(self, path: str, payload: dict, cancelled: Cancelled):
        """POST, backing off on 429.

        Free tiers are usually capped on *tokens* per minute rather than
        requests, so a couple of long-context calls can exhaust the window even
        at a modest request rate. That limit clears on its own in seconds, which
        makes failing the turn the wrong response -- waiting is.

        Honours `Retry-After` when the provider sends one, since a guess is
        strictly worse than being told.
        """
        for attempt in range(self.max_retries + 1):
            try:
                return self._post(path, payload)
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == self.max_retries:
                    raise
                # A per-minute window is worth waiting out. A daily quota is not:
                # backing off 4x30s against a limit that resets in 38 minutes
                # burns two minutes per call to fail anyway. The provider says
                # which it is -- believe it and fail fast.
                if self._is_long_quota(exc):
                    raise
                header = (exc.headers or {}).get("Retry-After")
                advised: Optional[float]
                try:
                    # `is not None`, not truthiness: `Retry-After: 0` means "now",
                    # and treating that 0 as "no advice" would sleep for nothing.
                    advised = float(header) if header is not None else None
                except (TypeError, ValueError):
                    advised = None
                wait = min(advised if advised is not None else 2.0 ** attempt,
                           self.max_backoff)
                log(f"[engine] 429 from {self.provider}; waiting {wait:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})")
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
        payload = {
            "model": self.model,
            "messages": self._messages(user, self.fold_system, system),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        self.stop_reason = None

        try:
            response = self._post_with_retry("/chat/completions", payload, cancelled)
            if response is None:                # cancelled while backing off
                self.stop_reason = "cancelled"
                return
        except urllib.error.HTTPError as exc:
            body = self._body(exc).lower()
            if exc.code in (400, 422) and any(m in body for m in self._NO_SYSTEM) \
                    and not self.fold_system:
                log(f"[engine] {self.model} rejects a system role; "
                    f"folding it into the user message")
                self.fold_system = True         # remember, so this costs one request
                payload["messages"] = self._messages(user, True, system)
                try:
                    response = self._post_with_retry("/chat/completions", payload,
                                                     cancelled)
                    if response is None:
                        self.stop_reason = "cancelled"
                        return
                except urllib.error.HTTPError as retry_exc:
                    yield self._explain(retry_exc)
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

                if choice.get("finish_reason"):
                    self.stop_reason = choice["finish_reason"]

        for is_thought, text in splitter.flush():
            yield Thought(text) if is_thought else text

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

    def _is_long_quota(self, exc: urllib.error.HTTPError) -> bool:
        """True when a 429 is a daily/monthly cap rather than a short window."""
        text = self._body(exc).lower()
        if any(m in text for m in self._LONG_QUOTA):
            return True
        header = (exc.headers or {}).get("Retry-After")
        try:
            return header is not None and float(header) > self.max_backoff
        except (TypeError, ValueError):
            return False

    def _explain(self, exc: urllib.error.HTTPError) -> str:
        """Surface the failure in the editor *and* the log. Never swallow it."""
        detail = self._body(exc)[:500]
        log(f"[engine] {self.base_url} returned HTTP {exc.code}: {detail}")

        if exc.code == 429 and any(m in detail.lower() for m in self._LONG_QUOTA):
            hint = ("daily or monthly quota exhausted -- retrying will not help; "
                    "wait for the window to reset, switch --model, or use "
                    "--provider ollama")
        elif exc.code in (401, 403):
            hint = f"the key in ${self.key_env} was rejected"
        elif exc.code == 404:
            hint = (f"model {self.model!r} was not found -- model ids change often; "
                    f"run `python -m harness --list-models --provider {self.provider}`")
        elif exc.code == 429:
            hint = "rate limit or free-tier daily cap reached"
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
        "You are Daedalus, a coding assistant. You are given excerpts retrieved "
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
