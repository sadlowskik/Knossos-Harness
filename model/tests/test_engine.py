"""Tests for the OpenAI-compatible engine, against a real local HTTP server.

The risky part is not the request shape, it is the streaming parse: server-sent
events arrive as arbitrarily chunked lines, with keepalives, comments and a
sentinel mixed in, and any provider may differ slightly in what it interleaves.
So these tests serve genuine SSE over a socket rather than stubbing `urlopen`.

The other claim worth pinning down is that failures are reported rather than
swallowed: a 429 must reach the user as a rate-limit message, not as silence or
an empty turn.

    pytest -q tests/test_engine.py
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from harness import OpenAICompatEngine, PROVIDERS, Thought, ThinkSplitter
from harness.engine import LOCAL_HOSTS

NEVER = lambda: False          # noqa: E731 -- "not cancelled", for readability


def sse(*chunks: str) -> bytes:
    """Frame text deltas the way an OpenAI-compatible endpoint does."""
    out = []
    for text in chunks:
        payload = {"choices": [{"delta": {"content": text}}]}
        out.append(f"data: {json.dumps(payload)}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    body = sse("hello ", "world")
    status = 200
    captured: dict = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        Handler.captured["payload"] = json.loads(raw)
        Handler.captured["auth"] = self.headers.get("Authorization")
        Handler.captured["ua"] = self.headers.get("User-Agent")
        self.send_response(Handler.status)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(Handler.body)

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(
            {"data": [{"id": "model-b"}, {"id": "model-a"}]}).encode())

    def log_message(self, *args):
        pass                                    # keep pytest output clean


@pytest.fixture(scope="module")
def _server():
    """One server for the module.

    `ThreadingHTTPServer.shutdown()` polls on a 0.5s interval, so a per-test
    server spends that on every teardown -- which was most of this file's
    runtime. The handler's mutable state is reset per test instead.
    """
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


@pytest.fixture()
def server(_server):
    Handler.body = sse("hello ", "world")
    Handler.status = 200
    Handler.captured = {}
    return _server


def engine_for(server, **kw):
    return OpenAICompatEngine(provider="local", base_url=server, key_env=None,
                              model="test-model", **kw)


# ----------------------------------------------------------------- streaming

def test_streams_deltas_in_order(server):
    out = list(engine_for(server).generate("q", "ctx", NEVER))
    assert "".join(out) == "hello world"


def test_stops_at_the_done_sentinel(server):
    Handler.body = (sse("a", "b") +
                    b'data: {"choices":[{"delta":{"content":"AFTER-DONE"}}]}\n\n')
    assert "".join(engine_for(server).generate("q", "", NEVER)) == "ab"


def test_survives_keepalives_and_junk(server):
    Handler.body = (b": keepalive comment\n\n"
                    b"\n"
                    b"data: {not valid json\n\n"
                    b'data: {"choices":[{}]}\n\n'          # delta absent entirely
                    b'data: {"choices":[{"delta":{}}]}\n\n'
                    + sse("survived"))
    assert "".join(engine_for(server).generate("q", "", NEVER)) == "survived"


def test_cancellation_stops_the_stream(server):
    Handler.body = sse(*[f"chunk{i} " for i in range(200)])
    seen = []

    def cancelled():
        return len(seen) >= 3

    for piece in engine_for(server).generate("q", "", cancelled):
        seen.append(piece)

    assert len(seen) == 3, "should stop promptly, not drain 200 chunks"


# -------------------------------------------------------------- request shape

def test_context_is_attached_to_the_prompt(server):
    list(engine_for(server).generate("why does this loop?", "# app.py:1-2\ncode", NEVER))
    message = Handler.captured["payload"]["messages"][-1]["content"]
    assert "why does this loop?" in message
    assert "# app.py:1-2" in message
    assert "<repository_excerpts>" in message


def test_empty_context_is_not_wrapped(server):
    list(engine_for(server).generate("just a question", "", NEVER))
    assert "<repository_excerpts>" not in Handler.captured["payload"]["messages"][-1]["content"]


def test_the_no_context_system_prompt_does_not_promise_excerpts(server):
    """Otherwise the control arm asks for excerpts instead of answering.

    Observed: with the excerpt-aware system prompt and no context, gemma4:e4b
    replied "Please provide the code excerpts so I can analyze them." Scoring
    that as the model's best unaided attempt overstates what retrieval added.
    """
    engine = engine_for(server)
    list(engine.generate("q", "", NEVER))
    system = Handler.captured["payload"]["messages"][0]["content"]
    assert "excerpt" not in system.lower()
    assert system == engine.SYSTEM_NO_CONTEXT


def test_the_with_context_system_prompt_explains_the_excerpts(server):
    engine = engine_for(server)
    list(engine.generate("q", "# a.py:1-2  [why]\ncode", NEVER))
    system = Handler.captured["payload"]["messages"][0]["content"]
    assert "excerpts" in system.lower()
    assert system == engine.SYSTEM


def test_both_system_prompts_ask_for_honesty_about_not_knowing(server):
    """The control must not be nudged toward guessing where the other is not."""
    engine = engine_for(server)
    for text in (engine.SYSTEM, engine.SYSTEM_NO_CONTEXT):
        assert "say so" in text.lower() and "guessing" in text.lower()


def test_streaming_is_requested(server):
    list(engine_for(server).generate("q", "", NEVER))
    assert Handler.captured["payload"]["stream"] is True


def test_user_agent_is_not_the_urllib_default(server):
    """Regression: Cloudflare-fronted providers reject `Python-urllib/3.x`.

    Groq answers such a request with HTTP 403 and a bare `error code: 1010`,
    which is indistinguishable from a rejected API key unless you read the body.
    Cost an hour of blaming a perfectly good key.
    """
    list(engine_for(server).generate("q", "", NEVER))
    ua = Handler.captured["ua"]
    assert ua and "urllib" not in ua.lower()
    assert ua.startswith("daedalus-harness/")


def test_api_key_comes_from_the_environment(server, monkeypatch):
    monkeypatch.setenv("TEST_KEY_ENV", "sk-secret")
    engine = OpenAICompatEngine(provider="local", base_url=server,
                                key_env="TEST_KEY_ENV", model="m")
    list(engine.generate("q", "", NEVER))
    assert Handler.captured["auth"] == "Bearer sk-secret"


def test_missing_key_names_the_variable_and_does_not_leak_one(server, monkeypatch):
    monkeypatch.delenv("TEST_KEY_ENV", raising=False)
    engine = OpenAICompatEngine(provider="local", base_url=server,
                                key_env="TEST_KEY_ENV", model="m")
    with pytest.raises(RuntimeError) as exc:
        list(engine.generate("q", "", NEVER))
    assert "TEST_KEY_ENV" in str(exc.value)


# ------------------------------------------------------------------- failures

@pytest.mark.parametrize("status,expected", [
    (401, "rejected"),
    (404, "model ids change"),
    (429, "rate limit"),
])
def test_http_failures_are_reported_not_swallowed(server, status, expected):
    Handler.status = status
    Handler.body = b'{"error": "nope"}'
    # max_retries=0: this asserts the message, not the backoff schedule, and
    # retrying here would spend 15s of the suite waiting to say the same thing.
    out = "".join(engine_for(server, max_retries=0).generate("q", "", NEVER))
    assert str(status) in out and expected in out


def test_429_is_retried_then_succeeds(server):
    """A token-per-minute cap clears in seconds; failing the turn is wrong."""
    state = {"calls": 0}
    original = Handler.do_POST

    def flaky(self):
        state["calls"] += 1
        if state["calls"] == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(b'{"error":"slow down"}')
            return
        original(self)

    Handler.do_POST = flaky
    try:
        out = "".join(engine_for(server, max_retries=3).generate("q", "", NEVER))
        assert out == "hello world"
        assert state["calls"] == 2, "should have retried exactly once"
    finally:
        Handler.do_POST = original


def test_daily_quota_is_not_retried(server):
    """Retrying a per-day cap wastes minutes to fail anyway.

    Observed for real: Groq's TPD limit reset in 38 minutes while the client
    backed off 4x30s per call. The body says "per day" -- believe it.
    """
    state = {"calls": 0}
    original = Handler.do_POST

    def quota(self):
        state["calls"] += 1
        self.send_response(429)
        self.end_headers()
        self.wfile.write(b'{"error":{"message":"Rate limit reached ... on tokens '
                         b'per day (TPD): Limit 100000, Used 99453"}}')

    Handler.do_POST = quota
    try:
        out = "".join(engine_for(server, max_retries=4).generate("q", "", NEVER))
        assert state["calls"] == 1, "a daily quota must not be retried"
        assert "quota" in out.lower()
        assert "ollama" in out, "should point at the offline escape hatch"
    finally:
        Handler.do_POST = original


def test_long_retry_after_is_not_waited_out(server):
    """Retry-After beyond max_backoff means fail fast, not sleep for an hour."""
    state = {"calls": 0}
    original = Handler.do_POST

    def slow(self):
        state["calls"] += 1
        self.send_response(429)
        self.send_header("Retry-After", "2400")
        self.end_headers()
        self.wfile.write(b'{"error":"come back later"}')

    Handler.do_POST = slow
    try:
        list(engine_for(server, max_retries=4, max_backoff=30).generate("q", "", NEVER))
        assert state["calls"] == 1
    finally:
        Handler.do_POST = original


def test_short_window_429_is_still_retried(server):
    """The fail-fast path must not swallow ordinary per-minute limits."""
    state = {"calls": 0}
    original = Handler.do_POST

    def flaky(self):
        state["calls"] += 1
        if state["calls"] == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(b'{"error":"tokens per minute exceeded"}')
            return
        original(self)

    Handler.do_POST = flaky
    try:
        out = "".join(engine_for(server, max_retries=3).generate("q", "", NEVER))
        assert out == "hello world" and state["calls"] == 2
    finally:
        Handler.do_POST = original


def test_429_gives_up_after_max_retries(server):
    Handler.status = 429
    Handler.body = b'{"error":"nope"}'
    engine = engine_for(server, max_retries=1, max_backoff=0.01)
    out = "".join(engine.generate("q", "", NEVER))
    assert "429" in out and "rate limit" in out


def test_unreachable_host_says_so():
    engine = OpenAICompatEngine(provider="local", base_url="http://127.0.0.1:1",
                                key_env=None, model="m", timeout=1.0)
    out = "".join(engine.generate("q", "", NEVER))
    assert "Could not reach" in out


# -------------------------------------------------------------------- thinking

def split_all(*chunks):
    s = ThinkSplitter()
    out = []
    for c in chunks:
        out += s.feed(c)
    out += s.flush()
    return out


def test_think_tags_split_from_the_reply():
    assert split_all("before<think>secret</think>after") == [
        (False, "before"), (True, "secret"), (False, "after")]


def test_think_tags_survive_being_split_across_chunks():
    """`<th` in one delta and `ink>` in the next must still be one tag."""
    assert split_all("a<", "th", "ink", ">reason</thi", "nk>b") == [
        (False, "a"), (True, "reason"), (False, "b")]


def test_no_tag_fragment_leaks_into_the_reply():
    for pieces in (["hello <", "world"], ["x</", "y"], ["a<thin", "k b"]):
        text = "".join(t for _, t in split_all(*pieces))
        assert text == "".join(pieces), pieces


def test_unterminated_think_block_is_still_emitted():
    assert split_all("<think>never closed") == [(True, "never closed")]


def test_reasoning_is_tagged_as_thought_over_the_wire(server):
    Handler.body = (b'data: {"choices":[{"delta":{"content":"<think>why</think>"}}]}\n\n'
                    b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
                    b"data: [DONE]\n\n")
    out = list(engine_for(server).generate("q", "", NEVER))
    assert [(isinstance(c, Thought), str(c)) for c in out] == [
        (True, "why"), (False, "answer")]


def test_reasoning_content_field_is_also_a_thought(server):
    Handler.body = (b'data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}\n\n'
                    b'data: {"choices":[{"delta":{"content":"done"}}]}\n\n'
                    b"data: [DONE]\n\n")
    out = list(engine_for(server).generate("q", "", NEVER))
    assert isinstance(out[0], Thought) and str(out[0]) == "hmm"
    assert not isinstance(out[1], Thought)


# ---------------------------------------------------------------- stop reasons

def test_finish_reason_is_recorded(server):
    Handler.body = (b'data: {"choices":[{"delta":{"content":"hi"},'
                    b'"finish_reason":"length"}]}\n\n'
                    b"data: [DONE]\n\n")
    engine = engine_for(server)
    list(engine.generate("q", "", NEVER))
    assert engine.stop_reason == "length"


def test_stop_reason_resets_between_calls(server):
    engine = engine_for(server)
    Handler.body = (b'data: {"choices":[{"delta":{"content":"a"},'
                    b'"finish_reason":"length"}]}\n\ndata: [DONE]\n\n')
    list(engine.generate("q", "", NEVER))
    assert engine.stop_reason == "length"

    Handler.body = sse("b")                    # no finish_reason at all
    list(engine.generate("q", "", NEVER))
    assert engine.stop_reason is None, "a stale reason would mislabel the next turn"


# ---------------------------------------------------------- gemma / no system

def test_system_role_rejection_falls_back_to_a_folded_prompt(server):
    """Gemma's template has no system role in several serving stacks.

    The request 400s rather than merging the message, so the engine folds the
    system prompt into the user turn and retries instead of failing the turn.
    """
    state = {"calls": 0}
    original = Handler.do_POST

    def picky(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        state["calls"] += 1
        if any(m["role"] == "system" for m in body["messages"]):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"System role not supported by this model"}')
            return
        Handler.captured["payload"] = body
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(Handler.body)

    Handler.do_POST = picky
    try:
        engine = engine_for(server)
        assert "".join(engine.generate("q", "", NEVER)) == "hello world"
        assert state["calls"] == 2, "one rejected, one folded"

        messages = Handler.captured["payload"]["messages"]
        assert [m["role"] for m in messages] == ["user"]
        assert engine.SYSTEM[:30] in messages[0]["content"]
        assert "q" in messages[0]["content"]

        # and it remembers, so the fallback costs one request per engine, not one
        # per turn
        state["calls"] = 0
        list(engine.generate("q2", "", NEVER))
        assert state["calls"] == 1
    finally:
        Handler.do_POST = original


def test_other_400s_are_not_mistaken_for_a_system_role_problem(server):
    Handler.status = 400
    Handler.body = b'{"error":"context length exceeded"}'
    out = "".join(engine_for(server, max_retries=0).generate("q", "", NEVER))
    assert "400" in out


# ------------------------------------------------------------- ollama homelab

@pytest.mark.parametrize("value,expected", [
    ("10.0.0.5", "http://10.0.0.5:11434/v1"),
    ("10.0.0.5:11434", "http://10.0.0.5:11434/v1"),
    ("http://10.0.0.5:11434", "http://10.0.0.5:11434/v1"),
    ("http://homelab:11434/", "http://homelab:11434/v1"),
    ("https://ollama.example.com:443/v1", "https://ollama.example.com:443/v1"),
])
def test_ollama_host_spellings_all_resolve(monkeypatch, value, expected):
    monkeypatch.setenv("OLLAMA_HOST", value)
    assert OpenAICompatEngine(provider="ollama", model="m").base_url == expected


def test_ollama_defaults_to_localhost_without_the_variable(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    engine = OpenAICompatEngine(provider="ollama", model="m")
    assert engine.base_url == "http://localhost:11434/v1"


@pytest.mark.parametrize("value,expected", [
    ("192.168.4.103", "http://192.168.4.103:3000/api"),
    ("192.168.4.103:3000", "http://192.168.4.103:3000/api"),
    ("http://192.168.4.103:3000", "http://192.168.4.103:3000/api"),
    ("http://192.168.4.103:3000/api", "http://192.168.4.103:3000/api"),
])
def test_openwebui_host_resolves_to_the_api_path(monkeypatch, value, expected):
    monkeypatch.setenv("OPENWEBUI_HOST", value)
    monkeypatch.setenv("OPENWEBUI_API_KEY", "x")
    assert OpenAICompatEngine(provider="openwebui", model="m").base_url == expected


def test_explicit_base_url_beats_the_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5")
    engine = OpenAICompatEngine(provider="ollama", base_url="http://other:1/v1",
                                model="m")
    assert engine.base_url == "http://other:1/v1"


# ------------------------------------------------------------------ providers

def test_models_are_listed_sorted(server):
    assert engine_for(server).models() == ["model-a", "model-b"]


def test_unknown_provider_lists_the_known_ones():
    with pytest.raises(ValueError) as exc:
        OpenAICompatEngine(provider="nonesuch")
    assert "groq" in str(exc.value)


def test_every_self_hosted_provider_has_a_host_variable():
    """Self-hosted means a homelab box, so each must be redirectable by env."""
    for name in LOCAL_HOSTS:
        assert name in PROVIDERS, name
        assert PROVIDERS[name].base_url.startswith("http://"), name


def test_hosted_providers_all_declare_a_key_variable():
    for name, provider in PROVIDERS.items():
        if name in LOCAL_HOSTS:
            continue                            # self-hosted: no vendor, no TLS
        assert provider.key_env and provider.key_env.endswith("_API_KEY"), name
        assert provider.base_url.startswith("https://"), name


def test_openwebui_uses_api_not_v1():
    """Its OpenAI-compatible surface is /api/chat/completions; /v1 401s."""
    assert PROVIDERS["openwebui"].base_url.endswith("/api")
    assert PROVIDERS["openwebui"].key_env == "OPENWEBUI_API_KEY"
