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
import io
import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from knossos import OpenAICompatEngine, PROVIDERS, Thought, ThinkSplitter
from knossos.engine import LOCAL_HOSTS, Usage
from knossos.tools import ToolRegistry, parse_calls

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


# --------------------------------------------------------------------- usage

def usage_sse(*chunks: str, usage: dict, trailing: bool = True) -> bytes:
    """SSE with a usage report, either as a final chunk or repeated throughout.

    Both shapes are real: OpenAI appends one chunk with empty `choices` and the
    totals, while some compatible servers repeat a running total on every chunk.
    """
    out = []
    for text in chunks:
        payload: dict = {"choices": [{"delta": {"content": text}}]}
        if not trailing:
            payload["usage"] = usage
        out.append(f"data: {json.dumps(payload)}\n\n")
    if trailing:
        out.append(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode("utf-8")


def test_usage_is_requested(server):
    """A streamed response omits usage entirely unless it is asked for."""
    list(engine_for(server).generate("q", "", NEVER))

    assert Handler.captured["payload"]["stream_options"] == {"include_usage": True}


def test_usage_is_read_from_the_final_chunk(server):
    Handler.body = usage_sse("hi", usage={"prompt_tokens": 120,
                                          "completion_tokens": 30})
    engine = engine_for(server)

    assert "".join(engine.generate("q", "", NEVER)) == "hi"
    assert engine.usage.prompt == 120
    assert engine.usage.completion == 30
    assert engine.usage.total == 150
    assert engine.usage.known


def test_a_repeated_running_total_is_not_added_up(server):
    """Some servers repeat cumulative usage on every chunk.

    Accumulating would multiply a turn's cost by its chunk count -- a wrong
    number stated confidently, which is worse than no number.
    """
    Handler.body = usage_sse("a", "b", "c", trailing=False,
                             usage={"prompt_tokens": 100, "completion_tokens": 3})
    engine = engine_for(server)

    list(engine.generate("q", "", NEVER))

    assert engine.usage.prompt == 100
    assert engine.usage.requests == 1


def test_cached_prompt_tokens_are_reported_separately(server):
    """A cache hit is the entire return on a stable prefix, so it needs its own
    number -- folded into `prompt` it is invisible."""
    Handler.body = usage_sse("hi", usage={
        "prompt_tokens": 900, "completion_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 850}})
    engine = engine_for(server)

    list(engine.generate("q", "", NEVER))

    assert engine.usage.cached == 850
    # A subset of prompt, not an addition to it.
    assert engine.usage.prompt == 900
    assert engine.usage.total == 910


def test_deepseeks_flat_cache_field_is_read_too(server):
    Handler.body = usage_sse("hi", usage={"prompt_tokens": 500,
                                          "completion_tokens": 5,
                                          "prompt_cache_hit_tokens": 400})
    engine = engine_for(server)

    list(engine.generate("q", "", NEVER))

    assert engine.usage.cached == 400


def test_usage_accumulates_across_requests(server):
    Handler.body = usage_sse("hi", usage={"prompt_tokens": 10,
                                          "completion_tokens": 2})
    engine = engine_for(server)

    list(engine.generate("one", "", NEVER))
    list(engine.generate("two", "", NEVER))

    assert engine.total_usage.total == 24
    assert engine.total_usage.requests == 2
    # The per-call figure is still just the last call.
    assert engine.usage.total == 12


def test_a_provider_that_says_nothing_reports_unknown_not_zero(server):
    """Absent usage means unmeasured. Printing 0 would be a confident wrong
    number for a turn that plainly cost something."""
    engine = engine_for(server)                    # default body, no usage block

    list(engine.generate("q", "", NEVER))

    assert not engine.usage.known
    assert "not reported" in engine.usage.report()


def test_a_provider_rejecting_stream_options_still_works(server):
    """One wasted request to find out, then never again -- and never a failed run.

    An unmeasured turn is a smaller loss than a turn that did not happen, so
    this must degrade to "no numbers", not to an error.
    """
    calls = {"n": 0}
    original = Handler.do_POST

    def flaky(self):
        calls["n"] += 1
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        Handler.captured["payload"] = payload
        if "stream_options" in payload:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"unknown field: stream_options"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(sse("recovered"))

    Handler.do_POST = flaky
    try:
        engine = engine_for(server)
        assert "".join(engine.generate("q", "", NEVER)) == "recovered"
        assert calls["n"] == 2                     # rejected, then retried

        # Sticky: the second call must not pay for the discovery again.
        assert "".join(engine.generate("q", "", NEVER)) == "recovered"
        assert calls["n"] == 3
        assert "stream_options" not in Handler.captured["payload"]
    finally:
        Handler.do_POST = original


def test_an_unexplained_400_retries_once_without_stream_options(server):
    """The marker list is a guess at how a provider phrases a refusal.

    A guess that misses turns a working setup into a failing one, over a
    parameter the harness added for its own benefit. Any unexplained 400
    therefore gets exactly one retry without it.
    """
    calls = {"n": 0}
    original = Handler.do_POST

    def picky(self):
        calls["n"] += 1
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        if "stream_options" in payload:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"something we did not anticipate"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(sse("survived"))

    Handler.do_POST = picky
    try:
        engine = engine_for(server)
        assert "".join(engine.generate("q", "", NEVER)) == "survived"
        assert calls["n"] == 2
    finally:
        Handler.do_POST = original


def test_a_real_bad_request_still_reports_its_own_error(server):
    """The retry must not swallow the error the caller actually needs."""
    original = Handler.do_POST

    def broken(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(400)
        self.end_headers()
        self.wfile.write(b'{"error":"model `nope` does not exist"}')

    Handler.do_POST = broken
    try:
        out = "".join(engine_for(server).generate("q", "", NEVER))
        assert "nope" in out or "400" in out
    finally:
        Handler.do_POST = original


def test_usage_arithmetic():
    a = Usage(prompt=10, completion=5, cached=4, requests=1)
    b = Usage(prompt=20, completion=1, cached=0, requests=1)

    assert (a + b) == Usage(prompt=30, completion=6, cached=4, requests=2)
    assert (a + b).total == 36
    assert Usage.from_payload("not a dict") == Usage()
    assert Usage.from_payload({"prompt_tokens": None}) == Usage(requests=1)


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
    assert ua.startswith("knossos/")


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


# ------------------------------------------------------- native tool calls

def test_a_native_tool_call_becomes_a_fenced_block(server):
    """A model may answer in OpenAI's tool-call format rather than in prose.

    Knossos reads tool calls out of the message content, so such a reply used to
    arrive empty and the turn looked like a silent no-op -- observed with
    gemma4:e4b, where the whole reply landed in `tool_calls`. Normalising to the
    fenced block means one convention reaches Talos either way.
    """
    Handler.body = (b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                    b'"function":{"name":"write_file","arguments":'
                    b'"{\\"path\\": \\"a.py\\"}"}}]},'
                    b'"finish_reason":"tool_calls"}]}\n\n'
                    b"data: [DONE]\n\n")
    engine = engine_for(server)
    reply = "".join(str(c) for c in engine.generate("q", "", NEVER))

    prose, calls = parse_calls(reply)
    assert [c.name for c in calls] == ["write_file"]
    assert calls[0].args == {"path": "a.py"}
    assert engine.native_tool_calls == 1


def test_streamed_argument_fragments_are_reassembled(server):
    """Arguments arrive split at arbitrary points, keyed only by index."""
    Handler.body = (
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        b'"function":{"name":"write_file","arguments":"{\\"pa"}}]}}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        b'"function":{"arguments":"th\\": \\"b.py\\"}"}}]}}]}\n\n'
        b"data: [DONE]\n\n")
    reply = "".join(str(c) for c in engine_for(server).generate("q", "", NEVER))

    _prose, calls = parse_calls(reply)
    assert calls[0].args == {"path": "b.py"}


def test_two_native_calls_keep_their_order(server):
    Handler.body = (
        b'data: {"choices":[{"delta":{"tool_calls":['
        b'{"index":0,"function":{"name":"read_file","arguments":"{}"}},'
        b'{"index":1,"function":{"name":"list_dir","arguments":"{}"}}]}}]}\n\n'
        b"data: [DONE]\n\n")
    reply = "".join(str(c) for c in engine_for(server).generate("q", "", NEVER))

    _prose, calls = parse_calls(reply)
    assert [c.name for c in calls] == ["read_file", "list_dir"]


def test_unparseable_native_arguments_are_dropped_not_guessed(server):
    """Fails closed, like the fenced-block parser: half an argument is worse
    than none, because it runs the wrong action rather than costing a turn."""
    Handler.body = (b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                    b'"function":{"name":"write_file","arguments":"{\\"path\\":"}}]}}]}\n\n'
                    b"data: [DONE]\n\n")
    engine = engine_for(server)
    reply = "".join(str(c) for c in engine.generate("q", "", NEVER))

    _prose, calls = parse_calls(reply)
    assert calls == []
    assert engine.native_tool_calls == 0


def test_native_tool_call_count_resets_between_calls(server):
    engine = engine_for(server)
    Handler.body = (b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                    b'"function":{"name":"read_file","arguments":"{}"}}]}}]}\n\n'
                    b"data: [DONE]\n\n")
    list(engine.generate("q", "", NEVER))
    assert engine.native_tool_calls == 1

    Handler.body = sse("plain text")
    list(engine.generate("q", "", NEVER))
    assert engine.native_tool_calls == 0, "a stale count would misdiagnose the next turn"


def test_the_eval_path_sends_no_tool_schema(server):
    """The measured retrieval deltas were taken without a `tools` payload.

    `eval.py` builds an engine directly and never constructs a Talos, so nothing
    sets `tool_schema` and the request shape is what it always was. Pinned here
    because if that ever changes silently, every number in the README becomes a
    measurement of a request nobody made.
    """
    from knossos.eval import _collect

    Handler.body = sse("an answer")
    engine = engine_for(server)

    _collect(engine, "what does the router do?", "some retrieved context")

    assert engine.tool_schema is None
    assert "tools" not in Handler.captured["payload"]


def test_the_tool_schema_is_sent_only_when_set(server):
    engine = engine_for(server)
    Handler.body = sse("hi")

    list(engine.generate("q", "", NEVER))
    assert "tools" not in Handler.captured["payload"]

    engine.tool_schema = ToolRegistry.default().openai_schema()
    list(engine.generate("q", "", NEVER))
    sent = Handler.captured["payload"]["tools"]
    assert {t["function"]["name"] for t in sent} >= {"write_file", "read_file"}
    assert sent[0]["function"]["parameters"]["type"] == "object"


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


# ------------------------------------------------ giving up on a dead provider

def test_repeated_exhaustion_stops_the_engine_asking(server):
    """Patience needs a way to notice futility.

    Measured against an exhausted free tier: every retry was individually
    correct, and the aggregate was 480s per case and 82.7 minutes to produce
    ten zeros. Waiting out a window that is never going to clear is not
    diligence.
    """
    engine = engine_for(server, give_up_after=2)
    assert engine.exhausted is False

    engine._consecutive_exhaustions = 2

    assert engine.exhausted is True


def test_an_exhausted_engine_fails_immediately(server):
    engine = engine_for(server, give_up_after=1)
    engine._consecutive_exhaustions = 1

    started = time.perf_counter()
    with pytest.raises(urllib.error.HTTPError) as caught:
        engine._post_with_retry("/chat/completions", {}, NEVER)
    elapsed = time.perf_counter() - started

    assert caught.value.code == 429
    assert elapsed < 1.0, "it must not wait before refusing"
    assert "not clearing" in caught.value.read().decode()


def test_a_successful_response_forgives_earlier_exhaustion(server):
    """A provider that recovers must be used again, not written off."""
    engine = engine_for(server, give_up_after=3)
    engine._consecutive_exhaustions = 2

    list(engine.generate("q", "", NEVER))

    assert engine._consecutive_exhaustions == 0
    assert engine.exhausted is False


# --------------------------------------------------- classifying a 429 correctly

class FakeHTTPError(urllib.error.HTTPError):
    def __init__(self, body, headers=None):
        super().__init__("http://x", 429, "Too Many Requests", headers or {},
                         io.BytesIO(body.encode("utf-8")))


#: Gemini's real per-minute refusal, trimmed. Note it says "Quota exceeded" --
#: the same words it would use for a daily cap -- and states the delay only in
#: the message body, with no `Retry-After` header at all.
GEMINI_429 = (
    "You exceeded your current quota, please check your plan and billing details. "
    "* Quota exceeded for metric: generate_content_free_tier_requests, limit: 30, "
    "model: gemini-3.6-flash. Please retry in 27.031023733s.")

GEMINI_NOT_ON_TIER = (
    "You exceeded your current quota. * Quota exceeded for metric: "
    "generate_content_free_tier_requests, limit: 0, model: gemini-3.1-pro. "
    "Please retry in 27.03s.")


def test_a_short_window_stated_in_the_body_is_retried(server):
    """Gemini states the delay in the body and sends no `Retry-After`.

    Reading only the header saw a 429 with no advice, fell through to the
    substring test, matched "quota exceeded" -- which Gemini uses for
    *per-minute* limits -- and gave up. That made the provider with the largest
    free daily allowance unusable.
    """
    engine = engine_for(server)

    assert engine._is_long_quota(FakeHTTPError(GEMINI_429)) is False
    assert engine._retry_delay(FakeHTTPError(GEMINI_429)) == pytest.approx(27.03, abs=0.1)


def test_a_zero_limit_is_terminal(server):
    """`limit: 0` means the model is not served on this tier. Waiting cannot fix it."""
    engine = engine_for(server)

    assert engine._is_long_quota(FakeHTTPError(GEMINI_NOT_ON_TIER)) is True


def test_a_zero_limit_says_so_rather_than_blaming_quota(server):
    engine = engine_for(server)

    text = engine._explain(FakeHTTPError(GEMINI_NOT_ON_TIER))

    assert "not served on this account's tier" in text
    assert "switch --model" in text


def test_an_advertised_delay_outranks_the_wording(server):
    """A stated 27s is a short window whatever words surround it."""
    engine = engine_for(server)
    body = "quota exceeded, daily limit reached. Please retry in 5s."

    assert engine._is_long_quota(FakeHTTPError(body)) is False


def test_a_long_advertised_delay_is_still_given_up_on(server):
    engine = engine_for(server, max_backoff=30.0, max_advised_wait=120.0)
    body = "rate limited. Please retry in 3600s."

    assert engine._is_long_quota(FakeHTTPError(body)) is True


def test_an_explicit_wait_gets_more_patience_than_a_guess(server):
    """`max_backoff` caps a guess; it must not overrule an instruction.

    Gemini's free tier says "retry in 59s" on a 20-requests-per-minute limit.
    Capping both at the same 30s meant the harness gave up on a window that was
    about to clear -- which is why the provider with the largest free daily
    allowance measured as 0/10.
    """
    engine = engine_for(server, max_backoff=30.0, max_advised_wait=120.0)
    body = "rate limited. Please retry in 59.18s."

    assert engine._is_long_quota(FakeHTTPError(body)) is False


def test_the_advised_ceiling_is_never_below_the_guess_ceiling(server):
    """A nonsensical configuration must not make advice less trusted than a guess."""
    engine = engine_for(server, max_backoff=60.0, max_advised_wait=5.0)

    assert engine.max_advised_wait == 60.0


def test_the_header_still_wins_when_present(server):
    engine = engine_for(server)
    exc = FakeHTTPError("please retry in 900s.", headers={"Retry-After": "3"})

    assert engine._retry_delay(exc) == 3.0
    assert engine._is_long_quota(exc) is False


def test_a_daily_cap_with_no_delay_is_still_terminal(server):
    """The substring fallback has to keep working where there is no advice."""
    engine = engine_for(server)

    assert engine._is_long_quota(FakeHTTPError("insufficient_quota: per day")) is True


# ------------------------------------------------------- provider token limits

GROQ_413 = ("request too large for model `x` in organization `y` service tier "
            "`on_demand` on tokens per minute (tpm): limit 8000, requested 8263, "
            "please reduce your message size and try again")


def test_a_token_limit_shrinks_the_reply_allowance(server):
    """Found by running the eval against a free tier and getting 0/10.

    Raising the default `max_tokens` to 8192 put every request over an 8 000
    token allowance on its own, before a single prompt token -- so all ten
    cases died in two steps. The provider states the arithmetic; the fix is to
    read it rather than to report it.
    """
    engine = engine_for(server, max_tokens=8192)

    assert engine._shrink_to_token_limit(413, GROQ_413) is True
    # 8263 requested - 8192 asked for = 71 prompt tokens, under an 8000 ceiling.
    assert engine.max_tokens == 8000 - 71 - engine._LIMIT_MARGIN


def test_a_token_limit_also_bounds_the_transcript(server):
    """The same ceiling applies to input, and Talos sizes Lethe from this."""
    engine = engine_for(server, max_tokens=8192)

    engine._shrink_to_token_limit(413, GROQ_413)

    assert engine.context_window == 8000


def test_a_smaller_discovered_window_is_not_widened(server):
    engine = engine_for(server, max_tokens=8192)
    engine.context_window = 4096

    engine._shrink_to_token_limit(413, GROQ_413)

    assert engine.context_window == 4096, "a stated limit must not raise a known one"


def test_a_prompt_that_cannot_fit_is_not_retried(server):
    """Shrinking the reply cannot rescue a prompt that is over on its own."""
    engine = engine_for(server, max_tokens=1024)
    body = "tokens per minute (tpm): limit 8000, requested 20000"

    assert engine._shrink_to_token_limit(413, body) is False
    assert engine.max_tokens == 1024, "no point shaving the reply"
    assert engine.context_window == 8000, "but the transcript budget still learns"


def test_an_unparseable_limit_is_left_alone(server):
    engine = engine_for(server, max_tokens=8192)

    assert engine._shrink_to_token_limit(413, "request too large") is False
    assert engine.max_tokens == 8192


def test_unrelated_errors_do_not_shrink_anything(server):
    """A 429 is the rate path and must stay there.

    `_post_with_retry` already backs off on 429, and `_is_long_quota` refuses to
    retry a *daily* quota at all. Handling 429 here as well turned that refusal
    back into a retry loop, which `test_daily_quota_is_not_retried` caught.
    """
    engine = engine_for(server, max_tokens=8192)

    assert engine._shrink_to_token_limit(500, "limit 8000 internal error") is False
    assert engine._shrink_to_token_limit(429, "tokens: limit 8000, requested 9000") is False
    assert engine.max_tokens == 8192


def test_the_allowance_is_never_shrunk_below_a_usable_reply(server):
    """A reply too short to finish a tool call is a worse failure than a 413."""
    engine = engine_for(server, max_tokens=4096)
    body = "tokens per minute (tpm): limit 1000, requested 4500"

    engine._shrink_to_token_limit(413, body)

    assert engine.max_tokens >= engine.MIN_OUTPUT_TOKENS


# --------------------------------------------------------- structured messages

def test_generate_messages_sends_the_array_unchanged(server):
    engine = engine_for(server)
    history = [
        {"role": "system", "content": "you are talos"},
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": "I did a thing"},
        {"role": "user", "content": "verify it"},
    ]

    out = "".join(engine.generate_messages(history, NEVER))

    assert out == "hello world"
    assert Handler.captured["payload"]["messages"] == history


def test_generate_messages_still_streams_and_carries_tools(server):
    engine = engine_for(server)
    engine.tool_schema = ToolRegistry.default().openai_schema()

    list(engine.generate_messages([{"role": "user", "content": "hi"}], NEVER))

    payload = Handler.captured["payload"]
    assert payload["stream"] is True
    assert payload["tools"], "the schema must survive the structured path"


def test_a_system_rejecting_provider_gets_the_array_folded(server):
    """Folding, not dropping: the system message carries the tool protocol."""
    from knossos.engine import _fold_system

    folded = _fold_system([
        {"role": "system", "content": "RULES"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "reply"},
    ])

    assert [m["role"] for m in folded] == ["user", "assistant"]
    assert folded[0]["content"] == "RULES\n\ntask"


def test_folding_a_system_only_array_keeps_the_rules(server):
    from knossos.engine import _fold_system

    folded = _fold_system([{"role": "system", "content": "RULES"}])

    assert folded == [{"role": "user", "content": "RULES"}]


def test_the_output_cap_leaves_room_for_reasoning(server):
    """A reasoning model is charged for `<think>` against this same budget."""
    list(engine_for(server).generate("q", "", NEVER))

    assert Handler.captured["payload"]["max_tokens"] >= 8192


# ------------------------------------------------------- context window discovery

class FakePs:
    """Stands in for `urlopen`, returning one canned `/api/ps` body."""

    def __init__(self, body=None, boom=False):
        self.body = body
        self.boom = boom
        self.calls = 0

    def __call__(self, url, timeout=None):
        self.calls += 1
        if self.boom:
            raise OSError("connection refused")
        assert str(url).endswith("/api/ps"), f"probed {url}, not /api/ps"

        payload = json.dumps(self.body).encode("utf-8")

        class Response:
            def read(self_inner):
                return payload

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        return Response()


def probing_engine(monkeypatch, fake):
    """An ollama-flavoured engine whose `/api/ps` probe is `fake`."""
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5")
    monkeypatch.delenv("KNOSSOS_CONTEXT_WINDOW", raising=False)
    monkeypatch.setattr("knossos.engine.urllib.request.urlopen", fake)
    return OpenAICompatEngine(provider="ollama", model="m")


def test_the_window_comes_from_ps_not_from_the_model_card(monkeypatch):
    """`/api/show` reports what the weights allow; `/api/ps` what is served.

    Measured on a real deployment: `show` said 262144 while the server was
    serving that model with a 16384 window. Sizing from the card would recreate
    the over-promise this exists to remove, with a larger number.
    """
    fake = FakePs({"models": [{"model": "m", "context_length": 16384}]})
    engine = probing_engine(monkeypatch, fake)

    assert engine.context_window == 16384


def test_a_model_that_is_not_loaded_is_unknown_and_retried(monkeypatch):
    """It cannot be resident before the first generation, and that is fine."""
    fake = FakePs({"models": []})
    engine = probing_engine(monkeypatch, fake)

    assert engine.context_window is None
    assert engine.context_window is None
    assert fake.calls == 2, "an unknown-but-knowable answer must be retried"


def test_the_answer_is_cached_once_known(monkeypatch):
    fake = FakePs({"models": [{"model": "m", "context_length": 8192}]})
    engine = probing_engine(monkeypatch, fake)

    assert engine.context_window == 8192
    assert engine.context_window == 8192
    assert fake.calls == 1


def test_an_unreachable_server_is_not_retried_every_turn(monkeypatch):
    """A hosted provider is not Ollama; paying a timeout per turn would be a bug."""
    fake = FakePs(boom=True)
    engine = probing_engine(monkeypatch, fake)

    assert engine.context_window is None
    assert engine.context_window is None
    assert fake.calls == 1


def test_the_environment_override_wins_and_skips_the_probe(monkeypatch):
    fake = FakePs({"models": [{"model": "m", "context_length": 16384}]})
    monkeypatch.setenv("KNOSSOS_CONTEXT_WINDOW", "40000")
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5")
    monkeypatch.setattr("knossos.engine.urllib.request.urlopen", fake)

    engine = OpenAICompatEngine(provider="ollama", model="m")

    assert engine.context_window == 40000
    assert fake.calls == 0


@pytest.mark.parametrize("bad", ["nonsense", "0", "-5", ""])
def test_a_malformed_override_is_ignored_not_fatal(monkeypatch, bad):
    fake = FakePs({"models": []})
    monkeypatch.setenv("KNOSSOS_CONTEXT_WINDOW", bad)
    engine = probing_engine(monkeypatch, fake)

    assert engine.context_window is None


def test_constructing_an_engine_touches_no_network(monkeypatch):
    """The constructor runs in tests, in --help, and before anyone commits to it."""
    fake = FakePs({"models": [{"model": "m", "context_length": 16384}]})
    probing_engine(monkeypatch, fake)

    assert fake.calls == 0


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


# ------------------------------------------------------- the request-size wall
#
# Several hosted providers cap the HTTP body far below the model's context
# window, and answer an oversized request with a bare 413. In a trace that is
# indistinguishable from a model that failed the task: `passed: false`, and
# `halt: "stuck"`, because an agent handed an error string in place of a reply
# makes no tool calls. Two entire eval runs in `model/traces/` were recorded as
# zero-percent capability scores when what actually happened was that the prompt
# was too big to send.
#
# The limit is not published by any provider here, so it cannot be tabulated in
# advance -- it has to be discovered once and then configured. These pin both
# halves of that: the error names the size, and a configured limit is enforced
# before the request leaves the process.


def _engine(**kwargs):
    from knossos.engine import OpenAICompatEngine
    kwargs.setdefault("provider", "groq")
    kwargs.setdefault("model", "m")
    return OpenAICompatEngine(**kwargs)


def test_an_oversized_body_is_refused_before_it_reaches_the_network(monkeypatch):
    from knossos.engine import EngineProfile, RequestTooLarge

    def explode(*a, **k):                      # noqa: ANN002, ANN003
        raise AssertionError("urlopen must not be reached")

    monkeypatch.setattr("urllib.request.urlopen", explode)
    engine = _engine()
    engine.profile = EngineProfile(max_request_bytes=512)

    with pytest.raises(RequestTooLarge) as caught:
        engine._post("/chat/completions", {"m": "x" * 5000})
    assert caught.value.limit == 512
    assert caught.value.size > 512


def test_a_body_within_the_limit_is_not_refused(monkeypatch):
    """The guard must not become a second, stricter context budget."""
    from knossos.engine import EngineProfile

    sent = {}

    def capture(req, **k):                     # noqa: ANN001, ANN003
        sent["bytes"] = len(req.data)
        return io.BytesIO(b"{}")

    monkeypatch.setattr("urllib.request.urlopen", capture)
    engine = _engine()
    engine.profile = EngineProfile(max_request_bytes=100_000)
    engine._post("/chat/completions", {"m": "x" * 100})
    assert sent["bytes"] < 100_000


def test_an_unknown_limit_is_not_treated_as_zero(monkeypatch):
    """`None` means unknown, which must not behave like "refuse everything"."""
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, **k: io.BytesIO(b"{}"))
    engine = _engine()
    assert engine.profile.max_request_bytes is None
    engine._post("/chat/completions", {"m": "x" * 50_000})


def test_a_413_says_how_big_the_request_was(monkeypatch):
    """Actionable beats accurate-but-opaque: the old text was 'see the log'."""
    from knossos.engine import RequestTooLarge

    engine = _engine()
    message = engine._explain(RequestTooLarge(4_300_000, 4_000_000))
    assert "413" in message
    assert "4.1 MB" in message
    assert "retrying will not help" in message


def test_the_limit_can_be_set_from_the_environment(monkeypatch):
    """The discovery workflow: hit it once, read the size, set the variable."""
    engine = _engine()
    assert engine.profile.max_request_bytes is None
    monkeypatch.setenv("KNOSSOS_MAX_REQUEST_BYTES", "4096")
    assert engine.profile.max_request_bytes == 4096


def test_request_too_large_travels_the_ordinary_http_error_path():
    """It subclasses HTTPError so no existing handler needs to learn a new type."""
    from knossos.engine import RequestTooLarge

    error = RequestTooLarge(10, 5)
    assert isinstance(error, urllib.error.HTTPError)
    assert error.code == 413


def test_human_bytes_does_not_report_a_five_kilobyte_body_as_zero_mb():
    from knossos.engine import _human_bytes

    assert _human_bytes(5_047) == "4.9 KB"
    assert _human_bytes(4_300_000) == "4.1 MB"
    assert _human_bytes(512) == "512 B"


# ------------------------------------------------- models that reject sampling
#
# Claude Sonnet 5 and the Opus 4.7+ line removed `temperature`, `top_p` and
# `top_k`: a non-default value is a 400, not a silently ignored field. This
# harness sends `temperature=0.2` on every request, so without a degrade path
# every turn against those models fails before the model is reached — and
# `codeval` would record that as a capability score, which is exactly what
# `CaseResult.unreachable` exists to prevent.


def _sampling_refusal():
    return urllib.error.HTTPError(
        url="", code=400, msg="Bad Request", hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(json.dumps({"error": {
            "message": "temperature: Extra inputs are not permitted",
        }}).encode()))


def test_temperature_is_sent_by_default(monkeypatch):
    """The degrade must not become a blanket removal — most models take it."""
    from knossos.engine import OpenAICompatEngine

    sent = []

    def fake_post(self, path, payload, cancelled):   # noqa: ANN001
        sent.append(dict(payload))
        return io.BytesIO(b"data: [DONE]\n\n")

    monkeypatch.setattr(OpenAICompatEngine, "_post_with_retry", fake_post)
    list(_engine().generate("hi", "", NEVER))
    assert "temperature" in sent[0]


def test_a_model_rejecting_sampling_is_retried_without_temperature(monkeypatch):
    from knossos.engine import OpenAICompatEngine

    sent = []
    calls = {"n": 0}

    def fake_post(self, path, payload, cancelled):   # noqa: ANN001
        sent.append(dict(payload))
        calls["n"] += 1
        if calls["n"] == 1:
            raise _sampling_refusal()
        return io.BytesIO(b"data: [DONE]\n\n")

    monkeypatch.setattr(OpenAICompatEngine, "_post_with_retry", fake_post)
    engine = _engine()
    list(engine.generate("hi", "", NEVER))

    assert "temperature" in sent[0], "first attempt should try it"
    assert "temperature" not in sent[1], "retry must drop it"


def test_the_refusal_is_sticky_for_the_engines_life(monkeypatch):
    """One wasted request per engine, not one per turn."""
    from knossos.engine import OpenAICompatEngine

    sent = []
    calls = {"n": 0}

    def fake_post(self, path, payload, cancelled):   # noqa: ANN001
        sent.append(dict(payload))
        calls["n"] += 1
        if calls["n"] == 1:
            raise _sampling_refusal()
        return io.BytesIO(b"data: [DONE]\n\n")

    monkeypatch.setattr(OpenAICompatEngine, "_post_with_retry", fake_post)
    engine = _engine()
    list(engine.generate("one", "", NEVER))
    list(engine.generate("two", "", NEVER))

    assert not engine._send_sampling
    assert all("temperature" not in p for p in sent[1:])


def test_a_sampling_refusal_is_not_misread_as_a_stream_options_one(monkeypatch):
    """Both are 400s about a rejected field, and the messages overlap.

    `_NO_STREAM_OPTIONS` matches "extra inputs are not permitted"-style text, so
    if that branch were checked first the retry would drop `stream_options` and
    resend `temperature` — the same 400, forever.
    """
    from knossos.engine import OpenAICompatEngine

    sent = []
    calls = {"n": 0}

    def fake_post(self, path, payload, cancelled):   # noqa: ANN001
        sent.append(dict(payload))
        calls["n"] += 1
        if calls["n"] == 1:
            raise _sampling_refusal()
        return io.BytesIO(b"data: [DONE]\n\n")

    monkeypatch.setattr(OpenAICompatEngine, "_post_with_retry", fake_post)
    engine = _engine()
    list(engine.generate("hi", "", NEVER))

    assert "temperature" not in sent[1]
    assert engine._send_stream_options, "stream_options was not the problem"


def test_the_default_budget_leaves_room_for_thinking():
    """Sonnet 5 and Opus 5 think by default; a budget sized around the answer
    alone is spent before the answer starts (`EMPTY_REPLY_NOTE`)."""
    assert _engine().max_tokens >= 32_000


def test_the_anthropic_provider_points_at_a_current_model():
    from knossos.engine import PROVIDERS
    assert PROVIDERS["anthropic"].default_model == "claude-sonnet-5"
