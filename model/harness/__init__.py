"""The Daedalus harness -- everything around the engine slot.

The `daedalus` package is the architecture: torch-only, no other dependencies,
meant to be trained from. This package is the system that *uses* a model, and it
is deliberately engine-agnostic. Whatever fills the engine slot -- a QLoRA'd
Qwen2.5-Coder today, a scaled-up Daedalus core later -- the harness does not
change.

    Argus      repo-wide index and retrieval (which parts to pull into context)
    Scribe     exact single-file symbol table (lives in daedalus.memory)

and the plumbing that puts them in an editor:

    acp        the Agent Client Protocol server -- run `python -m harness`
    jsonrpc    bidirectional JSON-RPC 2.0 over newline-delimited stdio
    engine     the swappable engine slot

The execution side, which only matters once the harness can *write*:

    workspace  the path jail and write-staging boundary
    tools      what an executor may do, and how a text-only engine asks for it
    ariadne    halting policy: a hard ceiling, and pressure before it
    talos      the executor loop -- completion decided by the verifier, not the engine
    oracle     tiered verification: fail fast, and model judgement last of all

Planned, not yet built:

    Metis      read-only planner
    Lethe      bounded context with summarise-and-reset
"""
from .argus import Argus, Symbol, FileRecord, Retrieved, ScanReport
from .engine import (Engine, RetrievalOnlyEngine, StaticEngine,
                     OpenAICompatEngine, Provider, PROVIDERS, Thought,
                     ThinkSplitter)
from .acp import DaedalusAgent, Session, PROTOCOL_VERSION
from .workspace import Workspace, PathEscape
from .tools import (Tool, ToolSpec, ToolResult, ToolCall, ToolRegistry,
                    parse_calls, tokenize)
from .ariadne import Ariadne, Halt, StepOutcome
from .talos import Talos, Verdict, Verifier, Outcome, Event
from .oracle import Oracle, OracleVerdict, Tier, TierResult

__all__ = [
    "Argus", "Symbol", "FileRecord", "Retrieved", "ScanReport",
    "Engine", "RetrievalOnlyEngine", "StaticEngine", "OpenAICompatEngine",
    "Provider", "PROVIDERS", "Thought", "ThinkSplitter",
    "DaedalusAgent", "Session", "PROTOCOL_VERSION",
    "Workspace", "PathEscape",
    "Tool", "ToolSpec", "ToolResult", "ToolCall", "ToolRegistry",
    "parse_calls", "tokenize",
    "Ariadne", "Halt", "StepOutcome",
    "Talos", "Verdict", "Verifier", "Outcome", "Event",
    "Oracle", "OracleVerdict", "Tier", "TierResult",
]
