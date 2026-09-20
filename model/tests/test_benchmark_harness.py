import pathlib
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from benchmark_harness import analyze_trace, percentile, run_once  # noqa: E402


def test_percentile_is_stable_for_small_samples():
    assert percentile([40, 10, 20, 30], 0.0) == 10
    assert percentile([40, 10, 20, 30], 0.5) == 30
    assert percentile([40, 10, 20, 30], 0.95) == 40


def test_run_once_records_a_successful_local_process():
    result = run_once([sys.executable, "-c", "print('ok')"], timeout=5)
    assert result["wall_ms"] > 0
    assert result["stdout_bytes"] == 3


def test_trace_analysis_measures_size_and_context_growth(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"event":"exchange_delta","response":{"usage":{"input_tokens":10}}}\n'
        '{"event":"exchange_delta","response":{"usage":{"input_tokens":24}}}\n',
        encoding="utf-8",
    )
    result = analyze_trace(trace)
    assert result["exchanges"] == 2
    assert result["bytes"] == trace.stat().st_size
    assert result["max_context_tokens"] == 24
    assert result["context_growth_tokens"] == 14
