import pathlib
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_recorded_ablations import ARMS, commands  # noqa: E402


def test_every_arm_reuses_the_same_recording(tmp_path):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    recording = recordings / "case-a.jsonl"
    recording.write_text("{}\n", encoding="utf-8")
    result = commands(
        tmp_path / "daedalus", tmp_path / "cases.json", recordings,
        tmp_path / "out", ["case-a"],
    )
    assert len(result) == len(ARMS) == 4
    assert {command[command.index("--experiment-arm") + 1] for command in result} == set(ARMS)
    assert {command[command.index("--replay-responses") + 1] for command in result} == {
        str(recording)
    }
