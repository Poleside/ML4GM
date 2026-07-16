import json
from pathlib import Path

from ml4gm.evaluation.results import FoldResult, RunResult


def test_run_result_writes_json(tmp_path: Path) -> None:
    result = RunResult(
        run_name="test",
        model="random_forest",
        validation="loyo",
        seed=42,
        folds=[FoldResult("year-2000", 8, 2, 0.4, 0.3, 0.2)],
    )
    path = result.write(tmp_path / "result.json")
    assert json.loads(path.read_text())["folds"][0]["fold"] == "year-2000"
