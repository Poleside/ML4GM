import hashlib
import json
from pathlib import Path

from ml4gm import __version__
from ml4gm.config import RunConfig
from ml4gm.evaluation.runner import run_evaluation


def test_quickstart_produces_all_year_folds(tmp_path: Path) -> None:
    config = RunConfig.from_yaml(Path("configs/quickstart.yaml"))
    config = RunConfig(
        data=type(config.data)(
            input=config.data.input,
            target=config.data.target,
            glacier_id=config.data.glacier_id,
            year=config.data.year,
            output_dir=tmp_path,
        ),
        model=config.model,
        validation=config.validation,
        random_seed=config.random_seed,
        run_name=config.run_name,
    )

    result = run_evaluation(config)

    assert len(result.folds) == 6
    assert [fold.fold for fold in result.folds] == [
        "year-2000",
        "year-2001",
        "year-2002",
        "year-2003",
        "year-2004",
        "year-2005",
    ]
    record = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["error"] is None
    assert record["resolved_config"] == {
        "data": {
            "input": "data/sample/glacier_sample.csv",
            "target": "dhdt",
            "glacier_id": "rgiid",
            "year": "year",
            "output_dir": tmp_path.name,
        },
        "model": {
            "name": "random_forest",
            "parameters": {"n_estimators": 40, "max_depth": 6, "n_jobs": 1},
        },
        "validation": {"strategy": "loyo", "folds": 5},
        "random_seed": 42,
        "run_name": "quickstart-rf-loyo",
    }
    assert record["software"]["ml4gm"] == __version__
    assert record["software"]["python"]
    assert record["input_manifest"]["input_sha256"] == hashlib.sha256(
        config.data.input.read_bytes()
    ).hexdigest()
    assert record["data_summary"] == {
        "rows": 72,
        "features": ["Area", "Zmed", "t2m", "tp"],
        "feature_count": 4,
        "glaciers": 12,
        "year_coverage": {
            "minimum": 2000,
            "maximum": 2005,
            "years": [2000, 2001, 2002, 2003, 2004, 2005],
        },
    }
    assert record["model_parameters"]["n_estimators"] == 40
    assert record["model_parameters"]["random_state"] == 42
    assert record["validation_details"] == {"strategy": "loyo", "folds": 5}
    assert record["artifacts"] == {"run_record": "result.json"}
