from pathlib import Path

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
    assert (tmp_path / "result.json").exists()
