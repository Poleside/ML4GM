from pathlib import Path

from ml4gm.cli import main


def test_cli_runs_quickstart(tmp_path: Path) -> None:
    exit_code = main(
        [
            "evaluate",
            "--config",
            "configs/quickstart.yaml",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "result.json").exists()


def test_cli_reports_missing_config(capsys) -> None:
    assert main(["evaluate", "--config", "missing.yaml"]) == 2
    assert "Configuration file not found" in capsys.readouterr().err
