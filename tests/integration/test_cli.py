from pathlib import Path

import pytest

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


def test_cli_reports_malformed_yaml(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text("model: [", encoding="utf-8")

    assert main(["evaluate", "--config", str(config_path)]) == 2
    assert "ML4GM error:" in capsys.readouterr().err


def test_cli_reports_config_path_is_directory(tmp_path: Path, capsys) -> None:
    assert main(["evaluate", "--config", str(tmp_path)]) == 2
    assert "ML4GM error:" in capsys.readouterr().err


def test_cli_reports_output_path_conflict(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "data",
                "prepare",
                "--config",
                "configs/quickstart.yaml",
                "--source",
                "synthetic-ml4gm",
                "--output",
                str(tmp_path),
                "--manifest",
                str(tmp_path / "manifest.json"),
            ]
        )
        == 2
    )
    assert "ML4GM error:" in capsys.readouterr().err


def test_cli_prepares_data(tmp_path: Path) -> None:
    output_path = tmp_path / "prepared.csv"
    manifest_path = tmp_path / "manifest.json"

    exit_code = main(
        [
            "data",
            "prepare",
            "--config",
            "configs/quickstart.yaml",
            "--source",
            "synthetic-ml4gm",
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert manifest_path.exists()


@pytest.mark.parametrize("command", ["train", "benchmark"])
def test_cli_runs_configured_evaluation(command: str, tmp_path: Path) -> None:
    output_dir = tmp_path / command

    exit_code = main(
        [
            command,
            "--config",
            "configs/quickstart.yaml",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "result.json").exists()
