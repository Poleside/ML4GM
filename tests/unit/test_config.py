from pathlib import Path

import pytest

from ml4gm.config import ConfigError, RunConfig


def test_load_quickstart_config() -> None:
    config = RunConfig.from_yaml(Path("configs/quickstart.yaml"))
    assert config.model.name == "random_forest"
    assert config.validation.strategy == "loyo"
    assert config.random_seed == 42


def test_reject_unknown_model(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "data:\n  input: data.csv\nmodel:\n  name: magic\nvalidation:\n  strategy: loyo\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Unsupported model"):
        RunConfig.from_yaml(path)
