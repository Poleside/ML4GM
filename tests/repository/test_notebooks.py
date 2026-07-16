import json
from pathlib import Path


def test_tutorial_notebooks_are_valid_json() -> None:
    notebooks = sorted(Path("notebooks/tutorials").glob("*.ipynb"))
    assert notebooks
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["cells"]


def test_quickstart_uses_the_supported_package_api() -> None:
    payload = json.loads(
        Path("notebooks/tutorials/01_quickstart.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "code"
    )
    assert "from ml4gm.config import RunConfig" in source
    assert "from ml4gm.evaluation.runner import run_evaluation" in source
    assert 'RunConfig.from_yaml(Path("../../configs/quickstart.yaml"))' in source
    assert "result = run_evaluation(config)" in source
