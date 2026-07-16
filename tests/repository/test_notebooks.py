import ast
import json
import os
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


def test_quickstart_executes_from_the_tutorial_directory() -> None:
    tutorial_dir = Path("notebooks/tutorials").resolve()
    payload = json.loads(
        (tutorial_dir / "01_quickstart.ipynb").read_text(encoding="utf-8")
    )
    assert [cell["cell_type"] for cell in payload["cells"]] == [
        "markdown",
        "code",
        "code",
        "code",
    ]

    namespace: dict[str, object] = {}
    previous_cwd = Path.cwd()
    try:
        os.chdir(tutorial_dir)
        code_cells = payload["cells"][1:]
        for cell in code_cells[:-1]:
            source = "".join(cell["source"])
            exec(compile(source, "01_quickstart.ipynb", "exec"), namespace)

        final_tree = ast.parse("".join(code_cells[-1]["source"]))
        assert isinstance(final_tree.body[-1], ast.Expr)
        prefix = ast.Module(body=final_tree.body[:-1], type_ignores=[])
        exec(compile(prefix, "01_quickstart.ipynb", "exec"), namespace)
        displayed = eval(
            compile(ast.Expression(final_tree.body[-1].value), "01_quickstart.ipynb", "eval"),
            namespace,
        )
    finally:
        os.chdir(previous_cwd)

    result = namespace["result"]
    assert len(result.folds) == 6
    assert namespace["fold_metrics"] is displayed
    assert len(displayed) == 6
    assert {"fold", "n_train", "n_test", "r2", "rmse", "mae"} <= set(displayed.columns)
