import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
WORKFLOW_PATH = ROOT / ".github/workflows/ci.yml"


def _workflow() -> dict[str, object]:
    document = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _commands(job: dict[str, object]) -> list[str]:
    steps = job["steps"]
    assert isinstance(steps, list)
    return [str(step["run"]) for step in steps if isinstance(step, dict) and "run" in step]


def test_github_community_files_exist() -> None:
    required = [
        ".github/workflows/ci.yml",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/pull_request_template.md",
    ]

    assert [path for path in required if not (ROOT / path).exists()] == []


def test_ci_has_minimal_permissions_and_required_triggers() -> None:
    workflow = _workflow()
    triggers = workflow.get("on", workflow.get(True))

    assert workflow["permissions"] == {"contents": "read"}
    assert isinstance(triggers, dict)
    assert triggers["push"]["branches"] == ["master", "codex/**"]
    assert "pull_request" in triggers


def test_ci_uses_stable_actions_and_checks_all_supported_sources() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    quality = jobs["quality"]
    assert isinstance(quality, dict)
    commands = _commands(quality)

    for name in ("quality", "package", "lightgbm", "pytorch", "dependency-audit"):
        job = jobs[name]
        assert isinstance(job, dict)
        steps = job["steps"]
        assert isinstance(steps, list)
        uses = [step["uses"] for step in steps if isinstance(step, dict) and "uses" in step]
        assert uses == ["actions/checkout@v4", "actions/setup-python@v5"]
    assert "ruff format --check ." in commands
    assert "ruff check ." in commands
    assert "pytest" in commands


def test_ruff_excludes_only_unsupported_legacy_artifacts() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["ruff"]["extend-exclude"] == [
        "compare",
        "notebooks/legacy",
    ]


def test_package_uses_pep639_license_metadata() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert "setuptools>=77" in pyproject["build-system"]["requires"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert "License :: OSI Approved :: Apache Software License" not in project["classifiers"]
    assert "pytest>=8.3,<10" in project["optional-dependencies"]["dev"]


def test_ci_validates_citation_and_built_distribution_quickstart() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    release = jobs["package"]
    assert isinstance(release, dict)
    commands = "\n".join(_commands(release))

    assert "cffconvert==2.0.0" in commands
    assert "cffconvert --validate" in commands
    assert "python -m build" in commands
    assert "python -m pip install dist/*.whl" in commands
    assert "ml4gm evaluate --config configs/quickstart.yaml" in commands


def test_ci_runs_real_lightgbm_round_trip_on_supported_ubuntu() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    lightgbm = jobs["lightgbm"]
    assert isinstance(lightgbm, dict)
    commands = "\n".join(_commands(lightgbm))

    assert lightgbm["runs-on"] == "ubuntu-latest"
    assert ".[dev,lightgbm]" in commands
    assert "tests/ci/lightgbm_roundtrip.py" in commands
    assert "pytest.skip" not in commands


def test_ci_runs_real_neural_tests_on_minimum_and_current_pytorch() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    torch = jobs["pytorch"]
    assert isinstance(torch, dict)
    strategy = torch["strategy"]
    assert isinstance(strategy, dict)
    matrix = strategy["matrix"]
    assert isinstance(matrix, dict)
    commands = "\n".join(_commands(torch))
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    script = (ROOT / "tests/ci/torch_roundtrip.py").read_text(encoding="utf-8")

    assert matrix["torch-spec"] == ["torch==2.4.*", "torch>=2.4,<3"]
    assert matrix["numpy-spec"] == ["numpy==1.26.*"]
    assert ".[dev,torch]" in commands
    assert "${{ matrix.numpy-spec }}" in commands
    assert "tests/ci/torch_roundtrip.py" in commands
    assert "tests/unit/test_neural_models.py" in commands
    assert "tests/integration/test_sequence_models.py" in commands
    assert "pytest.skip" not in commands
    assert pyproject["project"]["optional-dependencies"]["torch"] == [
        "torch>=2.4,<3",
        "numpy>=1.26,<2",
    ]
    assert "pytest.skip" not in script
    assert "pytest.importorskip" not in script
    assert "np.__version__" in script
    assert "torch.__version__" in script
    assert "torch.from_numpy" in script
    assert ".numpy()" in script
    assert "Failed to initialize NumPy" in script


def test_ci_blocks_dependency_vulnerabilities_after_installing_project() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    audit = jobs["dependency-audit"]
    assert isinstance(audit, dict)
    commands = _commands(audit)

    assert audit["runs-on"] == "ubuntu-latest"
    assert audit["permissions"] == {"contents": "read"}
    assert commands[-3:] == [
        'python -m pip install --upgrade pip "setuptools>=83"',
        'python -m pip install -e ".[dev]" "pip-audit>=2.9,<3"',
        "pip-audit --cache-dir /tmp/pip-audit-cache",
    ]


def test_ci_scans_full_git_history_for_secrets_with_minimal_permissions() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    secret_scan = jobs["secret-scan"]
    assert isinstance(secret_scan, dict)
    steps = secret_scan["steps"]
    assert isinstance(steps, list)

    assert secret_scan["runs-on"] == "ubuntu-latest"
    assert secret_scan["permissions"] == {"contents": "read"}
    assert steps == [
        {"uses": "actions/checkout@v4", "with": {"fetch-depth": 0}},
        {
            "uses": "gitleaks/gitleaks-action@v3",
            "env": {
                "GITHUB_TOKEN": "${{ secrets.GITHUB_TOKEN }}",
                "GITLEAKS_ENABLE_COMMENTS": "false",
            },
        },
    ]


def test_issue_and_pr_templates_cover_data_rights_and_leakage() -> None:
    combined = "\n".join(
        (ROOT / path).read_text(encoding="utf-8").lower()
        for path in (
            ".github/ISSUE_TEMPLATE/bug_report.yml",
            ".github/ISSUE_TEMPLATE/feature_request.yml",
            ".github/pull_request_template.md",
        )
    )

    assert "restricted data" in combined
    assert "data rights" in combined
    assert "leakage" in combined
