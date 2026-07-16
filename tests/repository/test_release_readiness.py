from pathlib import Path

ROOT = Path(__file__).parents[2]
CHECKLIST = ROOT / "docs" / "release-checklist.md"
AUDIT = ROOT / "docs" / "completion-audit.md"
CHANGELOG = ROOT / "CHANGELOG.md"


def test_release_checklist_covers_every_required_gate() -> None:
    text = CHECKLIST.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    required_gates = [
        "clean Python 3.11 environment",
        "editable install",
        "Ruff format and lint",
        "complete pytest suite",
        "wheel and sdist build",
        "quickstart CLI",
        "package import",
        "forbidden-path scan",
        "no restricted data in Git",
        "license and governance files",
        "application character limits",
        "GitHub CI status",
        "public repository visibility",
        "maintainer permission evidence",
    ]
    for gate in required_gates:
        assert gate in text

    assert "43 files formatted" in normalized
    assert "167 tests passed" in normalized
    assert "Original Task 16 inventory: expected exit 0" in normalized
    assert "Supported assertion: expected exit 1 and no output" in normalized
    assert "':!notebooks/legacy/**' ':!docs/superpowers/**'" in text
    assert (
        "-- src configs README.md DATA_SOURCES.md CONTRIBUTING.md SECURITY.md MAINTAINERS.md docs"
    ) in text
    assert "':!docs/release-checklist.md' ':!docs/completion-audit.md'" in text


def test_completion_audit_maps_all_design_success_criteria() -> None:
    text = AUDIT.read_text(encoding="utf-8")
    normalized = " ".join(text.split()).lower()
    for criterion in range(1, 11):
        assert f"| {criterion} |" in text

    assert "public branch and ci are not yet verified" in normalized
    assert "USER INPUT REQUIRED" in text
    assert "Do not create `v0.1.0`" in text
    assert "revision `514d7f2`" in text
    assert "Ruff: 43 files formatted" in text
    assert "Tests: 167 passed, one LightGBM runtime test skipped" in text
    assert "Original forbidden-path inventory: exit status 0" in text
    assert "Supported-scope forbidden-path assertion: exit status 1 with no output" in text
    assert "preparation-branch publication blocker" in text


def test_changelog_keeps_release_unreleased_while_external_gates_are_open() -> None:
    text = CHANGELOG.read_text(encoding="utf-8")
    assert "## [Unreleased]" in text
    assert "## [0.1.0] -" not in text
    assert "Release gating" in text
