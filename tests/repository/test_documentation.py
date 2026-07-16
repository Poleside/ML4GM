from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_required_open_source_documents_exist() -> None:
    required = [
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "MAINTAINERS.md",
        "CITATION.cff",
        "DATA_SOURCES.md",
        "ROADMAP.md",
        "CHANGELOG.md",
        "docs/data-schema.md",
        "docs/full-data-setup.md",
        "docs/benchmark-protocol.md",
        "docs/scientific-limitations.md",
    ]
    assert [path for path in required if not (ROOT / path).is_file()] == []


def test_readme_has_required_sections_and_runnable_quickstart() -> None:
    text = _read("README.md")
    sections = [
        "Why ML4GM exists",
        "Scientific scope",
        "Project status",
        "Quickstart",
        "Full scientific data",
        "Models",
        "Validation protocols",
        "Results and reproducibility",
        "Data licensing",
        "Contributing",
        "Citation",
        "Security",
        "Scientific limitations",
    ]
    assert re.findall(r"^## (.+)$", text, flags=re.MULTILINE) == sections
    assert 'python -m pip install -e ".[dev]"' in text
    assert "ml4gm evaluate --config configs/quickstart.yaml" in text
    assert "synthetic" in text.lower()
    assert "cannot support scientific conclusions" in text


def test_data_sources_record_citations_and_redistribution_status() -> None:
    text = _read("DATA_SOURCES.md")
    assert (
        "Source | Version | Purpose | Official URL/DOI | Access | License/terms | "
        "Redistribution | Required citation"
    ) in text
    for doi in (
        "10.7265/4m1f-gd79",
        "10.24381/cds.e2161bac",
        "10.1038/s41586-021-03436-z",
        "10.6096/13",
    ):
        assert doi in text
    assert "verify" in text.lower()
    assert "CC-BY" in text


def test_governance_facts_and_policies_are_explicit() -> None:
    maintainers = _read("MAINTAINERS.md")
    assert "Poleside" in maintainers and "repository owner" in maintainers
    assert "HectorGao" in maintainers and "write access" in maintainers
    assert "Private vulnerability reporting" in _read("SECURITY.md")
    conduct = _read("CODE_OF_CONDUCT.md")
    assert "Contributor Covenant" in conduct
    assert "version 2.1" in conduct
    assert "https://www.contributor-covenant.org/version/2/1/code_of_conduct.html" in conduct


def test_citation_file_is_valid_and_matches_the_release() -> None:
    citation = yaml.safe_load(_read("CITATION.cff"))
    assert citation["cff-version"] == "1.2.0"
    assert citation["version"] == "0.1.0"
    assert citation["license"] == "Apache-2.0"
    assert citation["repository-code"] == "https://github.com/Poleside/ML4GM"
    assert [author["name"] for author in citation["authors"]] == ["Poleside", "HectorGao"]


def test_documentation_uses_only_portable_runtime_placeholders() -> None:
    paths = [
        "README.md",
        "DATA_SOURCES.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "MAINTAINERS.md",
        "docs/data-schema.md",
        "docs/full-data-setup.md",
        "docs/benchmark-protocol.md",
        "docs/scientific-limitations.md",
    ]
    forbidden = re.compile(r"TBD|TODO|C:\\ML4GM|/Users/|/openbayes/")
    assert {
        path: forbidden.findall(_read(path)) for path in paths if forbidden.search(_read(path))
    } == {}


def test_internal_markdown_links_resolve() -> None:
    markdown_files = [
        ROOT / "README.md",
        ROOT / "DATA_SOURCES.md",
        ROOT / "CONTRIBUTING.md",
        *sorted((ROOT / "docs").glob("*.md")),
    ]
    broken: list[str] = []
    for document in markdown_files:
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "#")):
                continue
            path = target.split("#", maxsplit=1)[0]
            if path and not (document.parent / path).resolve().exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert broken == []


def test_readme_quickstart_executes(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/ml4gm"),
            "evaluate",
            "--config",
            str(ROOT / "configs/quickstart.yaml"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "R2=" in result.stdout
