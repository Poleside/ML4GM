from pathlib import Path

import pytest

APPLICATIONS = Path("docs/applications")
CODEX_FOR_OSS = APPLICATIONS / "codex-for-oss.md"
OPEN_SOURCE_FUND = APPLICATIONS / "codex-open-source-fund.md"


def _field(text: str, name: str) -> str:
    marker = f"## {name}\n\n"
    assert marker in text, f"missing field heading: {name}"
    return text.split(marker, 1)[1].split("\n## ", 1)[0].strip()


@pytest.mark.parametrize(
    "name",
    [
        "Why does this repository qualify?",
        "How will you use API credits for your project?",
        "Anything else we should know?",
    ],
)
def test_codex_for_oss_limited_answers_fit_500_characters(name: str) -> None:
    text = CODEX_FOR_OSS.read_text(encoding="utf-8")
    answer = _field(text, name)
    assert answer
    assert len(answer) <= 500


def test_personal_fields_are_not_inferred() -> None:
    codex = CODEX_FOR_OSS.read_text(encoding="utf-8")
    for name in ["First name", "Last name", "Email", "OpenAI Organization ID"]:
        assert _field(codex, name) == "USER INPUT REQUIRED"

    fund = OPEN_SOURCE_FUND.read_text(encoding="utf-8")
    for name in ["First name", "Last name", "Email address", "LinkedIn URL"]:
        assert _field(fund, name) == "USER INPUT REQUIRED"


def test_verified_repository_identity_and_role_are_used() -> None:
    codex = CODEX_FOR_OSS.read_text(encoding="utf-8")
    assert _field(codex, "GitHub username") == "HectorGao"
    assert _field(codex, "GitHub repository URL") == "https://github.com/Poleside/ML4GM"
    assert _field(codex, "Maintainer role") == "Core maintainer"

    fund = OPEN_SOURCE_FUND.read_text(encoding="utf-8")
    assert _field(fund, "GitHub personal") == "https://github.com/HectorGao"
    assert _field(fund, "GitHub repository") == "https://github.com/Poleside/ML4GM"


@pytest.mark.parametrize("path", [CODEX_FOR_OSS, OPEN_SOURCE_FUND])
def test_public_metrics_are_dated_submission_snapshots(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for claim in [
        "Verified 2026-07-17",
        "1 star",
        "0 forks",
        "0 open issues",
        "0 open pull requests",
        "0 releases",
        "default branch `master`",
        "viewer permission `WRITE`",
        "GitHub-detected license: none",
        "submission-preparation snapshot",
    ]:
        assert claim in text


def test_scientific_scale_claims_are_tied_to_legacy_evidence() -> None:
    text = CODEX_FOR_OSS.read_text(encoding="utf-8")
    evidence = _field(text, "Evidence checked")
    assert "8,101 glaciers" in text
    assert "162,020 glacier-year rows" in text
    assert "2000–2019" in text
    assert "legacy notebook outputs" in evidence
    assert "not validated release results" in evidence


def test_drafts_do_not_claim_adoption() -> None:
    combined = (
        CODEX_FOR_OSS.read_text(encoding="utf-8")
        + OPEN_SOURCE_FUND.read_text(encoding="utf-8")
    ).lower()
    for unsupported_claim in [
        "widely adopted",
        "broad adoption",
        "thousands of users",
        "monthly downloads",
        "production users",
    ]:
        assert unsupported_claim not in combined
