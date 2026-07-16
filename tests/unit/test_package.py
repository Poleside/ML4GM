from importlib.metadata import version
from pathlib import Path

import ml4gm


def test_package_version_matches_metadata() -> None:
    assert ml4gm.__version__ == "0.1.0"
    assert version("ml4gm") == ml4gm.__version__


def test_license_starts_with_exact_apache_header() -> None:
    license_lines = Path("LICENSE").read_text().splitlines()

    assert license_lines[:2] == ["Apache License", "Version 2.0, January 2004"]
