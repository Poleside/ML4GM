from importlib.metadata import version

import ml4gm


def test_package_version_matches_metadata() -> None:
    assert ml4gm.__version__ == "0.1.0"
    assert version("ml4gm") == ml4gm.__version__
