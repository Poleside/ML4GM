from pathlib import Path

FORBIDDEN = ("C:\\ML4GM", "/Users/", "/openbayes/")
SUPPORTED_ROOTS = [Path("src"), Path("configs"), Path("notebooks/tutorials"), Path("README.md")]
TEXT_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml", ".ipynb"}


def test_supported_files_do_not_contain_machine_paths() -> None:
    offenders: list[str] = []
    for root in SUPPORTED_ROOTS:
        paths = (
            [root]
            if root.is_file()
            else [
                path for path in root.rglob("*") if path.is_file() and path.suffix in TEXT_SUFFIXES
            ]
        )
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(value in text for value in FORBIDDEN):
                offenders.append(str(path))
    assert offenders == []
