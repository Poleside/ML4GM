import hashlib
import json
from pathlib import Path

import pytest

from ml4gm.config import DataConfig, ModelConfig, RunConfig, ValidationConfig
from ml4gm.evaluation.runner import _input_manifest, run_evaluation


def _write_manifest(path: Path, digest: str) -> None:
    path.write_text(json.dumps({"sha256": digest}), encoding="utf-8")


def test_input_manifest_prefers_matching_stem_specific_candidate(tmp_path: Path) -> None:
    input_path = tmp_path / "prepared.csv"
    input_path.write_text("content", encoding="utf-8")
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    _write_manifest(tmp_path / "prepared.manifest.json", digest)
    _write_manifest(tmp_path / "manifest.json", digest)

    identity = _input_manifest(input_path)

    assert identity["manifest_path"] == "prepared.manifest.json"
    assert identity["manifest_input_sha256"] == digest


def test_input_manifest_skips_mismatches_and_continues_to_matching_candidate(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "prepared.csv"
    input_path.write_text("current", encoding="utf-8")
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    _write_manifest(tmp_path / "prepared.manifest.json", "0" * 64)
    _write_manifest(tmp_path / "manifest.json", digest)

    identity = _input_manifest(input_path)

    assert identity["manifest_path"] == "manifest.json"
    assert identity["manifest_input_sha256"] == digest


def test_input_manifest_does_not_associate_any_mismatched_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "prepared.csv"
    input_path.write_text("current", encoding="utf-8")
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    _write_manifest(tmp_path / "prepared.manifest.json", "0" * 64)
    _write_manifest(tmp_path / "manifest.json", "1" * 64)

    assert _input_manifest(input_path) == {"input_sha256": digest}


def test_provenance_read_failure_writes_failed_record_and_preserves_original_error(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("rgiid,year,dhdt,x\nA,2000,1,2\nA,2001,2,3\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    config = RunConfig(
        data=DataConfig(input_path, output_dir=output_dir),
        model=ModelConfig("random_forest", {"n_estimators": 1}),
        validation=ValidationConfig("loyo", 2),
    )
    original_read_bytes = Path.read_bytes

    def fail_input_digest(path: Path) -> bytes:
        if path == input_path:
            raise OSError("original provenance failure")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_input_digest)

    with pytest.raises(OSError, match="original provenance failure"):
        run_evaluation(config)

    record = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["input_manifest"] == {}
    assert record["error"]["type"] == "OSError"


def test_run_consumes_the_same_single_input_snapshot_used_for_digest(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "input.csv"
    original = b"rgiid,year,dhdt,x\nA,2000,1,2\nA,2001,2,3\nB,2000,3,4\nB,2001,4,5\n"
    replacement = original + b"C,2000,5,6\nC,2001,6,7\n"
    input_path.write_bytes(original)
    output_dir = tmp_path / "output"
    config = RunConfig(
        data=DataConfig(input_path, output_dir=output_dir),
        model=ModelConfig("random_forest", {"n_estimators": 1}),
        validation=ValidationConfig("loyo", 2),
    )
    original_read_bytes = Path.read_bytes
    reads = 0

    def mutate_after_read(path: Path) -> bytes:
        nonlocal reads
        data = original_read_bytes(path)
        if path == input_path:
            reads += 1
            path.write_bytes(replacement)
        return data

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)

    run_evaluation(config)

    record = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert reads == 1
    assert record["input_manifest"]["input_sha256"] == hashlib.sha256(original).hexdigest()
    assert record["data_summary"]["rows"] == 4
    assert record["data_summary"]["glaciers"] == 2


def test_each_manifest_candidate_uses_one_byte_snapshot_for_parse_and_digest(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "prepared.csv"
    input_bytes = b"current-input"
    input_path.write_bytes(input_bytes)
    digest = hashlib.sha256(input_bytes).hexdigest()
    exact = tmp_path / "prepared.manifest.json"
    generic = tmp_path / "manifest.json"
    exact.write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
    matching_bytes = json.dumps({"sha256": digest}).encode()
    generic.write_bytes(b"this path content must not be read")
    original_read_bytes = Path.read_bytes
    reads = {exact: 0, generic: 0}

    def candidate_snapshots(path: Path) -> bytes:
        if path == exact:
            reads[path] += 1
            raise OSError("optional exact manifest became unreadable")
        if path == generic:
            reads[path] += 1
            path.write_text(json.dumps({"sha256": "1" * 64}), encoding="utf-8")
            return matching_bytes
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", candidate_snapshots)

    identity = _input_manifest(input_path, input_bytes=input_bytes)

    assert reads == {exact: 1, generic: 1}
    assert identity["manifest_path"] == "manifest.json"
    assert identity["manifest_sha256"] == hashlib.sha256(matching_bytes).hexdigest()
    assert identity["manifest_input_sha256"] == digest
