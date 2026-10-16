"""CPU-only verification of pinned VACE14B conversion and scratch ownership."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_vace14b_under_test", ROOT / "scripts/prepare_vace14b_weights.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("Cannot load VACE14B preparation script")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def raw_fixture(tmp_path: Path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = raw_root / "fixture.safetensors"
    values = {
        "float": torch.tensor([0.1, -2.125, 3.5], dtype=torch.float32),
        "integer": torch.tensor([2**40, -17], dtype=torch.int64),
        "boolean": torch.tensor([True, False]),
    }
    save_file(values, str(raw))
    entry = {"path": raw.name, **prepare.hashes(raw)}
    return raw_root, raw, values, entry


def test_real_conversion_resume_and_cleanup_preserve_nonfloating_values(tmp_path):
    raw_root, raw, values, entry = raw_fixture(tmp_path)
    destination = tmp_path / "derived.safetensors"
    receipt_path = tmp_path / "receipt.json"
    receipt = prepare.convert_shard(raw, destination, entry, set(values), receipt_path)
    assert raw.exists(), "conversion alone must not delete the original"
    assert receipt["raw"]["sha256"] == entry["sha256"]
    assert receipt["derived"] == prepare.hashes(destination)
    with safe_open(str(destination), framework="pt", device="cpu") as derived:
        assert set(derived.keys()) == set(values)
        for name, original in values.items():
            actual = derived.get_tensor(name)
            expected = original.to(torch.bfloat16) if original.is_floating_point() else original
            assert actual.dtype == expected.dtype
            assert torch.equal(actual, expected)
    assert prepare.resume_shard(destination, entry, set(values), receipt_path) == receipt
    prepare.delete_verified_owned_raw(raw, raw_root, entry, receipt_path)
    assert not raw.exists() and destination.exists() and receipt_path.exists()


def test_tampered_receipt_or_derived_file_cannot_resume_or_delete_raw(tmp_path):
    raw_root, raw, values, entry = raw_fixture(tmp_path)
    destination = tmp_path / "derived.safetensors"
    receipt_path = tmp_path / "receipt.json"
    receipt = prepare.convert_shard(raw, destination, entry, set(values), receipt_path)
    receipt["raw"]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="persisted verified"):
        prepare.delete_verified_owned_raw(raw, raw_root, entry, receipt_path)
    with pytest.raises(RuntimeError, match="pinned upstream"):
        prepare.resume_shard(destination, entry, set(values), receipt_path)
    assert raw.exists()
    receipt["raw"]["sha256"] = entry["sha256"]
    receipt_path.write_text(json.dumps(receipt))
    with destination.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RuntimeError, match="hash/size mismatch"):
        prepare.resume_shard(destination, entry, set(values), receipt_path)
    assert raw.exists()


@pytest.mark.parametrize("value", [float("nan"), torch.finfo(torch.float32).max])
def test_nonfinite_input_or_bf16_overflow_retains_raw_without_receipt(tmp_path, value):
    raw = tmp_path / "raw.safetensors"
    destination = tmp_path / "derived.safetensors"
    receipt = tmp_path / "receipt.json"
    save_file({"weight": torch.tensor([value], dtype=torch.float32)}, str(raw))
    entry = {"path": raw.name, **prepare.hashes(raw)}
    with pytest.raises(RuntimeError, match="Nonfinite"):
        prepare.convert_shard(raw, destination, entry, {"weight"}, receipt)
    assert raw.exists() and not destination.exists() and not receipt.exists()


def test_raw_cleanup_rejects_outside_root_and_symlink(tmp_path):
    raw_root, raw, values, entry = raw_fixture(tmp_path)
    receipt = tmp_path / "receipt.json"
    prepare.convert_shard(raw, tmp_path / "derived.safetensors", entry, set(values), receipt)
    wrong_root = tmp_path / "other"
    wrong_root.mkdir()
    with pytest.raises(RuntimeError, match="ownership/path mismatch"):
        prepare.delete_verified_owned_raw(raw, wrong_root, entry, receipt)
    link = raw_root / "link.safetensors"
    link.symlink_to(raw)
    with pytest.raises(RuntimeError, match="ownership/path mismatch"):
        prepare.delete_verified_owned_raw(link, raw_root, entry, receipt)
    assert raw.exists() and link.is_symlink()


def test_destination_ownership_rejects_existing_files_and_foreign_marker(tmp_path):
    destination = tmp_path / "output"
    destination.mkdir()
    sentinel = destination / "user-data"
    sentinel.write_text("keep")
    with pytest.raises(RuntimeError, match="Nonempty unowned"):
        prepare.own_directory(destination, "derived_bf16", "inventory-a")
    assert sentinel.read_text() == "keep"
    clean = tmp_path / "clean"
    prepare.own_directory(clean, "derived_bf16", "inventory-a")
    with pytest.raises(RuntimeError, match="Ownership mismatch"):
        prepare.own_directory(clean, "derived_bf16", "inventory-b")
    marker = clean / ".vace14b_owned.json"
    external = tmp_path / "external-marker.json"
    marker.rename(external)
    marker.symlink_to(external)
    with pytest.raises(RuntimeError, match="Symlink ownership marker"):
        prepare.own_directory(clean, "derived_bf16", "inventory-a")
    assert external.exists()


def test_root_paths_reject_overlap_and_symlink_before_resolving(tmp_path):
    args = SimpleNamespace(raw_dir=tmp_path / "raw", output_dir=tmp_path / "out", shared_dir=tmp_path / "shared")
    args.raw_dir.mkdir()
    args.shared_dir.mkdir()
    args.output_dir = args.raw_dir / "nested"
    with pytest.raises(RuntimeError, match="disjoint"):
        prepare.resolved_directories(args)
    args.output_dir = tmp_path / "linked"
    args.output_dir.symlink_to(args.shared_dir, target_is_directory=True)
    with pytest.raises(RuntimeError, match="Symlink"):
        prepare.resolved_directories(args)


def test_separate_filesystems_rejected_before_destination_writes(tmp_path, monkeypatch):
    args = SimpleNamespace(raw_dir=tmp_path / "raw", output_dir=tmp_path / "out", shared_dir=tmp_path / "shared")

    def ancestor(path):
        return SimpleNamespace(stat=lambda: SimpleNamespace(st_dev=2 if path == args.raw_dir else 1))

    monkeypatch.setattr(prepare, "existing_directory_ancestor", ancestor)
    with pytest.raises(RuntimeError, match="share a filesystem"):
        prepare.resolved_directories(args)
    assert not any(path.exists() for path in (args.raw_dir, args.output_dir, args.shared_dir))


def test_plan_uses_destination_filesystem_and_pinned_inventory_without_network(tmp_path, monkeypatch, capsys):
    import huggingface_hub

    def no_network(*_args, **_kwargs):
        pytest.fail("plan must not use network")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_network)
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", no_network)
    checked = []

    def disk_usage(path):
        checked.append(path)
        return SimpleNamespace(free=10**12)

    monkeypatch.setattr(prepare.shutil, "disk_usage", disk_usage)
    output = tmp_path / "not-created" / "model"
    prepare.main([
        "--plan", "--shared-dir", str(tmp_path / "shared"),
        "--output-dir", str(output), "--raw-dir", str(tmp_path / "scratch"),
    ])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert checked == [tmp_path]
    assert rows[0]["filesystem_path"] == str(tmp_path)
    assert rows[-1]["mode"] == "plan_no_download"
    assert rows[-1]["output"] == str(output)
    assert not output.parent.exists()
    assert prepare.DEFAULT_INVENTORY == ROOT / "requirements/vace14b_inventory.json"
    assert prepare.hashes(prepare.DEFAULT_INVENTORY)["sha256"] == (
        "0517668be596aa4c8d9f885f6572678e6c6ec8728c3e5df74b8a1453c7e99cd6"
    )


@pytest.mark.parametrize("omitted", ["--shared-dir", "--output-dir", "--raw-dir"])
def test_cli_requires_all_storage_locations(omitted):
    argv = ["--plan"]
    for flag in ("--shared-dir", "--output-dir", "--raw-dir"):
        if flag != omitted:
            argv.extend([flag, f"/example/{flag[2:]}"])
    with pytest.raises(SystemExit) as failure:
        prepare.build_arg_parser().parse_args(argv)
    assert failure.value.code == 2


def test_admission_rejects_insufficient_reserve_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1))
    target = tmp_path / "uncreated"
    with pytest.raises(RuntimeError, match="Insufficient disk reserve"):
        prepare.admission(target, 0, 1024, "fixture")
    assert not target.exists()
