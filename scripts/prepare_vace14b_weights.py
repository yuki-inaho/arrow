#!/usr/bin/env python3
"""Pinned VACE14B -> explicitly derived BF16, CPU only, one raw shard at a time.

No network in --plan or --self-test. Actual download requires --execute.
Admission: remaining 35GiB derived budget + one largest raw shard +64MiB
metadata headroom, while preserving a separate35GiB free-space reserve.
Only owned downloaded raw shards are deleted, after persisted verified receipts.
Existing source weights are hash-verified and hardlinked, never overwritten.
The pinned inventory is repository-owned; all model/scratch locations are explicit.
Derived output, raw scratch, and shared weights must use one filesystem.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
REPO = 'Wan-AI/Wan2.1-VACE-14B-diffusers'
REVISION = 'db79b90c60bbb45ceec9e41b9d5a4df934538ac4'
HERE = Path(__file__).resolve().parent
DEFAULT_INVENTORY = HERE.parent/'requirements/vace14b_inventory.json'
GIB = 1024**3
RESERVE = 35*GIB
DERIVED_BUDGET = 35*GIB
HEADROOM = 64*1024**2


def hashes(path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    blob = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            sha.update(block)
            blob.update(block)
    return {'bytes': size, 'sha256': sha.hexdigest(), 'git_blob_sha1': blob.hexdigest()}


def verified_file(path, entry):
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f'Expected regular file: {path}')
    actual = hashes(path)
    if actual['bytes'] != entry['bytes']:
        raise RuntimeError(f'Size mismatch: {path}')
    expected = entry.get('sha256') or entry.get('lfs')
    if expected:
        if actual['sha256'] != expected:
            raise RuntimeError(f'LFS SHA256 mismatch: {path}')
    elif actual['git_blob_sha1'] != entry['blob_id']:
        raise RuntimeError(f'Git blob SHA1 mismatch: {path}')
    return actual


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('x') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def own_directory(path, role, inventory_sha):
    if path.is_symlink():
        raise RuntimeError(f'Symlink directory rejected: {path}')
    marker = path/'.vace14b_owned.json'
    if marker.is_symlink():
        raise RuntimeError(f'Symlink ownership marker rejected: {marker}')
    expected = {'repo': REPO, 'revision': REVISION, 'role': role, 'inventory_sha256': inventory_sha}
    if path.exists() and not marker.exists() and any(path.iterdir()):
        raise RuntimeError(f'Nonempty unowned destination: {path}')
    path.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise RuntimeError(f'Ownership mismatch: {path}')
    else:
        atomic_json(marker, expected)


def existing_directory_ancestor(path):
    """Find the filesystem that will contain a destination without creating it."""
    path = Path(path).expanduser().absolute()
    while not path.exists():
        parent = path.parent
        if parent == path:
            raise RuntimeError(f'No existing ancestor for destination: {path}')
        path = parent
    if not path.is_dir():
        raise RuntimeError(f'Destination ancestor is not a directory: {path}')
    return path


def resolved_directories(args):
    requested = [Path(p).expanduser() for p in (args.raw_dir, args.output_dir, args.shared_dir)]
    if any(p.is_symlink() for p in requested):
        raise RuntimeError('Symlink raw/output/shared directory rejected')
    raw_root, output, shared = (p.resolve() for p in requested)
    pairs = [(raw_root, output), (raw_root, shared), (output, shared)]
    if any(a == b or a in b.parents or b in a.parents for a, b in pairs):
        raise RuntimeError('Raw/output/shared paths must be disjoint')
    devices = {existing_directory_ancestor(p).stat().st_dev for p in (raw_root, output, shared)}
    if len(devices) != 1:
        raise RuntimeError('Raw/output/shared paths must share a filesystem for disk admission and hardlinks')
    return raw_root, output, shared


def admission(path, derived_bytes, largest_raw, reason):
    filesystem_path = existing_directory_ancestor(path)
    free = shutil.disk_usage(filesystem_path).free
    remaining = max(0, DERIVED_BUDGET-derived_bytes)
    need = RESERVE + remaining + largest_raw + HEADROOM
    row = {'event': 'disk_admission', 'reason': reason, 'filesystem_path': str(filesystem_path), 'free_bytes': free,
           'reserve_bytes': RESERVE, 'remaining_derived_budget': remaining,
           'raw_peak_bytes': largest_raw, 'required_free_bytes': need}
    print(json.dumps(row), flush=True)
    if free < need:
        raise RuntimeError(f'Insufficient disk reserve before {reason}: free={free}, need={need}')
    return row


def cpu_modules():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    return torch, safe_open, save_file


def converted_file_upper_bound(raw):
    """Read only the safetensors header before admitting the derived writer."""
    with raw.open('rb') as f:
        header_bytes = int.from_bytes(f.read(8), 'little')
        if header_bytes <= 0 or header_bytes > 16*1024**2:
            raise RuntimeError('Invalid safetensors header length')
        header = json.loads(f.read(header_bytes))
    payload = 0
    for key, info in header.items():
        if key == '__metadata__':
            continue
        dtype = info['dtype']
        if dtype in {'F64', 'F32', 'F16', 'BF16', 'F8_E4M3', 'F8_E5M2'}:
            payload += 2*math.prod(info['shape'])
        elif dtype in {'BOOL', 'I8', 'U8', 'I16', 'U16', 'I32', 'U32', 'I64', 'U64'}:
            payload += info['data_offsets'][1]-info['data_offsets'][0]
        else:
            raise RuntimeError(f'Unsupported raw dtype: {dtype}')
    # Header can grow slightly with conversion provenance and dtype labels.
    return payload + header_bytes + 1024**2


def validate_derived(path, schemas, expected_keys, raw=None):
    torch, safe_open, _ = cpu_modules()
    with safe_open(str(path), framework='pt', device='cpu') as out:
        if set(out.keys()) != set(expected_keys) or set(schemas) != set(expected_keys):
            raise RuntimeError('Derived key set mismatch')
        original = safe_open(str(raw), framework='pt', device='cpu') if raw else None
        if original:
            original.__enter__()
        try:
            for key in sorted(expected_keys):
                tensor = out.get_tensor(key)
                schema = schemas[key]
                if list(tensor.shape) != schema['shape'] or str(tensor.dtype) != schema['derived_dtype']:
                    raise RuntimeError(f'Derived schema mismatch: {key}')
                if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                    raise RuntimeError(f'Nonfinite derived tensor: {key}')
                if original:
                    reference = original.get_tensor(key)
                    expected = reference.to(torch.bfloat16) if reference.is_floating_point() else reference
                    if not torch.equal(tensor, expected):
                        raise RuntimeError(f'Derived values differ from declared conversion: {key}')
        finally:
            if original:
                original.__exit__(None, None, None)


def convert_shard(raw, destination, entry, expected_keys, receipt_path):
    """No deletion here: caller must persist receipt before deleting owned raw."""
    torch, safe_open, save_file = cpu_modules()
    before = verified_file(raw, entry)
    if destination.exists() or receipt_path.exists():
        raise RuntimeError('Unreceipted/existing derived output is never overwritten')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    if partial.exists():
        raise RuntimeError(f'Orphan partial needs explicit review: {partial}')
    values = {}
    schemas = {}
    raw_dtypes = Counter()
    derived_dtypes = Counter()
    payload_bytes = 0
    tensor = converted = None
    with safe_open(str(raw), framework='pt', device='cpu') as source:
        if set(source.keys()) != set(expected_keys):
            raise RuntimeError('Raw key set differs from pinned weight_map')
        for key in sorted(source.keys()):
            tensor = source.get_tensor(key)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f'Nonfinite upstream tensor: {key}')
            converted = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
            values[key] = converted.contiguous()
            schemas[key] = {'shape': list(tensor.shape), 'raw_dtype': str(tensor.dtype),
                            'derived_dtype': str(converted.dtype), 'numel': tensor.numel()}
            raw_dtypes[str(tensor.dtype)] += 1
            derived_dtypes[str(converted.dtype)] += 1
            payload_bytes += converted.numel()*converted.element_size()
        save_file(values, str(partial), metadata={'format': 'pt', 'derived_from_revision': REVISION,
                                                  'conversion': 'floating tensors -> BF16 on CPU; nonfloats unchanged'})
    del values, tensor, converted
    gc.collect()
    validate_derived(partial, schemas, expected_keys, raw=raw)
    with partial.open('rb') as f:
        os.fsync(f.fileno())
    os.rename(partial, destination)
    receipt = {'status': 'verified_derived_bf16', 'repo': REPO, 'revision': REVISION,
               'raw_path': entry['path'], 'raw': before, 'derived': hashes(destination),
               'derived_payload_bytes': payload_bytes, 'tensor_count': len(schemas),
               'raw_dtype_counts': dict(raw_dtypes), 'derived_dtype_counts': dict(derived_dtypes),
               'schemas': schemas, 'all_derived_finite': True, 'exact_declared_conversion_verified': True}
    atomic_json(receipt_path, receipt)
    return receipt


def resume_shard(destination, entry, keys, receipt_path):
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('status') != 'verified_derived_bf16' or receipt.get('revision') != REVISION
            or receipt['raw_path'] != entry['path'] or receipt['raw']['sha256'] != entry['sha256']):
        raise RuntimeError('Resume receipt differs from pinned upstream shard')
    if hashes(destination) != receipt['derived']:
        raise RuntimeError('Existing derived hash/size mismatch; no force skip')
    validate_derived(destination, receipt['schemas'], keys)
    return receipt


def delete_verified_owned_raw(raw, raw_root, entry, receipt_path):
    if (raw.is_symlink() or not raw.resolve().is_relative_to(raw_root.resolve())
            or raw.resolve() != (raw_root/entry['path']).resolve()):
        raise RuntimeError('Raw cleanup ownership/path mismatch')
    receipt = json.loads(receipt_path.read_text())
    if receipt['raw']['sha256'] != entry['sha256'] or receipt['status'] != 'verified_derived_bf16':
        raise RuntimeError('Missing persisted verified conversion receipt')
    if raw.exists():
        verified_file(raw, entry)
        raw.unlink()


def execute(args, inventory):
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    from importlib.metadata import version
    inventory_sha = hashes(args.inventory)['sha256']
    raw_root, output, shared = resolved_directories(args)
    shards = [e for e in inventory['transformer'] if e['path'].endswith('.safetensors')]
    existing = sum(p.stat().st_size for p in (output/'transformer').glob('*.safetensors'))
    admission(output, existing, max(e['bytes'] for e in shards), 'before ownership/lock writers')
    own_directory(output, 'derived_bf16', inventory_sha)
    own_directory(raw_root, 'raw_download_scratch', inventory_sha)
    for directory in (output, raw_root):
        if any(p.is_symlink() for p in directory.rglob('*')):
            raise RuntimeError(f'Symlinks inside owned directory rejected: {directory}')
    with (output/'.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute_locked(args, inventory, inventory_sha, raw_root, output, shared,
                              hf_hub_download, get_hf_file_metadata, hf_hub_url, version)


def execute_locked(args, inventory, inventory_sha, raw_root, output, shared, download, metadata, url, version):
    started = time.monotonic()
    shards = [e for e in inventory['transformer'] if e['path'].endswith('.safetensors')]
    largest = max(e['bytes'] for e in shards)
    receipts_dir = output/'conversion_receipts'
    receipts_dir.mkdir(exist_ok=True)
    derived_bytes = sum(p.stat().st_size for p in (output/'transformer').glob('*.safetensors'))
    admission(output, derived_bytes, largest, 'initial complete-job budget')
    # One sequential download at a time; reject unexpected retained full shards.
    expected_paths = {raw_root/e['path'] for e in shards}
    retained = list(raw_root.rglob('*.safetensors'))
    if any(p not in expected_paths for p in retained) or len(retained) > 1:
        raise RuntimeError('Unexpected/multiple retained raw shards')
    shared_receipts = []
    for entry in inventory['shared_candidates']:
        if not entry.get('identical13b'):
            raise RuntimeError('Shared candidate is not pinned identical')
        src, dst = shared/entry['path'], output/entry['path']
        identity = verified_file(src, entry)
        admission(output, derived_bytes, largest, 'hardlink '+entry['path'])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            verified_file(dst, entry)
            if not os.path.samefile(src, dst):
                raise RuntimeError('Existing shared file is not the verified hardlink')
        else:
            os.link(src, dst)
        shared_receipts.append({'path': entry['path'], 'source': str(src), 'identity': identity, 'hardlinked': True})
    tiny = [e for e in inventory['transformer'] if not e['path'].endswith('.safetensors')]
    meta = metadata(url(REPO, 'model_index.json', revision=REVISION))
    if meta.commit_hash != REVISION or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', meta.etag or ''):
        raise RuntimeError('model_index metadata is not exact pinned commit/hash')
    tiny.append({'path': 'model_index.json', 'bytes': meta.size,
                 'sha256': meta.etag if len(meta.etag) == 64 else None, 'blob_id': meta.etag})
    tiny_receipts = []
    upstream_index = None
    for entry in tiny:
        admission(output, derived_bytes, largest, 'tiny pinned metadata '+entry['path'])
        raw = Path(download(REPO, entry['path'], revision=REVISION, local_dir=raw_root))
        identity = verified_file(raw, entry)
        if entry['path'].endswith('.index.json'):
            upstream_index = json.loads(raw.read_text())
            dst = output/'upstream_transformer.index.json'
        else:
            dst = output/entry['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if hashes(dst) != identity:
                raise RuntimeError('Existing metadata differs from pinned upstream')
        else:
            with dst.open('xb') as f:
                f.write(raw.read_bytes())
                f.flush()
                os.fsync(f.fileno())
        tiny_receipts.append({'path': entry['path'], 'identity': identity})
    if upstream_index is None:
        raise RuntimeError('Missing upstream weight index')
    weight_map = upstream_index['weight_map']
    if set(weight_map.values()) != {Path(e['path']).name for e in shards}:
        raise RuntimeError('Upstream index shard set differs from inventory')
    receipts = []
    # Finish any already downloaded shard first, so at most one raw is retained.
    retained = list(raw_root.rglob('*.safetensors'))
    shards.sort(key=lambda e: (0 if raw_root/e['path'] in retained else 1, e['path']))
    for entry in shards:
        dst = output/entry['path']
        raw = raw_root/entry['path']
        recpath = receipts_dir/(Path(entry['path']).name+'.json')
        keys = {k for k, shard in weight_map.items() if shard == Path(entry['path']).name}
        if recpath.exists():
            receipt = resume_shard(dst, entry, keys, recpath)
            delete_verified_owned_raw(raw, raw_root, entry, recpath)
        else:
            if dst.exists():
                raise RuntimeError('Existing unreceipted derived shard; stop for review')
            admission(output, derived_bytes, largest, 'download '+entry['path'])
            if any(p != raw for p in raw_root.rglob('*.safetensors')):
                raise RuntimeError('Another raw shard is retained')
            raw = Path(download(REPO, entry['path'], revision=REVISION, local_dir=raw_root))
            verified_file(raw, entry)
            upper_bound = converted_file_upper_bound(raw)
            if derived_bytes + upper_bound > DERIVED_BUDGET:
                raise RuntimeError('Next derived shard exceeds admitted complete-job budget')
            # Raw is already on disk; a second full-raw allowance is unnecessary.
            admission(output, derived_bytes, 0, 'convert '+entry['path'])
            receipt = convert_shard(raw, dst, entry, keys, recpath)
            derived_bytes += receipt['derived']['bytes']
            if derived_bytes > DERIVED_BUDGET:
                raise RuntimeError('Actual derived files exceeded admitted budget')
            delete_verified_owned_raw(raw, raw_root, entry, recpath)
        receipts.append(receipt)
        print(json.dumps({'event': 'shard_verified_and_raw_removed', 'path': entry['path'],
                          'derived_bytes': receipt['derived']['bytes'], 'free_bytes': shutil.disk_usage(output).free}), flush=True)
        if shutil.disk_usage(output).free < RESERVE:
            raise RuntimeError('Free-space reserve violated')
    payload = sum(r['derived_payload_bytes'] for r in receipts)
    derived_index = dict(upstream_index)
    derived_index['metadata'] = dict(upstream_index.get('metadata', {}), total_size=payload)
    admission(output, derived_bytes, 0, 'derived index and provenance')
    atomic_json(output/'transformer/diffusion_pytorch_model.safetensors.index.json', derived_index)
    provenance = {'status': 'complete_verified_derived_bf16_not_untouched_upstream', 'repo': REPO, 'revision': REVISION,
        'inventory_sha256': inventory_sha, 'script_sha256': hashes(Path(__file__))['sha256'],
        'conversion': 'All floating transformer tensors converted on CPU to BF16; all nonfloating tensors unchanged. '
                      'Original config/model_index copied exactly; upstream torch_dtype metadata does not override derived tensor provenance.',
        'shared_files': shared_receipts, 'upstream_metadata_files': tiny_receipts,
        'shards': [{k: v for k, v in r.items() if k != 'schemas'} for r in receipts],
        'derived_transformer_file_bytes': derived_bytes, 'derived_tensor_payload_bytes': payload,
        'index_weight_map_unchanged': derived_index['weight_map'] == upstream_index['weight_map'],
        'reserve_bytes': RESERVE, 'max_raw_shards_in_flight_or_retained': 1,
        'versions': {n: version(n) for n in ['torch', 'safetensors', 'huggingface-hub']},
        'wall_seconds': time.monotonic()-started, 'final_free_bytes': shutil.disk_usage(output).free}
    atomic_json(output/'DERIVED_BF16_PROVENANCE.json', provenance)
    print(json.dumps({'status': provenance['status'], 'wall_seconds': provenance['wall_seconds'],
                      'derived_bytes': derived_bytes, 'final_free_bytes': provenance['final_free_bytes']}), flush=True)


def self_test():
    torch, _, save_file = cpu_modules()
    with tempfile.TemporaryDirectory(prefix='vace14b-conversion-fixture-') as folder:
        root = Path(folder)
        raw, dst, receipt_path = root/'raw.safetensors', root/'derived.safetensors', root/'receipt.json'
        values = {'f32': torch.tensor([1., -2.125, .1]), 'f16': torch.ones(2, 3, dtype=torch.float16),
                  'bf16': torch.arange(4, dtype=torch.bfloat16), 'integer': torch.tensor([1, -2], dtype=torch.int64),
                  'boolean': torch.tensor([True, False])}
        save_file(values, str(raw))
        entry = {'path': raw.name, **hashes(raw)}
        receipt = convert_shard(raw, dst, entry, set(values), receipt_path)
        resume_shard(dst, entry, set(values), receipt_path)
        delete_verified_owned_raw(raw, root, entry, receipt_path)
        assert not raw.exists() and dst.exists() and receipt_path.exists()
        try:
            resume_shard(dst, entry | {'sha256': '0'*64}, set(values), receipt_path)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Bad upstream hash receipt was accepted')
        badraw = root/'nonfinite.safetensors'
        badout = root/'nonfinite-derived.safetensors'
        badreceipt = root/'nonfinite-receipt.json'
        save_file({'bad': torch.tensor([float('nan')])}, str(badraw))
        try:
            convert_shard(badraw, badout, {'path': badraw.name, **hashes(badraw)}, {'bad'}, badreceipt)
        except RuntimeError:
            assert badraw.exists() and not badout.exists() and not badreceipt.exists()
        else:
            raise AssertionError('Nonfinite raw tensor was accepted')
        print(json.dumps({'self_test': 'pass', 'tensor_count': receipt['tensor_count'],
                          'raw_dtype_counts': receipt['raw_dtype_counts'], 'derived_dtype_counts': receipt['derived_dtype_counts'],
                          'checks': ['finite_shape_key_dtype_value_validation', 'receipt_resume_verification',
                                     'nonfloating_preserved', 'owned_raw_deleted_only_after_receipt', 'bad_receipt_rejected',
                                     'nonfinite_rejected_without_raw_deletion'],
                          'network_calls': 0, 'gpu_calls': 0}), flush=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--inventory', type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument('--shared-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, required=True)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return
    inventory = json.loads(args.inventory.read_text())
    if inventory['revision'] != REVISION:
        raise RuntimeError('Inventory revision mismatch')
    for entry in inventory['transformer'] + inventory['shared_candidates']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Inventory contains a nonrelative/escaping path')
    if args.plan:
        raw_root, output, shared = resolved_directories(args)
        largest = max(e['bytes'] for e in inventory['transformer'] if e['path'].endswith('.safetensors'))
        existing = sum(p.stat().st_size for p in (output/'transformer').glob('*.safetensors'))
        admission(output, existing, largest, 'network-free plan')
        print(json.dumps({'repo': REPO, 'revision': REVISION, 'mode': 'plan_no_download',
                          'raw_shards_at_once': 1, 'additional_disk_peak_bound': DERIVED_BUDGET+largest+HEADROOM,
                          'output': str(output), 'raw_scratch': str(raw_root), 'shared': str(shared),
                          'reserve_bytes': RESERVE, 'actual_execution_requires': '--execute'}), flush=True)
    else:
        execute(args, inventory)


if __name__ == '__main__':
    main()
