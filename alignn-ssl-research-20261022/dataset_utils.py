"""Standard-library checks for the exact published experiment split.

Material IDs, structure bytes and selected labels must match the reference.
These checks do not load Torch and never alter the source dataset.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'data/splits/manifest_10000_seed123.json'
IDENTITY = ROOT / 'data/splits/dataset_identity.json'


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def reference(manifest_path=MANIFEST, identity_path=IDENTITY):
    identity = json.loads(Path(identity_path).read_text(encoding='utf-8'))
    if file_sha256(manifest_path) != identity['manifest_sha256']:
        raise ValueError('Reference manifest checksum mismatch')
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    names = sum((manifest['splits'][k] for k in ('train', 'val', 'test')), [])
    if len(names) != len(set(names)) or set(names) != set(manifest['structure_sha256']):
        raise ValueError('Invalid reference IDs')
    for name in names:
        if not name or name in ('.', '..') or '/' in name or '\\' in name or ':' in name:
            raise ValueError(f'Unsafe structure filename: {name!r}')
    return manifest, identity, names


def verify_prepared_data(data, manifest_path=MANIFEST, identity_path=IDENTITY):
    data = Path(data).resolve()
    manifest, identity, names = reference(manifest_path, identity_path)
    for name, expected in [('manifest.json', identity['manifest_sha256']),
                           ('targets.json', identity['targets_sha256'])]:
        if file_sha256(data/name) != expected:
            raise ValueError(f'{name} does not match the fixed experiment; do not resample or reformat it')
    for name in names:
        if file_sha256(data/'structures'/name) != manifest['structure_sha256'][name]:
            raise ValueError(f'Structure checksum mismatch: {name}')
    return {'materials': len(names), 'manifest_sha256': identity['manifest_sha256'],
            'targets_sha256': identity['targets_sha256']}
