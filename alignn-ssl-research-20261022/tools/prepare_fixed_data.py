"""Rebuild the existing 10,000-material split without random resampling."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_utils import MANIFEST, IDENTITY, reference, file_sha256, verify_prepared_data


def prepare_fixed(source, out, manifest_path=MANIFEST, identity_path=IDENTITY):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite: {out}')
    manifest, identity, names = reference(manifest_path, identity_path)
    labels = {}
    with (source/'id_prop.csv').open(encoding='utf-8-sig', newline='') as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if len(row) != 2 or row[0] in labels:
                raise ValueError(f'Expected unique two-column ID/target rows, no header; line {line_number}')
            value = float(row[1])
            if not math.isfinite(value):
                raise ValueError(f'Nonfinite target: {row[0]}')
            labels[row[0]] = value
    # Preserve the exact historical JSON serialization and material ordering.
    payload = json.dumps({name: labels[name] for name in names}).encode('utf-8')
    if hashlib.sha256(payload).hexdigest() != identity['targets_sha256']:
        raise ValueError('Selected labels differ from the original experiment')
    for name in names:
        if file_sha256(source/name) != manifest['structure_sha256'][name]:
            raise ValueError(f'Structure differs from original experiment: {name}')
    temporary = out.with_name(out.name+'.preparing')
    if temporary.exists():
        raise FileExistsError(f'Interrupted preparation exists; inspect it first: {temporary}')
    (temporary/'structures').mkdir(parents=True)
    for name in names:
        shutil.copyfile(source/name, temporary/'structures'/name)
    shutil.copyfile(manifest_path, temporary/'manifest.json')
    (temporary/'targets.json').write_bytes(payload)
    result = verify_prepared_data(temporary, manifest_path, identity_path)
    temporary.rename(out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Original POSCAR directory containing id_prop.csv')
    parser.add_argument('--out', type=Path, default=Path('data/data10000'))
    args = parser.parse_args()
    print(json.dumps(prepare_fixed(args.source, args.out), indent=2))


if __name__ == '__main__':
    main()
