"""Verify the released files after extraction or cloning; no training imports."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = ROOT/'provenance/package_files.sha256'
    count = 0
    failures = []
    for row in manifest.read_text(encoding='utf-8').splitlines():
        expected, name = row.split('  ', 1)
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            failures.append(name)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
        count += 1
    if failures:
        raise SystemExit('Changed or missing release files: '+', '.join(failures))
    print(f'OK: {count} released files match their recorded SHA-256.')
    print('This verifies recorded files, not authenticity or additional local files.')

if __name__ == '__main__':
    main()
