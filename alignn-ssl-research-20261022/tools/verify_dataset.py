"""Verify IDs, split, targets and all structure bytes without training."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_utils import verify_prepared_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_prepared_data(args.data), indent=2))
