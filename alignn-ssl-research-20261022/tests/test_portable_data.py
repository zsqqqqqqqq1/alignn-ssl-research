"""Protect fixed-dataset identity when moving to another machine."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from dataset_utils import verify_prepared_data
from tools.prepare_fixed_data import prepare_fixed


class PortableDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root/'raw'; self.raw.mkdir()
        self.names = ['a.vasp','b.vasp','c.vasp']
        for n in self.names:
            (self.raw/n).write_text('synthetic '+n)
        (self.raw/'id_prop.csv').write_text('a.vasp,-1.0\nb.vasp,-2.0\nc.vasp,-3.0\n')
        self.manifest = self.root/'reference.json'
        self.manifest.write_text(json.dumps({'splits':{'train':self.names[:1], 'val':self.names[1:2],
            'test':self.names[2:]}, 'structure_sha256':{n:hashlib.sha256((self.raw/n).read_bytes()).hexdigest() for n in self.names}}))
        self.identity = self.root/'identity.json'
        targets = json.dumps(dict(zip(self.names, [-1.0,-2.0,-3.0]))).encode()
        self.identity.write_text(json.dumps({'manifest_sha256':hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            'targets_sha256':hashlib.sha256(targets).hexdigest()}))
        self.out=self.root/'prepared'

    def test_reconstruct_and_verify_preserves_reference_bytes(self):
        prepare_fixed(self.raw,self.out,self.manifest,self.identity)
        self.assertEqual((self.out/'manifest.json').read_bytes(), self.manifest.read_bytes())
        self.assertEqual(verify_prepared_data(self.out,self.manifest,self.identity)['materials'],3)
        with self.assertRaises(FileExistsError):
            prepare_fixed(self.raw,self.out,self.manifest,self.identity)

    def test_label_drift_fails_before_output(self):
        (self.raw/'id_prop.csv').write_text('a.vasp,0.0\nb.vasp,-2.0\nc.vasp,-3.0\n')
        with self.assertRaisesRegex(ValueError,'labels differ'):
            prepare_fixed(self.raw,self.out,self.manifest,self.identity)
        self.assertFalse(self.out.exists())

    def test_modified_structure_is_rejected(self):
        prepare_fixed(self.raw,self.out,self.manifest,self.identity)
        (self.out/'structures/a.vasp').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Structure checksum'):
            verify_prepared_data(self.out,self.manifest,self.identity)

    def test_manifest_reordering_is_rejected(self):
        prepare_fixed(self.raw,self.out,self.manifest,self.identity)
        (self.out/'manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'manifest.json'):
            verify_prepared_data(self.out,self.manifest,self.identity)


if __name__ == '__main__':
    unittest.main()
