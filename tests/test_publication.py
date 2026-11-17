"""Focused, offline checks for the publication's data and training contracts."""
from pathlib import Path
import gzip,hashlib,json,sys,tempfile,unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import read,download,decode_prefix
from train_paper import command

class PublicationTests(unittest.TestCase):
    def test_frozen_case_counts(self):
        index=read(ROOT/'reproducibility/evaluation_cases.json')
        self.assertEqual(sum(v['suite']=='role_authority' for v in index.values()),800)
        self.assertEqual(sum(v['suite']=='task_controls' for v in index.values()),890)
        self.assertEqual(len({cid.rsplit(':',1)[0] for cid,v in index.items() if v['suite']=='role_authority'}),200)

    def test_training_index_and_split(self):
        with gzip.open(ROOT/'reproducibility/training/source_rows.jsonl.gz','rt',encoding='utf8') as f:rows=[json.loads(s) for s in f]
        a={r['source_row'] for r in rows if r['split']=='train'};b={r['source_row'] for r in rows if r['split']=='validation'}
        self.assertEqual((len(a),len(b)),(32046,3552));self.assertFalse(a&b)
        self.assertFalse({r['group_id'] for r in rows if r['split']=='train'}&{r['group_id'] for r in rows if r['split']=='validation'})

    def test_historical_training_source(self):
        arch=read(ROOT/'reproducibility/training/reported_architecture.json')
        for name,expected in arch['provenance']['source_sha256'].items():
            p=ROOT/'reproducibility/historical_training/ha-aside-paper'/name
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),expected,name)

    def test_inference_source(self):
        for name,expected in read(ROOT/'reproducibility/inference_source_sha256.json').items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected,name)

    def test_full_recipe_is_not_step_limited(self):
        c=command(ROOT/'outputs/test')
        self.assertIn('--single',c);self.assertIn('--no-rgag',c);self.assertNotIn('--pilot-steps',c)
        self.assertIn('--pilot-steps',command(ROOT/'outputs/test-smoke',True))

    def test_eos_boundaries(self):
        class Tokenizer:
            def decode(self,ids,skip_special_tokens):return ','.join(map(str,ids))
        t=Tokenizer()
        self.assertEqual(decode_prefix(t,[5,151645,6]),'5')
        self.assertEqual(decode_prefix(t,[5,151643,6]),'5')

    def test_modified_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';source.write_bytes(b'modified')
            with self.assertRaises(ValueError):download('https://example.invalid/',Path(d)/'target','0'*64,source)

if __name__=='__main__':unittest.main()
