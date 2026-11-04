"""Validate MM2 indexers exactly and unchanged non-indexer GGUF tensors."""
import hashlib
import json
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from safetensors import safe_open

root = Path('/mnt/data2/qsa27b')
model = root / 'models/Qwen3.8-27B-QSA-mm2'
path = root / 'gguf/Qwen3.8-27B-QSA-mm2-Q8_0.gguf'
old = GGUFReader(root / 'gguf/Qwen3.8-27B-QSA-Q8_0.gguf')
new = GGUFReader(path)
before, after = ({t.name: t for t in reader.tensors} for reader in (old, new))
assert before.keys() == after.keys()
count, unchanged, payload = 0, 0, 0
with safe_open(model / 'qsa-indexers.safetensors', framework='numpy') as source:
    for name in source.keys():
        prefix, suffix = name.split('._qsa_indexer.')
        layer = int(prefix.split('.')[3])
        mapped = {'q_proj.weight': 'q_proj.weight', 'k_proj.weight': 'k_proj.weight',
                  'q_norm': 'q_norm.weight', 'k_norm': 'k_norm.pre_weight',
                  'k_block_norm': 'k_norm.weight'}[suffix]
        want = source.get_tensor(name)
        if suffix.endswith('norm'):
            want = want + np.float32(1)
        tensor = after[f'blk.{layer}.indexer.{mapped}']
        assert int(tensor.tensor_type) == 0
        np.testing.assert_array_equal(tensor.data.reshape(want.shape), want)
        count += 1
assert count == sum('.indexer.' in k for k in after) == 80
print('Verified all 80 MM2 indexer tensors', flush=True)
for name, a in before.items():
    b = after[name]
    assert a.tensor_type == b.tensor_type and np.array_equal(a.shape, b.shape)
    if '.indexer.' in name:
        continue
    av, bv = a.data.view(np.uint8).reshape(-1), b.data.view(np.uint8).reshape(-1)
    assert av.size == bv.size
    for offset in range(0, av.size, 8*1024*1024):
        np.testing.assert_array_equal(av[offset:offset+8*1024*1024], bv[offset:offset+8*1024*1024])
    unchanged += 1
    payload += av.size
with path.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
assert digest == '97c4ecdd175664b056df1822a4e2a93e504a5a7fc8789a3a5cd51517f73c711f'
manifest = json.loads((model / 'conversion_manifest.json').read_text())
assert manifest['indexer_checkpoint_sha256'] == 'e44193e5ef832b6220c8355fff92ac8f07d8c13538f584d2b6231b5dd2f4bc51'
report = {'gguf_sha256': digest, 'indexer_tensors_verified_exactly': count,
          'unchanged_non_indexer_tensors': unchanged, 'unchanged_tensor_bytes': payload,
          'checkpoint_sha256': manifest['indexer_checkpoint_sha256']}
(root / 'logs/mm2-artifacts-validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
