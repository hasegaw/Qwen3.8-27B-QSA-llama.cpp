"""Check native M-RoPE sparse attention and cache restoration against HF."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from qsa.modeling_qsa import QSAQwen3_5ForConditionalGeneration

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('fixture', type=Path)
parser.add_argument('executable', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
model = QSAQwen3_5ForConditionalGeneration.from_pretrained(args.fixture, dtype=torch.float32,
                                                         attn_implementation='sdpa').eval()
ids = torch.arange(1, 30)[None]
rotary = []
for i in range(29):
    image = 3 <= i < 19
    t = i if i < 3 else 3 if image else i - 12
    rotary.append([t, 3 + (i-3)//4 if image else t, 3 + (i-3)%4 if image else t])
positions = torch.cat((torch.arange(29).reshape(1, 1, 29), torch.tensor(rotary).T[:, None]), 0)
with torch.inference_mode():
    sparse = model(ids, position_ids=positions, use_cache=False).logits.float().numpy()[0]
    for layer in model.model.language_model.layers:
        if hasattr(layer, 'self_attn'):
            layer.self_attn._qsa_budget = 1000
    dense = model(ids, position_ids=positions, use_cache=False).logits.float().numpy()[0]
rows = []
for mode, label, expected in [(0, 'dense', dense), (1, 'sparse', sparse), (2, 'mask_reference', sparse)]:
    for chunk in (7, 64):
        output = args.fixture / f'mm2-mrope-{mode}-{chunk}.f32'
        run = subprocess.run([str(args.executable), str(args.fixture/'model.gguf'), str(output), str(mode), str(chunk)],
                             capture_output=True, text=True)
        (args.fixture/f'mm2-mrope-{mode}-{chunk}.log').write_text(run.stdout + run.stderr)
        run.check_returncode()
        got = np.fromfile(output, dtype=np.float32).reshape(29, 128)
        assert np.isfinite(got).all()
        error = float(np.abs(got - expected).max())
        same = int((got.argmax(-1) == expected.argmax(-1)).sum())
        row = {'mode': label, 'ubatch': chunk, 'max_abs_logit_error': error, 'argmax_agreement': same, 'tokens': 29}
        rows.append(row)
        print(json.dumps(row), flush=True)
        assert error < 0.003 and same == 29, row
args.output.write_text(json.dumps({'fixture': str(args.fixture), 'checks': rows}, indent=2))
