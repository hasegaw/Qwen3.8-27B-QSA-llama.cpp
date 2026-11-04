"""Extend the small native-shape fixture with a deterministic native MTP head."""
import json,shutil
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
root=Path('/mnt/data2/qsa27b')
src=root/'fixture-native-shape';dst=root/'fixture-mtp'
dst.mkdir(exist_ok=True)
weights=load_file(src/'model.safetensors')
for name,value in list(weights.items()):
    prefix='model.language_model.layers.3.'
    if name.startswith(prefix) and '._qsa_indexer.' not in name:
        weights['mtp.layers.0.'+name[len(prefix):]]=value.clone()
torch.manual_seed(17)
hidden=128
weights['mtp.fc.weight']=torch.randn(hidden,hidden*2)*0.02
for name in ['pre_fc_norm_hidden','pre_fc_norm_embedding','norm']:
    weights[f'mtp.{name}.weight']=torch.randn(hidden)*0.05
save_file(weights,dst/'model.safetensors',metadata={'format':'pt'})
cfg=json.loads((src/'config.json').read_text());cfg['text_config']['mtp_num_hidden_layers']=1
(dst/'config.json').write_text(json.dumps(cfg))
print('MTP fixture',dst)
