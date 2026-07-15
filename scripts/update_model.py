"""
This script translates the weights from the model instantiated for
https://doi.org/10.5194/egusphere-2023-1953, which was created using
quantnn < 0.0.5, to the same network but that uses different parameter
names as it is instantiated with quantnn == 0.0.5
"""

import argparse
from pathlib import Path

from ccic.models import CCICModel
from quantnn.mrnn import MRNN

def translate_parameter_name(old: str) -> str:
    if 'decoder.stages' in old:
        s = old.split('.')
        pre = '.'.join(s[:3])
        post = '.'.join(s[4:])
        return f'{pre}.{post}'
    if 'encoder.stages' in old:
        s = old.split('.')
        pre = '.'.join(s[:2])
        post = '.'.join(s[3:])
        return f'{pre}.{int(s[2])+1}.{post}'
    return old

parser = argparse.ArgumentParser(
    description="Translate network weights from old to new model"
)

parser.add_argument('--old', required=True, help='The old network.')
parser.add_argument('--new', required=True, help='Where to save the new network.')

args = parser.parse_args()

assert not Path(args.new).exists()

old = MRNN.load(args.old)

old_named_params = [e[0] for e in old.model.named_parameters()]

n_stages=len(set(['.'.join(e.split('.')[:3]) for e in old_named_params if 'encoder' in e]))
n_blocks=len(set([e.split('block')[-1].split('.')[1] for e in old_named_params if 'encoder' in e and 'block' in e]))
n_features=min([param.shape[0] for name, param in old.model.named_parameters() if 'decoder' in name])

new = CCICModel(n_stages=n_stages, features=n_features, n_quantiles=64, n_blocks=n_blocks)


translated = [translate_parameter_name(p) for p in old_named_params]
for n, _ in new.named_parameters():
    assert n in translated

old_state_dict = old.model.state_dict()
new_state_dict = new.state_dict()
for old_name in old_named_params:
    new_state_dict[translate_parameter_name(old_name)] = old_state_dict[old_name]

new.load_state_dict(new_state_dict)

mrnn = MRNN(model=new, losses=old.losses, transformation=old.transformation)

mrnn.save(args.new)