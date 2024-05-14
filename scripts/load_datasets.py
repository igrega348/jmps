# %%
import sys
from pathlib import Path
folder = Path(__file__).parents[1]
if str(folder) not in sys.path:
    sys.path.insert(0, str(folder))
import random

from tqdm import trange, tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from lattices.lattices import Catalogue
# %% Load just without imperfections and choose the ones which have smaller number of nodes/edges
base_names = set()
all_num_edges = []
all_num_nodes = []
for i in range(5):
    f = Path(f"E:/Dropbox (Cambridge University) (Old)/neural-networks/GLAMM/rad_dset_0/cat_{i:02d}_rad.lat")
    cat = Catalogue.from_file(f, 0, regex='.*_p_0.0_.*') # no imperfections
    for data in tqdm(cat):
        name = data['name']
        bn = Catalogue.n_2_bn(name)
        num_edges = len(data.fundamental_edge_adjacency)
        num_nodes = len(np.unique(data.fundamental_edge_adjacency))
        if num_edges < 50 and num_nodes < 20:
            all_num_edges.append(num_edges)
            all_num_nodes.append(num_nodes)
            base_names.add(bn)
print(len(base_names)) # 8723 -- filtered
base_names = list(base_names)
# %%
df = pd.DataFrame({'num_edges': all_num_edges, 'num_nodes': all_num_nodes})
df.describe()
# %% Split base names into train, val, test approx 80/10/10
n = len(base_names)
n_train, n_val = 3600, 450
n_test = n-n_train-n_val
random.shuffle(base_names)
train = base_names[:n_train]
val = base_names[n_train:n_train+n_val]
test = base_names[n_train+n_val:]
# save to file
Path('train.txt').write_text('\n'.join(train))
Path('val.txt').write_text('\n'.join(val))
Path('test.txt').write_text('\n'.join(test))
# %% Assemble train catalogue
cat_dicts = {'train':{}, 'val':{}, 'test':{}}
for i in range(5):
    f = Path(f"E:/Dropbox (Cambridge University) (Old)/neural-networks/GLAMM/rad_dset_0/cat_{i:02d}_rad.lat")
    cat = Catalogue.from_file(f, 0)
    for data in tqdm(cat):
        name = data['name']
        bn = Catalogue.n_2_bn(name)
        if bn in train:
            cat_dicts['train'][name] = data
        elif bn in val:
            cat_dicts['val'][name] = data
        elif bn in test:
            cat_dicts['test'][name] = data
train_cat = Catalogue.from_dict(cat_dicts['train'])
train_cat.to_file('train.lat')
val_cat = Catalogue.from_dict(cat_dicts['val'])
val_cat.to_file('val.lat')
test_cat = Catalogue.from_dict(cat_dicts['test'])
test_cat.to_file('test.lat')
# %%
