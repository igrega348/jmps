# %%
import sys
from pathlib import Path
folder = Path(__file__).parents[1]
if str(folder) not in sys.path:
    sys.path.insert(0, str(folder))
import random

from torch_geometric.data import Batch
from tqdm import trange, tqdm

from lattices.lattices import Catalogue
from gnn import GLAMM_Dataset
# %% Load all base names from catalogues
base_names = set()
for i in trange(50):
    f = Path(f'E:/rad_dset_0/cat_{i:02d}_rad.lat')
    base_names = base_names | set([Catalogue.n_2_bn(n) for n in Catalogue.get_names(f)])
print(len(base_names)) # 10163
# %%
base_names = set()
for i in range(5):
    f = Path(f"E:/Dropbox (Cambridge University)/neural-networks/GLAMM/rad_dset_0/cat_{i:02d}_rad.lat")
    base_names = base_names | set([Catalogue.n_2_bn(n) for n in Catalogue.get_names(f)])
print(len(base_names)) # 8723 -- filtered
base_names = list(base_names)
# %% Split base names into train, val, test approx 80/10/10
n = len(base_names)
n_train, n_val, n_test = 7000, 873, n-7000-873
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
    f = Path(f"E:/Dropbox (Cambridge University)/neural-networks/GLAMM/rad_dset_0/cat_{i:02d}_rad.lat")
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
