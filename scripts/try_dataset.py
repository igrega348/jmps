# %%
import sys
from pathlib import Path
folder = Path(__file__).parents[1]
if str(folder) not in sys.path:
    sys.path.insert(0, str(folder))

from torch_geometric.data import Batch

from gnn import GLAMM_Dataset
# %%
dset = GLAMM_Dataset(
    root=Path(__file__).parents[1] / 'dset',
    catalogue_path=r"E:\Dropbox (Cambridge University)\neural-networks\GLAMM\rad_dset_0\cat_00_rad.lat",
    dset_fname='sample.pt',
    choose_reldens='all',
)

print(dset)
print(dset[0])

batch = Batch.from_data_list([dset[0], dset[1]])
print(batch)
# %%
d = dset[0]
d
# %%
batch.pos
# %%
d.pos
# %%
d.transformed_edge_shifts