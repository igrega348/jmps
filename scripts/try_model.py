# %%
import sys
from pathlib import Path
folder = Path(__file__).parents[1]
if str(folder) not in sys.path:
    sys.path.insert(0, str(folder))
from argparse import Namespace
from torch_geometric.data import Batch

from gnn import GLAMM_Dataset, PositiveLiteGNN
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
params = Namespace(
    hidden_irreps='16x0e+16x1o+16x2e+16x3o+16x4e',
    num_edge_bases=10,
    max_edge_radius=0.018, # dset.data.edge_attr.max()=0.0178
    lmax=4,
    readout_irreps='8x0e+8x2e+8x4e',
    message_passes=2,
    agg_norm_const=4.0,
    interaction_reduction='sum',
    correlation=3,
    inter_MLP_dim=64,
    inter_MLP_layers=3,
    global_reduction='mean',
    positive_function='matrix_power_2'
)
model = PositiveLiteGNN(params)
# %%
model(batch)
# %%
