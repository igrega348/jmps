# %%
import os
import sys
from pathlib import Path
par_folder = Path(__file__).absolute().parents[1]
if str(par_folder) not in sys.path:
    sys.path.insert(0, str(par_folder))
from argparse import Namespace
import time
import yaml
from typing import Any, Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelSummary, 
    ModelCheckpoint, 
    TQDMProgressBar,
    EarlyStopping
) 
from gnn.mace import get_edge_vectors_and_lengths
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch_geometric.loader import DataLoader
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from plotly.subplots import make_subplots

from gnn import PositiveLiteGNN
from gnn import GLAMM_Dataset
from gnn.callbacks import PrintTableMetrics
from train_utils import load_datasets, CfgDict, LightningWrappedModel
from lattices.lattices import plotting, elasticity_func, Catalogue
# %%
def setup_model(log_dir: Union[Path, str]):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    cfg_path = list(log_dir.glob('*.yml'))[0]
    print(f'Loading config from {cfg_path}')
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg = CfgDict(cfg)

    ############# setup data ##############
    # train_dset = load_datasets(parent=cfg.data.dset_parent, tag='train', reldens_norm=False)
    # valid_dset = load_datasets(parent=cfg.data.dset_parent, tag='valid', reldens_norm=False)
    test_dset = load_datasets(parent=Path('../../JMPS_data/aug1'), tag='test', reldens_norm=False)
    train_dset = None
    valid_dset = None

    ############# setup model ##############
    model_ckpt = list(log_dir.glob('**/*.ckpt'))[0]
    print(f'Loading model from {model_ckpt}')
    lightning_model = LightningWrappedModel.load_from_checkpoint(checkpoint_path=model_ckpt, model=PositiveLiteGNN, cfg=cfg)

    ############# setup trainer ##############
    trainer = pl.Trainer(
        accelerator='auto',
        default_root_dir=cfg.log_dir,
    )
    return trainer, lightning_model, train_dset, valid_dset, test_dset
# %%
trainer, ckpt_model, train_dset, valid_dset, test_dset = setup_model('../experiments/38')
# %%
results = {}
# %%
train_loader = DataLoader(
    dataset=train_dset[:2**13], 
    batch_size=16,
    shuffle=False,
)
train_results = trainer.predict(ckpt_model, train_loader, return_predictions=True)
# %%
names_train = np.concatenate([x[1]['name'] for x in train_results])
gt_train = torch.cat([x[1]['stiffness'] for x in train_results], dim=0)
pred_train = torch.cat([x[0]['stiffness'] for x in train_results], dim=0)
results['gt_train'] = gt_train
results['pred_train'] = pred_train
results['names_train'] = names_train
# %%
plotting.plotly_tensor_projection(elasticity_func.stiffness_Mandel_to_cart_4(gt_train[0]), title='gt')
# %%
plotting.plotly_tensor_projection(elasticity_func.stiffness_Mandel_to_cart_4(pred_train[0]), title='pred')
# %%
test_loader = DataLoader(
    dataset=test_dset, 
    batch_size=16,
    shuffle=False,
)
test_results = trainer.predict(ckpt_model, test_loader, return_predictions=True)
# %%
names_test = np.concatenate([x[1]['name'] for x in test_results])
gt_test = torch.cat([x[1]['stiffness'] for x in test_results], dim=0)
pred_test = torch.cat([x[0]['stiffness'] for x in test_results], dim=0)
results['gt_test'] = gt_test
results['pred_test'] = pred_test
results['names_test'] = names_test
# %%
i = 101
axis_conf = {}
axis_conf['range']=[-0.8, 0.8]
fig = plotting.plotly_tensor_projection(elasticity_func.stiffness_Mandel_to_cart_4(gt_test[i]), title='gt')
fig.update_layout(scene=dict(xaxis=axis_conf, yaxis=axis_conf, zaxis=axis_conf, aspectmode='cube'))
fig.show()
fig = plotting.plotly_tensor_projection(elasticity_func.stiffness_Mandel_to_cart_4(pred_test[i]), title='pred')
fig.update_layout(scene=dict(xaxis=axis_conf, yaxis=axis_conf, zaxis=axis_conf, aspectmode='cube'))
fig.show()
# %%
err_results = {}
for split in ['test']:
    gt = results[f'gt_{split}']
    pred = results[f'pred_{split}']
    gt_eig = torch.linalg.eigvalsh(gt)
    errors = torch.nn.functional.mse_loss(pred, gt, reduction='none').mean(dim=(1,2)).numpy()
    magnitudes = gt.pow(2).mean(dim=(1,2)).numpy()
    err_results[f'err_{split}'] = errors
    err_results[f'anis_{split}'] = gt_eig[:, -1] / gt_eig[:, 0]
    err_results[f'mag_{split}'] = magnitudes
    err_results[f'rel_{split}'] = errors / magnitudes
    err_results[f'names_{split}'] = results[f'names_{split}']
df = pd.DataFrame(err_results).sort_values('rel_test')
# df.to_csv('../results-allaug.csv')
df
# %%
# sns.histplot(df['err_train'], log_scale=True, kde=True, legend=True, alpha=0.2, label='train', stat='density', color='C0')
sns.histplot(df['err_test'], log_scale=True, kde=True, legend=True, alpha=0.2, label='test', stat='density', color='C1')
plt.legend()
plt.xlabel('MSE')
plt.ylabel('Probability density')
plt.xlim(1e-5, 1)
# plt.savefig('../mse_hist_allaug.svg')
plt.show()
# %%
# sns.histplot(df['rel_train'], log_scale=True, bins=20, kde=True, legend=True, alpha=0.2, label='train', stat='density', color='C0')
sns.histplot(df['rel_test'], log_scale=True, bins=20, kde=True, legend=True, alpha=0.2, label='test', stat='density', color='C1')
indices_plot = [100, 4000, 8000]
# for idx in indices_plot:
#     plt.axvline(df['rel_test'].sort_values().iloc[idx], color='C1', linestyle='--', alpha=0.5)
#     plt.axvline(df['rel_train'].sort_values().iloc[idx], color='C0', linestyle='--', alpha=0.5)
plt.legend()
plt.xlabel('Rel MSE')
plt.ylabel('Probability density')
plt.xlim(1e-3, 100)
# plt.savefig('../figs/relmse_hist.svg')
# plt.savefig('mse_hist.png')
plt.show()
# %% Plot examples
for split in ['train', 'test']:
    gt = results[f'gt_{split}']
    pred = results[f'pred_{split}']
    for i in indices_plot:
        # sample +- 20 indices
        i = i + np.random.randint(-20,20)
        df_row = df.sort_values(f'rel_{split}').iloc[i]
        idx = df_row.name
        eigs_gt = np.linalg.eigvalsh(gt[idx])
        eigs_pred = np.linalg.eigvalsh(pred[idx])
        eig_min = min(eigs_gt[0], eigs_pred[0])
        eig_max = max(eigs_gt[-1], eigs_pred[-1])
        plot_gt = elasticity_func.stiffness_Mandel_to_cart_4(gt[idx])
        plot_pred = elasticity_func.stiffness_Mandel_to_cart_4(pred[idx])
        fig = make_subplots(rows=1, cols=2, subplot_titles=['t' for _ in range(2)], specs=[[{'type':'surface'},{'type':'surface'}]])
        fig = plotting.plotly_tensor_projection(plot_gt, title=f'{split}, i: {i}, gt', clim=(eig_min,eig_max), fig=fig, subplot={'index':0, 'ncols':2})
        fig = plotting.plotly_tensor_projection(plot_pred, title=f'{split} i: {i}, pred', clim=(eig_min,eig_max), fig=fig, subplot={'index':1, 'ncols':2})
        fig.update_layout(width=2000, height=1000)
        camera_conf = dict(eye=dict(x=1.4, y=-3.4, z=1.4))
        axis_conf = dict(showbackground=False, range=[-eig_max,eig_max])
        fig.update_layout(scene=dict(
            xaxis=axis_conf, yaxis=axis_conf, zaxis=axis_conf, aspectmode='cube', camera=camera_conf
        ), scene2=dict(
            xaxis=axis_conf, yaxis=axis_conf, zaxis=axis_conf, aspectmode='cube', camera=camera_conf
        ))
        fn = Path('../figs/') / f'{split}_i_{i}_idx_{idx}.png'
        print(f'Saving image to {fn}')
        fig.write_image(fn)
        # fig.show()
        del fig
# %% What did the network learn as the function for strut thickness?
data_list = []
for i, data in enumerate(train_dset):
    data_list.append(data)
    if i==0:
        break
# %%
edge_radii = data.edge_attr
edge_index = data.edge_index
shifts = data.transformed_edge_shifts

vectors, lengths = get_edge_vectors_and_lengths(
    positions=data.pos, edge_index=edge_index, shifts=shifts
)
self=ckpt_model.model
edge_length_embedding = soft_one_hot_linspace(
            lengths.squeeze(-1), start=0, end=0.6, number=self.number_of_edge_basis, basis='gaussian', cutoff=False
        )
edge_radius_embedding = soft_one_hot_linspace(
    edge_radii.squeeze(-1), 0, ckpt_model.model.max_edge_radius, ckpt_model.model.number_of_edge_basis, 'gaussian', False
)
edge_feats = torch.cat(
    (edge_length_embedding, edge_radius_embedding), 
    dim=1
)
# %%
edge_radius_embedding
# %%
plt.plot(edge_length_embedding.detach().numpy().T)
# %%
# %%
mult = np.linspace(0.5,2.0, 16)

for k, data in enumerate(test_dset):
    data_list = []
    for i in range(16):
        data_list.append(data.clone())
        data_list[-1].lattice_constants[0,:3] *= mult[i]

    batch = Batch.from_data_list(data_list)
    
    out = ckpt_model.model(batch)
    mag = out['stiffness'].pow(2).mean(dim=(1,2))
    # mag.backward()
    # print(scaler.grad)
    

    rhobar = 1/mult**2
    plt.plot(mult, mag.detach().numpy())
    # plt.plot(rhobar, mag.detach().numpy())

    if k==8:
        break
plt.xscale('log')
plt.yscale('log')
# plt.xlabel(r'$\bar{\rho}$')
plt.xlabel(r'$L/L_0$')
plt.ylabel(r'$\langle C^2 \rangle$')
plt.show()
# %%
mult = np.linspace(0.7,2.0, 16)

for k, data in enumerate(test_dset):
    data_list = []
    for i in range(16):
        data_list.append(data.clone())
        data_list[-1].edge_attr *= mult[i]

    batch = Batch.from_data_list(data_list)
    
    out = ckpt_model.model(batch)
    mag = out['stiffness'].pow(2).mean(dim=(1,2))
    # mag.backward()
    # print(scaler.grad)
    

    rhobar = 1/mult**2
    plt.plot(mult, mag.detach().numpy())
    # plt.plot(rhobar, mag.detach().numpy())

    if k==8:
        break
plt.xscale('log')
plt.yscale('log')
# plt.xlabel(r'$\bar{\rho}$')
plt.xlabel(r'$r/r_0$')
plt.ylabel(r'$\langle C^2 \rangle$')
plt.show()
# %% Gradient-based optimization
# find best-performing lattice from the test set
idx = results['aggr_test'].argmin()
data = test_dset[idx]
data['stiffness']
# %%
df[df['names_test']=='hex_Z13.1_R2012_p_0.01_5026340585019598058']
# %%
# df.loc[2802]
# test_dset[7571]
plotting.plotly_tensor_projection(elasticity_func.stiffness_Mandel_to_cart_4(gt_test[2802]), title='gt')