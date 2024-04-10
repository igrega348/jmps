# %%
import os
import sys
from pathlib import Path
folder = Path(__file__).parents[1]
if str(folder) not in sys.path:
    sys.path.insert(0, str(folder))
from typing import Any, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from e3nn import o3

from gnn import GLAMM_Dataset
from lattices.lattices import elasticity_func
# %%
class CfgDict(dict):
    """dot.notation access to dictionary attributes"""
    # based on https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    # but allowing for implicit nesting
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getattr__(self, key):
        val = self.get(key)
        if isinstance(val, dict):
            return CfgDict(val)
        return val
    
class RotateLat:
    def __init__(self, rotate=True):
        self.rotate = rotate

    def __call__(self, lat: Data, Q: Optional[Tensor] = None):
        if self.rotate:
            if Q is None:
                Q = o3.rand_matrix()
            C = torch.einsum('...ijkl,ai,bj,ck,dl->...abcd', lat.stiffness, Q, Q, Q, Q)
            S = torch.einsum('...ijkl,ai,bj,ck,dl->...abcd', lat.compliance, Q, Q, Q, Q)
            pos = torch.einsum('ij,...j->...i', Q, lat.positions)
            shifts = torch.einsum('ij,...j->...i', Q, lat.shifts)
        else:
            assert Q is None, 'Q should be None if instance initialized with rotate=False'
            C = lat.stiffness
            S = lat.compliance
            pos = lat.positions
            shifts = lat.shifts
            
        C_mand = elasticity_func.stiffness_cart_4_to_Mandel(C)
        S_mand = elasticity_func.stiffness_cart_4_to_Mandel(S)
        transformed = Data(
            node_attrs=lat.node_attrs,
            edge_attr=lat.edge_attr,
            edge_index=lat.edge_index,
            positions = pos,
            shifts = shifts,
            rel_dens=lat.rel_dens,
            stiffness=C_mand,
            compliance=S_mand,
            name = lat.name
        )
        return transformed


def obtain_errors(results, tag: str):
    # multiply by 10 because I reduced the multiplier in load_datasets
    target = 10*torch.cat([x[1]['stiffness'] for x in results]) # [num_graphs, 6, 6]
    prediction = 10*torch.cat([x[0]['stiffness'] for x in results]) # [num_graphs, 6, 6]
    names = np.concatenate([x[1]['name'] for x in results])
    rel_dens = torch.cat([x[1]['rel_dens'] for x in results]).numpy()
    directions = torch.randn(250, 3)
    directions = directions / directions.norm(dim=1, keepdim=True)
    mse_loss = torch.nn.functional.mse_loss(prediction, target, reduction='none').mean(dim=(1,2)).numpy()
    loss = torch.nn.functional.l1_loss(prediction, target, reduction='none').mean(dim=(1,2)).numpy()
    target_4 = elasticity_func.stiffness_Mandel_to_cart_4(target)
    prediction_4 = elasticity_func.stiffness_Mandel_to_cart_4(prediction)
    c = torch.einsum('...ijkl,pi,pj,pk,pl->...p', target_4, directions, directions, directions, directions).numpy()
    c_pred = torch.einsum('...ijkl,pi,pj,pk,pl->...p', prediction_4, directions, directions, directions, directions).numpy()
    dir_loss = np.abs(c - c_pred).mean(axis=1)
    mean_stiffness = c.mean(axis=1)
    # eigenvalues
    target_eig = [x for x in torch.linalg.eigvalsh(target).numpy()]
    try:
        predicted_eig = [x for x in torch.linalg.eigvalsh(prediction).numpy()]
    except:
        predicted_eig = np.nan
    return pd.DataFrame({'name':names, 'rel_dens':rel_dens, 'mean_stiffness':mean_stiffness, 'loss':loss, 'mseloss':mse_loss, 'dir_loss':dir_loss, 'tag':tag, 'target_eig':target_eig, 'predicted_eig':predicted_eig})

def aggr_errors(df_errors):
    params = {}
    if df_errors['loss'].isna().sum() > 0:
        return params
    df_errors['rel_loss'] = df_errors['loss'] / df_errors['mean_stiffness']
    df_errors['mse_rel_loss'] = df_errors['mseloss'].map(np.sqrt) / df_errors['mean_stiffness']
    df_errors['rel_dir_loss'] = df_errors['dir_loss'] / df_errors['mean_stiffness']
    df_errors['min_pred_eig'] = df_errors['predicted_eig'].map(np.min)
    df_errors['min_target_eig'] = df_errors['target_eig'].map(np.min)
    # eigenvalue loss will be calculated as loss between the two volumes calculated from eigenvalues
    predicted_volumes = df_errors['predicted_eig'].map(np.prod)
    target_volumes = df_errors['target_eig'].map(np.prod)
    df_errors['eig_loss'] = np.abs(predicted_volumes - target_volumes)
    df_errors['rel_eig_loss'] = df_errors['eig_loss'] / target_volumes

    means = df_errors[['tag','loss','rel_loss','mseloss','mse_rel_loss','dir_loss','rel_dir_loss','eig_loss','rel_eig_loss']].groupby(['tag']).mean()
    for tag in means.index:
        for col in means.columns:
            params[f'{col}_{tag}'] = means.loc[tag, col]

    mins = df_errors[['tag','min_pred_eig','min_target_eig']].groupby(['tag']).min()
    for tag in mins.index:
        for col in mins.columns:
            params[f'{col}_{tag}'] = mins.loc[tag, col]
    
    prop_eig_negative = (df_errors['min_pred_eig']<0).groupby(df_errors['tag']).sum() / df_errors['tag'].value_counts()
    for tag in prop_eig_negative.index:
        params[f'prop_eig_negative_{tag}'] = prop_eig_negative.loc[tag]
    
    return params

def load_datasets(tag: str, parent: str = '../../datasets', reldens_norm: bool = False, rotate: bool = True):
    if tag == 'test':
        dset_file = 'test.lat'
        processed_fname = 'test.pt'
    elif tag == 'train':
        dset_file = 'train.lat'
        processed_fname = 'train.pt'
    elif tag == 'valid':
        dset_file = 'val.lat'
        processed_fname = 'validation.pt'
    root = parent
    rank_zero_info(f'Loading dataset {tag} from {root}')
    dset = GLAMM_Dataset(
        root=root,
        catalogue_path=os.path.join(root, 'raw', dset_file),
        # transform=RotateLat(rotate=rotate),
        dset_fname=processed_fname,
        n_reldens=3,
        choose_reldens='all',
        graph_ft_format='Mandel', # no rotation and Mandel format
    )
    rank_zero_info(dset)

    # scaling and normalization
    if reldens_norm:
        normalization_factor = 10 / dset.data.rel_dens.view(-1,1,1,1,1)
    else:
        normalization_factor = 10000 # increased again because we're targeting relative densities on the order of 0.001

    dset.data.stiffness = (dset.data.stiffness * normalization_factor).float()
    dset.data.compliance = (dset.data.compliance / normalization_factor).float()

    return dset