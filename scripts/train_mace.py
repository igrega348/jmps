# %%
import os
import sys
from pathlib import Path
par_folder = Path(__file__).absolute().parents[1]
if str(par_folder) not in sys.path:
    sys.path.insert(0, str(par_folder))
from argparse import Namespace
import json
import yaml
import time
from typing import Any, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelSummary, 
    ModelCheckpoint, 
    TQDMProgressBar,
    EarlyStopping
) 
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.utilities.seed import seed_everything
from torch_geometric.loader import DataLoader
from e3nn import o3

from gnn import PositiveLiteGNN
from gnn import GLAMM_Dataset
from gnn.callbacks import PrintTableMetrics
from train_utils import load_datasets, obtain_errors, aggr_errors, CfgDict, LightningWrappedModel
# %%
def main():
    # df = pd.read_csv('./mace-hparams-216.csv', index_col=0)
    # num_hp_trial = int(os.environ['NUM_HP_TRIAL'])
    num_hp_trial = 0

    desc = "Exp-2. Run smaller dataset, modify model to L_a, L_r"
    rank_zero_info(desc)
    seed_everything(0, workers=True)

    cfg = {
        'desc':desc,
        'model':{
            'hidden_irreps':'16x0e+16x1o+16x2e+16x3o+16x4e',
            'readout_irreps':'8x0e+8x1o+8x2e+8x3o+8x4e',
            'num_edge_bases':16,
            'max_edge_L_a': 1.5,
            'max_edge_r_L': 0.1,
            'lmax':4,
            'message_passes':2,
            'agg_norm_const':3.0,
            'interaction_reduction':'sum',
            'correlation':3,
            'inter_MLP_dim':64,
            'inter_MLP_layers':3,
            'global_reduction':'mean',
            'positive_function':'matrix_power_2',
        },
        'data':{
            'dset_parent':str(par_folder/'dset'),
        },
        'training':{
            'batch_size':16,
            'valid_batch_size':64,
            'log_every_n_steps':100,
            'optimizer':'adamw',
            'lr':1e-3, 
            'amsgrad':True,
            'weight_decay':1e-8,
            'beta1':0.9,
            'epsilon':1e-8,
            'num_workers':4,
        }
    }
    cfg = CfgDict(cfg)

    # run_name = os.environ['SLURM_JOB_ID']
    run_name = '9'
    log_dir = par_folder/f'experiments/{run_name}'
    while log_dir.is_dir():
        run_name = str(int(run_name)+1)
        log_dir = par_folder/f'experiments/{run_name}'
    log_dir.mkdir(parents=True)
    rank_zero_info(log_dir)
    cfg.log_dir = str(log_dir)

    ############# setup data ##############
    train_dset = load_datasets(parent=cfg.data.dset_parent, tag='train', reldens_norm=False)
    valid_dset = load_datasets(parent=cfg.data.dset_parent, tag='valid', reldens_norm=False)

    # randomize the order of the dataset into loader
    train_loader = DataLoader(
        dataset=train_dset, 
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )

    valid_loader = DataLoader(
        dataset=valid_dset,
        batch_size=cfg.training.valid_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    ############# setup model ##############
    lightning_model = LightningWrappedModel(PositiveLiteGNN, cfg)

    ############# setup trainer ##############
    wandb_logger = WandbLogger(project="JMPS", entity="ivan-grega", save_dir=cfg.log_dir, 
                               tags=['exp-2'])
    callbacks = [
        ModelSummary(max_depth=3),
        ModelCheckpoint(filename='{epoch}-{step}-{val_loss:.3f}', every_n_epochs=1, monitor='val_loss', save_top_k=1),
        PrintTableMetrics(['epoch','step','loss','val_loss'], every_n_steps=1000),
        EarlyStopping(monitor='val_loss', patience=50, verbose=True, mode='min', strict=False) 
    ]
    trainer = pl.Trainer(
        accelerator='auto',
        accumulate_grad_batches=4, # increase effective batch size
        gradient_clip_val=10.0,
        default_root_dir=cfg.log_dir,
        logger=wandb_logger,
        enable_progress_bar=False,
        # overfit_batches=1,
        callbacks=callbacks,
        max_steps=100000,
        # val_check_interval=1000,
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch=1,
        # limit_val_batches=0.1
    )

    ############# save params ##############
    if trainer.is_global_zero:
        params_path = log_dir/f'params-{num_hp_trial}.yml'
        params_path.write_text(yaml.dump(dict(cfg)))

    ############# run training ##############
    trainer.fit(lightning_model, train_loader, valid_loader, ckpt_path=par_folder/f'experiments/8/JMPS/1wjinivo/checkpoints/epoch=2-step=35409-val_loss=2830841.750.ckpt')

    ############# run testing ##############
    rank_zero_info('Testing')
    valid_loader = DataLoader(
        dataset=valid_dset, batch_size=cfg.training.valid_batch_size,
        shuffle=False,
    )
    test_dset = load_datasets(parent=cfg.data.dset_parent, tag='test', reldens_norm=False)
    test_loader = DataLoader(
        dataset=test_dset, batch_size=cfg.training.valid_batch_size, 
        shuffle=False, 
    )
    valid_results = trainer.predict(lightning_model, valid_loader, return_predictions=True, ckpt_path='best')
    test_results = trainer.predict(lightning_model, test_loader, return_predictions=True, ckpt_path='best')
    df_errors = pd.concat([obtain_errors(valid_results, 'valid'), obtain_errors(test_results, 'test')], axis=0, ignore_index=True)
    eval_params = aggr_errors(df_errors)
    pd.Series(eval_params, name=num_hp_trial).to_csv(log_dir/f'aggr_results-{num_hp_trial}-step={trainer.global_step}.csv')

if __name__=='__main__':
    main()
