# %%
import os
import sys
from pathlib import Path
par_folder = Path(__file__).absolute().parents[1]
if str(par_folder) not in sys.path:
    sys.path.insert(0, str(par_folder))
from argparse import Namespace
import json
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
from train_utils import load_datasets, obtain_errors, aggr_errors
# %%
class LightningWrappedModel(pl.LightningModule):
    _time_metrics = {}
    
    def __init__(self, model: torch.nn.Module, params: Namespace, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(params, dict):
            params = Namespace(**params)
        self.params = params
        self.model = model(params)
       
        self.save_hyperparameters(params)

    def configure_optimizers(self):
        params = self.params
        optim = torch.optim.AdamW(params=self.model.parameters(), lr=params.lr, 
            betas=(params.beta1,0.999), eps=params.epsilon,
            amsgrad=params.amsgrad, weight_decay=params.weight_decay,)
        return optim

    def training_step(self, batch, batch_idx):
        
        output = self.model(batch)

        true_stiffness = batch['stiffness']
        pred_stiffness = output['stiffness']

        target = true_stiffness # [N, 6, 6]
        predicted = pred_stiffness # [N, 6, 6]
        mean_stiffness = target.pow(2).mean(dim=(1,2)) # [N]
        stiffness_loss = torch.nn.functional.mse_loss(predicted, target, reduction='none').mean(dim=(1,2)) # [N]

        stiffness_loss_mean = stiffness_loss.mean()

        loss = stiffness_loss
        loss = 100*(loss / mean_stiffness).mean() # [1]
    
        self.log('loss', loss, batch_size=batch.num_graphs, logger=True)
        self.log('stiffness_loss', stiffness_loss_mean, batch_size=batch.num_graphs, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        
        output = self.model(batch)
        true_stiffness = batch['stiffness']
        pred_stiffness = output['stiffness']

        target = true_stiffness
        predicted = pred_stiffness
        stiffness_loss = torch.nn.functional.mse_loss(predicted, target)
  
        loss = stiffness_loss
    
        self.log('val_loss', loss, batch_size=batch.num_graphs, logger=True, prog_bar=True, sync_dist=True)
        return loss
        
    def predict_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0) -> Tuple:
        """Returns (prediction, true)"""
        return self.model(batch), batch
    
    def on_train_epoch_start(self) -> None:
        self._time_metrics['_last_step'] = self.trainer.global_step
        self._time_metrics['_last_time'] = time.time()

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        step = self.trainer.global_step
        steps_done = step - self._time_metrics['_last_step']
        time_now = time.time()
        time_taken = time_now - self._time_metrics['_last_time']
        steps_per_sec = steps_done / time_taken
        self._time_metrics['_last_step'] = step
        self._time_metrics['_last_time'] = time_now
        self.log('steps_per_time', steps_per_sec, prog_bar=False, logger=True)
        # check if loss is nan
        loss = outputs['loss']
        if torch.isnan(loss):
            self.trainer.should_stop = True
            rank_zero_info('Loss is NaN. Stopping training')

def main():
    # df = pd.read_csv('./mace-hparams-216.csv', index_col=0)
    # num_hp_trial = int(os.environ['NUM_HP_TRIAL'])
    num_hp_trial = 0

    desc = "Exp-0. First trial"
    rank_zero_info(desc)
    seed_everything(0, workers=True)

    params = Namespace(
        # network
        lmax=4,
        hidden_irreps='+'.join([f'16x{i}e' if i%2==0 else f'16x{i}o' for i in range(0,5)]),
        readout_irreps='+'.join([f'8x{i}e' if i%2==0 else f'8x{i}o' for i in range(0,5)]),
        num_edge_bases=10,
        max_edge_radius=0.018,
        interaction_reduction='sum',
        agg_norm_const=4.0,
        inter_MLP_dim=64,
        inter_MLP_layers=3,
        correlation=3,
        global_reduction='mean',
        message_passes=2,
        positive_function='matrix_power_2',
        # dataset
        dset_parent=str(par_folder/'dset'),
        # training
        # num_hp_trial=num_hp_trial,
        batch_size=16,
        valid_batch_size=64,
        log_every_n_steps=100,
        optimizer='adamw',
        lr=1e-3, 
        amsgrad=True,
        weight_decay=1e-8,
        beta1=0.9,
        epsilon=1e-8,
        num_workers=4,
    )
    params.desc = desc

    # run_name = os.environ['SLURM_JOB_ID']
    run_name = '0'
    log_dir = par_folder/f'experiments/{run_name}'
    while log_dir.is_dir():
        run_name = str(int(run_name)+1)
        log_dir = par_folder/f'experiments/{run_name}'
    log_dir.mkdir(parents=True)
    rank_zero_info(log_dir)
    params.log_dir = str(log_dir)

    ############# setup data ##############
    train_dset = load_datasets(parent=params.dset_parent, tag='test', reldens_norm=False)
    valid_dset = load_datasets(parent=params.dset_parent, tag='test', reldens_norm=False)

    # randomize the order of the dataset into loader
    train_loader = DataLoader(
        dataset=train_dset, 
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
    )

    valid_loader = DataLoader(
        dataset=valid_dset,
        batch_size=params.valid_batch_size,
        shuffle=False,
        num_workers=params.num_workers,
    )

    ############# setup model ##############
    lightning_model = LightningWrappedModel(PositiveLiteGNN, params)

    ############# setup trainer ##############
    wandb_logger = WandbLogger(project="JMPS", entity="ivan-grega", save_dir=params.log_dir, 
                               tags=['exp-0'])
    callbacks = [
        ModelSummary(max_depth=3),
        ModelCheckpoint(filename='{epoch}-{step}-{val_loss:.3f}', every_n_epochs=1, monitor='val_loss', save_top_k=1),
        PrintTableMetrics(['epoch','step','loss','val_loss'], every_n_steps=100),
        EarlyStopping(monitor='val_loss', patience=50, verbose=True, mode='min', strict=False) 
    ]
    # max_time = '00:01:27:00' if os.environ['SLURM_JOB_PARTITION']=='ampere' else '00:05:45:00'
    max_time = '00:01:00:00'
    trainer = pl.Trainer(
        accelerator='auto',
        accumulate_grad_batches=4, # effective batch size 256
        gradient_clip_val=10.0,
        default_root_dir=params.log_dir,
        logger=wandb_logger,
        enable_progress_bar=False,
        overfit_batches=0.1,
        callbacks=callbacks,
        max_steps=50000,
        max_time=max_time,
        # val_check_interval=1000,
        log_every_n_steps=params.log_every_n_steps,
        check_val_every_n_epoch=1,
        # limit_val_batches=0.1
    )

    ############# save params ##############
    if trainer.is_global_zero:
        params_path = log_dir/f'params-{num_hp_trial}.json'
        params_path.write_text(json.dumps(vars(params), indent=2))

    ############# run training ##############
    trainer.fit(lightning_model, train_loader, valid_loader)

    ############# run testing ##############
    rank_zero_info('Testing')
    train_dset = load_datasets(parent=params.dset_parent, tag='test', reldens_norm=False)
    train_loader = DataLoader(
        dataset=train_dset, batch_size=params.valid_batch_size, 
        shuffle=False,)
    valid_loader = DataLoader(
        dataset=valid_dset, batch_size=params.valid_batch_size,
        shuffle=False,
    )
    test_dset = load_datasets(parent=params.dset_parent, tag='test', reldens_norm=False)
    test_loader = DataLoader(
        dataset=test_dset, batch_size=params.valid_batch_size, 
        shuffle=False, 
    )
    train_results = trainer.predict(lightning_model, train_loader, return_predictions=True, ckpt_path='best')
    valid_results = trainer.predict(lightning_model, valid_loader, return_predictions=True, ckpt_path='best')
    test_results = trainer.predict(lightning_model, test_loader, return_predictions=True, ckpt_path='best')
    df_errors = pd.concat([obtain_errors(train_results, 'train'), obtain_errors(valid_results, 'valid'), obtain_errors(test_results, 'test')], axis=0, ignore_index=True)
    eval_params = aggr_errors(df_errors)
    pd.Series(eval_params, name=num_hp_trial).to_csv(log_dir/f'aggr_results-{num_hp_trial}-step={trainer.global_step}.csv')
 
    if eval_params['loss_test']>10:
        for f in log_dir.glob('**/epoch*.ckpt'):
            rank_zero_info(f'Test loss: {eval_params["loss_test"]}. Removing checkpoint {f}')
            f.unlink()

if __name__=='__main__':
    main()
