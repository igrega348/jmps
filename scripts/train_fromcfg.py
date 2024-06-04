# %%
import os
import sys
from pathlib import Path
par_folder = Path(__file__).absolute().parents[1]
if str(par_folder) not in sys.path:
    sys.path.insert(0, str(par_folder))
import yaml
from pathlib import Path
from datetime import datetime

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelSummary, 
    ModelCheckpoint, 
    EarlyStopping,
    LearningRateMonitor,
    StochasticWeightAveraging,
) 
from pytorch_lightning.utilities.rank_zero import rank_zero_info
# from pytorch_lightning.utilities.seed import seed_everything
from torch_geometric.loader import DataLoader

from gnn import PositiveLiteGNN
from gnn.callbacks import PrintTableMetrics
from train_utils import load_datasets, obtain_errors, aggr_errors, CfgDict, LightningWrappedModel
# %%
def main():
    run_name = os.environ['RUNNAME']
    log_dir = par_folder/'experiments'/f'{run_name}'
    rank_zero_info(log_dir)
    cfg_path = log_dir/f'params-{run_name}.yml'
    cfg = yaml.safe_load(cfg_path.read_text())   
    cfg = CfgDict(cfg)

    assert log_dir.is_dir()

    ############# setup data ##############
    train_dset = load_datasets(parent=cfg.data.train_dset_parent, tag='train', reldens_norm=False)
    valid_dset = load_datasets(parent=cfg.data.val_dset_parent, tag='valid', reldens_norm=False)

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
                               tags=['exp-6'])
    wandb_logger.watch(lightning_model, log="all")
    
    callbacks = [
        ModelSummary(max_depth=3),
        ModelCheckpoint(filename='{epoch}-{step}-{val_loss:.4f}', every_n_epochs=1, monitor='val_loss', save_top_k=1, save_last=True),
        PrintTableMetrics(['epoch','step','loss','val_loss','lr'], every_n_steps=1010*cfg.training.log_every_n_steps),
        LearningRateMonitor(logging_interval='step'),
        # EarlyStopping(monitor='val_loss', patience=50, verbose=True, mode='min', strict=False) 
    ]
    trainer = pl.Trainer(
        accelerator='auto',
        accumulate_grad_batches=cfg.training.accumulate_grad_batches, 
        gradient_clip_val=cfg.training.gradient_clip_val,
        default_root_dir=cfg.log_dir,
        logger=wandb_logger,
        enable_progress_bar=False,
        callbacks=callbacks,
        max_steps=cfg.training.max_steps,
        max_time=cfg.training.max_time,
        val_check_interval=0.1,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )

    ############# save params ##############
    if trainer.is_global_zero:
        params_path = log_dir/f'_params-{run_name}.yml'
        params_path.write_text(yaml.dump(dict(cfg)))

    ############# run training ##############
    ckpt_path = cfg.get('ckpt_path', None)
    if ckpt_path is not None:
        trainer.fit(lightning_model, train_loader, valid_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(lightning_model, train_loader, valid_loader)

    ############# run testing ##############
    rank_zero_info('Testing')
    valid_loader = DataLoader(
        dataset=valid_dset, batch_size=cfg.training.valid_batch_size,
        shuffle=False,
    )
    # test_dset = load_datasets(parent=cfg.data.dset_parent, tag='test', reldens_norm=False)
    # test_loader = DataLoader(
        # dataset=test_dset, batch_size=cfg.training.valid_batch_size, 
        # shuffle=False, 
    # )
    valid_results = trainer.predict(lightning_model, valid_loader, return_predictions=True, ckpt_path='best')
    # test_results = trainer.predict(lightning_model, test_loader, return_predictions=True, ckpt_path='best')
    df_errors = obtain_errors(valid_results, 'valid')
    # df_errors = pd.concat([obtain_errors(valid_results, 'valid'), obtain_errors(test_results, 'test')], axis=0, ignore_index=True)
    eval_params = aggr_errors(df_errors)
    pd.Series(eval_params, name=run_name).to_csv(log_dir/f'aggr_results-{run_name}-step={trainer.global_step}.csv')
    rank_zero_info(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__=='__main__':
    main()
