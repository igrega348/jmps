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
from train_utils import load_datasets, obtain_errors, aggr_errors, CfgDict
# %%

def main():
    ############# setup data ##############
    dset_parent = par_folder/'aug5'
    train_dset = load_datasets(parent=dset_parent, tag='train', reldens_norm=False)
    valid_dset = load_datasets(parent=dset_parent, tag='valid', reldens_norm=False)
    test_dset = load_datasets(parent=dset_parent, tag='test', reldens_norm=False)

if __name__=='__main__':
    main()
