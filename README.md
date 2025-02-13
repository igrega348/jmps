# Graph neural networks for strut-based architected solids

Link to paper:

https://doi.org/10.1016/j.jmps.2024.105966

![1-s2 0-S0022509624004320-gr8_lrg](https://github.com/user-attachments/assets/4baacdaa-a8f5-4b8a-8102-6b593f1bc863)


## Code structure
```
├── lattices: submodule for lattice processing, elasticity and plotting functions
├── gnn: ML modules
    ├── ...
├── scripts
    ├── train_utils.py: utilities for training
    ├── train_fromcfg.py: training script for the main model
    ├── load_datasets.py: script that can be used to assemble train/val/test datasets
```

## Usage
Set up environment using requirement.txt or environment.yml file.

Try the scripts in `scripts` folder.
You can run the code using the following:
```
cd scripts
python train_fromcfg.py --cfg-path params-0.yml
```
This will use a small sample dataset of 1500 training and 500 validation lattices.

Data to assemble full datasets is available for download from https://doi.org/10.17863/CAM.115813
