import logging
import time
from collections import OrderedDict
from typing import Any
import os
import shutil

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from catalyst import dl, utils
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim

from catalyst.engines.torch import GPUEngine


from __init__  import get_architecture, test_model
from data import load_data, FeatureSetup, NUM_CLASSES

from train import *


@hydra.main(version_base=None, config_path="configs", config_name="default")

##NEXT ALL CONV 01 NAIVE OVERSAMPLE
def trainer(cfg: DictConfig) -> None: 
    GEMS_9 = ['Wonder', 'Transcendence', 'Nostalgia', 'Tenderness', 'Peacfulness', 'Joy', 'Power', 'Tension', 'Sadness']

    print(OmegaConf.to_yaml(cfg))
    for i in GEMS_9:

        run_training(cfg, GEMS_9 = i)


if __name__ == "__main__":

    trainer()
