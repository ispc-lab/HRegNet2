import os
import torch
import numpy as np

from data.dataloader import make_data_loader
from models.trainer import RegTrainer
from models.utils import set_seed
from config import get_config

def main(config):

    train_loader, val_loader, _ = make_data_loader(config)

    trainer = RegTrainer(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader
                )
    
    trainer.logger.info(config)
    trainer.train()

if __name__ == '__main__':

    config = get_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    set_seed(config.seed)

    main(config)