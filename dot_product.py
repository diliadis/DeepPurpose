import os
aff = os.sched_getaffinity(0)
print('**********************before import torch******************************'+str(aff))
import torch
print('**********************after import torch******************************'+str(os.sched_getaffinity(0)))
os.sched_setaffinity(0, aff)

from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models
import warnings
import itertools
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import sys
warnings.filterwarnings("ignore")
import random
import argparse
import json


def main(general_architecture_version, val_setting, cuda_id, num_workers, dataset_name, performance_threshold=1.0, wandb_project_name='test', wandb_dir='/data/gent/vo/000/gvo00048/vsc43483', config_file_name=None):
    
    configs = [
    {   "cls_hidden_dims": [
        1024,
        1024,
        512
        ],
        "cnn_target_filters": [
        32,
        64,
        128
        ],
        "cnn_target_kernels": [
        4,
        8,
        16
        ],
        "LR": 0.001,
        "hidden_dim_drug": 128,
        "hidden_dim_protein": 128,
        "mpnn_depth": 3,
        "validation_setting": 'B'
    },
    {   "cls_hidden_dims": [
        1024,
        1024,
        512
        ],
        "cnn_target_filters": [
        32,
        64,
        128
        ],
        "cnn_target_kernels": [
        4,
        8,
        16
        ],
        "LR": 0.001,
        "hidden_dim_drug": 256,
        "hidden_dim_protein": 256,
        "mpnn_depth": 3,
        "validation_setting": 'B'
    }, 
    {   "cls_hidden_dims": [
        1024,
        1024,
        512
        ],
        "cnn_target_filters": [
        32,
        64,
        64
        ],
        "cnn_target_kernels": [
        4,
        8,
        16
        ],
        "LR": 0.001,
        "hidden_dim_drug": 128,
        "hidden_dim_protein": 128,
        "mpnn_depth": 3,
        "validation_setting": 'B'
    },  
    {   "cls_hidden_dims": [
        512,
        512,
        512
        ],
        "cnn_target_filters": [
        32,
        64,
        64
        ],
        "cnn_target_kernels": [
        4,
        8,
        8
        ],
        "LR": 0.001,
        "hidden_dim_drug": 512,
        "hidden_dim_protein": 512,
        "mpnn_depth": 3,
        "validation_setting": 'B'
    },  
    {   "cls_hidden_dims": [
        1024,
        1024,
        512
        ],
        "cnn_target_filters": [
        32,
        64,
        64
        ],
        "cnn_target_kernels": [
        4,
        8,
        16
        ],
        "LR": 0.001,
        "hidden_dim_drug": 128,
        "hidden_dim_protein": 128,
        "mpnn_depth": 3,
        "validation_setting": 'B'
    },     
    {   "cls_hidden_dims": [
        1024,
        512,
        512
        ],
        "cnn_target_filters": [
        32,
        64,
        128
        ],
        "cnn_target_kernels": [
        4,
        8,
        8
        ],
        "LR": 0.001,
        "hidden_dim_drug": 256,
        "hidden_dim_protein": 256,
        "mpnn_depth": 3,
        "validation_setting": 'B'
    },     
    {   "cls_hidden_dims": [
        1024,
        1024,
        512
        ],
        "cnn_target_filters": [
        32,
        64,
        128
        ],
        "cnn_target_kernels": [
        4,
        8,
        16
        ],
        "LR": 0.001,
        "hidden_dim_drug": 256,
        "hidden_dim_protein": 256,
        "mpnn_depth": 3,
        "validation_setting": 'C'
    },
    {   "cls_hidden_dims": [
        1024,
        1024,
        256
        ],
        "cnn_target_filters": [
        32,
        64,
        128
        ],
        "cnn_target_kernels": [
        4,
        8,
        8
        ],
        "LR": 0.001,
        "hidden_dim_drug": 256,
        "hidden_dim_protein": 256,
        "mpnn_depth": 3,
        "validation_setting": 'C'
    }, 
    {   "cls_hidden_dims": [
        1024,
        1024,
        256
        ],
        "cnn_target_filters": [
        32,
        64,
        64
        ],
        "cnn_target_kernels": [
        4,
        8,
        16
        ],
        "LR": 0.001,
        "hidden_dim_drug": 512,
        "hidden_dim_protein": 512,
        "mpnn_depth": 3,
        "validation_setting": 'C'
    },  
    {   "cls_hidden_dims": [
        1024,
        512,
        512
        ],
        "cnn_target_filters": [
        32,
        64,
        128
        ],
        "cnn_target_kernels": [
        4,
        8,
        8
        ],
        "LR": 0.001,
        "hidden_dim_drug": 256,
        "hidden_dim_protein": 256,
        "mpnn_depth": 3,
        "validation_setting": 'C'
    },  
    {   "cls_hidden_dims": [
        1024,
        1024,
        512
        ],
        "cnn_target_filters": [
        32,
        64,
        64
        ],
        "cnn_target_kernels": [
        4,
        8,
        8
        ],
        "LR": 0.001,
        "hidden_dim_drug": 128,
        "hidden_dim_protein": 128,
        "mpnn_depth": 3,
        "validation_setting": 'C'
    },
    {   "cls_hidden_dims": [
        1024,
        256,
        256
        ],
        "cnn_target_filters": [
        32,
        64,
        64
        ],
        "cnn_target_kernels": [
        4,
        8,
        8
        ],
        "LR": 0.001,
        "hidden_dim_drug": 256,
        "hidden_dim_protein": 256,
        "mpnn_depth": 3,
        "validation_setting": 'C'
    }        
    ]
    
    
    for temp_config in configs:
        
        val_setting = temp_config['validation_setting']
    
        split_method = 'random'
        if str(val_setting) == 'B':
            split_method = 'cold_drug'
        elif str(val_setting) == 'C':
            split_method = 'cold_protein'
        elif str(val_setting) == 'A':
            split_method = 'random'
            
        wandb_project_name = wandb_project_name
        wandb_project_entity = 'diliadis'
        general_architecture_version = general_architecture_version
        
        if dataset_name.lower() == 'davis':
            X_drugs, X_targets, y = dataset.load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30)
        elif dataset_name.lower() == 'kiba':
            X_drugs, X_targets, y = dataset.load_process_KIBA(path = './data/', binary=False)
        elif dataset_name.lower() == 'bindingdb':
            X_drugs, X_targets, y = dataset.process_BindingDB(path = '/data/gent/vo/000/gvo00048/vsc43483/BindingDB_All.tsv', y='Kd', binary=False, convert_to_log=True)
        else:
            raise AttributeError('invalid dataset name passed.')
        
        drug_encoding, target_encoding = 'MPNN', 'CNN'
        print('Processing the dataset...')
        train, val, test = utils.data_process(X_drugs, X_targets, y,
                                    drug_encoding, target_encoding, 
                                    split_method=split_method,frac=[0.7,0.1,0.2],
                                    random_seed = 1)
        print('Done! ')
        
        
        config = utils.generate_config(drug_encoding = drug_encoding, 
                                target_encoding = target_encoding, 
                                # cls_hidden_dims = [1024,1024,512], 
                                train_epoch = 100, 
                                LR = 0.001, 
                                batch_size = 256,
                                hidden_dim_drug = 128,
                                hidden_dim_protein = 128,
                                mpnn_hidden_size = 128,
                                mpnn_depth = 3,
                                cnn_target_filters = [32,64,96],
                                cnn_target_kernels = [4,8,12],
                                cls_hidden_dims = [1024,1024,512], 
                                general_architecture_version = general_architecture_version,
                                cuda_id=str(cuda_id),
                                wandb_project_name = wandb_project_name,
                                wandb_project_entity = wandb_project_entity,
                                wandb_dir = wandb_dir,
                                use_early_stopping = True,
                                patience = 100,
                                delta = 0.001,
                                metric_to_optimize_early_stopping = 'loss',
                                num_workers=int(num_workers),
                                performance_threshold = {'metric_name':'MSE', 'value': performance_threshold, 'direction': 'min', 'max_epochs_allowed': 30},
                                validation_setting=val_setting,
                                dataset_name = dataset_name.upper()
                                )
        
        # updating config with the json file provided by the user
        if config_file_name is not None:
            config.update(temp_config)
        
        model = models.model_initialize(**config)
        print(str(model.model))
        print(str(model.config))
        model.train(train, val, test)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepPurpose DTI example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--general_architecture_version", help="variant of dual-encoder architecture")
    parser.add_argument("--val_setting", help="the validation setting that will be used to split the data")
    parser.add_argument("--cuda_id", help="the id of the GPU that will be used for training")
    parser.add_argument("--num_workers", help="the number of workers that will be used by the dataloaders")
    parser.add_argument("--dataset_name", help="the name of the dataset that will be used. (DAVIS and KIBA are the current valid options)")
    parser.add_argument("--performance_threshold", help="performance threshold checked before epoch 30")
    parser.add_argument("--wandb_project_name", help="performance threshold checked before epoch 30")
    parser.add_argument("--config_file_name", help="name of the json file with the config to be tested.")
    args = parser.parse_args()
    config = vars(args)
    
    main(config['general_architecture_version'], config['val_setting'], config['cuda_id'], config['num_workers'], config['dataset_name'], wandb_project_name=config['wandb_project_name'], performance_threshold=float(config['performance_threshold']), config_file_name=config['config_file_name'])
    