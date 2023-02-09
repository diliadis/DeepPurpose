import os
aff = os.sched_getaffinity(0)
print('**********************before import torch******************************'+str(aff))
import torch
print('**********************after import torch******************************'+str(os.sched_getaffinity(0)))
os.sched_setaffinity(0, aff)

from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models
from DeepPurpose.utils import *

import pandas as pd
import wandb
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")
import argparse

def main(run_id, cuda_id, source_wandb_project_name, target_wandb_project_name, general_architecture_version):
        
    wandb_project_entity = 'diliadis'
    
    # load the config of the requested run from wandb
    api = wandb.Api(timeout=19)
    run = api.run(path=wandb_project_entity+'/'+source_wandb_project_name+'/'+run_id)
    best_config = run.config
    best_config['cuda_id'] = cuda_id
    
    split_method = 'A'
    if best_config['validation_setting'] == 'B':
        split_method = 'cold_drug'
    if best_config['validation_setting'] == 'C':
        split_method = 'cold_protein'
    if best_config['validation_setting'] == 'A':
        split_method = 'random'
    
    if best_config['dataset_name'] == 'DAVIS':
        # load and split dataset
        X_drugs, X_targets, y = dataset.load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30) # http://staff.cs.utu.fi/~aatapa/data/DrugTarget/
    else:
        X_drugs, X_targets, y = dataset.load_process_KIBA(path = './data/', binary=False)
        
    drug_encoding, target_encoding = best_config['drug_encoding'], best_config['target_encoding']
    print('Processing the dataset...')

    print('Done! ')
    config = {}
    '''
    config = utils.generate_config(drug_encoding = 'MPNN', 
                            target_encoding = 'CNN', 
                            train_epoch = 100, 
                            LR = 0.0001, 
                            batch_size = 256,
                            hidden_dim_drug = 32,
                            hidden_dim_protein = 32,
                            mpnn_depth = 1,
                            mpnn_hidden_size = 50,
                            cnn_target_filters = [64,32,128],
                            cnn_target_kernels = [8, 16, 4],
                            
                            general_architecture_version = general_architecture_version,
                            cuda_id='0',
                            wandb_project_name = wandb_project_name,
                            wandb_project_entity = wandb_project_entity,
                            use_early_stopping = True,
                            patience = 5,
                            delta = 0.001,
                            metric_to_optimize_early_stopping = 'loss',
                            num_workers=4,
                            experiment_name='best_'+general_architecture_version+'model',
                            )
    '''
    config['parent_wandb_id'] = run.id
    if 
    config['explicit_plus_one_hot_drug_features_mode'] = False
    config['explicit_plus_one_hot_protein_features_mode'] = False
    # updating the dummy config with the dictionary loaded from wandb
    config.update(best_config)
    config['wandb_project_name'] = target_wandb_project_name
    
    train, val, test = utils.data_process(X_drugs, X_targets, y, 
                                drug_encoding, target_encoding, 
                                split_method=split_method,frac=[0.7,0.1,0.2],
                                random_seed = 1,
                                explicit_plus_one_hot_drug_features_mode = config['explicit_plus_one_hot_drug_features_mode'],
     				            explicit_plus_one_hot_protein_features_mode = config['explicit_plus_one_hot_protein_features_mode']
                                )
    
    # initialize the model
    model = models.model_initialize(**config)
    print(str(model.model))
    print(str(model.config))
    # start training, validating, testing
    model.train(train, val, test)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepPurpose DTI example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--id", help="the wandb_id of the config to be trained")
    parser.add_argument("--cuda_id", help="the id of the GPU that will be used for training")
    parser.add_argument("--source_wandb_project_name", help="the project name where the config is stored")
    parser.add_argument("--target_wandb_project_name", help="the project name where the config will is stored")
    parser.add_argument("--general_architecture_name", help="the type of the architecture that is tested [dot_product, mlp]")

    args = parser.parse_args()
    config = vars(args)
    
    main(config['id'], config['cuda_id'], config['source_wandb_project_name'], config['target_wandb_project_name'], config['general_architecture_name'])