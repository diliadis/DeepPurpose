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
import threading
import os

def main(cuda_id, num_workers, source_wandb_project_name, target_wandb_project_name, source_validation_setting, target_validation_setting, dataset_name, wandb_dir='/data/gent/vo/000/gvo00048/vsc43483'):
    
    run_ids_list = ['2013kul5',
                    '1ww0jpmc'
                    '1rv5fc81',
                    '1qbukn8t',
                    '3e3nahsv',
                    'ih7mtdzt',
                    '3zzvoi71',
                    '1x65f2ak',
                    '1vliy8wy',
                    '3j3z0wva',
                    '1yxpibtt'
                    ]
    
    api = wandb.Api()
    entity, source_project = 'diliadis', source_wandb_project_name  # set to your entity and project 
    if len(run_ids_list) == 0:
        source_runs = api.runs(entity + "/" + source_project, filters={"config.validation_setting": source_validation_setting, "config.dataset_name": dataset_name}, order="+created_at")
    else:
        source_runs = [ api.run(entity + "/" + source_project + "/" + run_id) for run_id in run_ids_list]
    # source_runs = api.runs(entity + "/" + source_project)

    print(str(len(source_runs))+' runs loaded')
    for run in source_runs:
        if run.state != 'crashed':
            source_config = {k: v for k, v in run.config.items() if not k.startswith('_')}

            if target_validation_setting == 'B':
                split_method = 'cold_drug'
            if target_validation_setting == 'C':
                split_method = 'cold_protein'
            if target_validation_setting == 'A':
                split_method = 'random'
            source_config['validation_setting'] = target_validation_setting
                            
            if dataset_name.lower() == 'davis':
                X_drugs, X_targets, y = dataset.load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30)
            elif dataset_name.lower() == 'kiba':
                X_drugs, X_targets, y = dataset.load_process_KIBA(path = './data/', binary=False)
            else:
                raise AttributeError('invalid dataset name passed.')

            drug_encoding, target_encoding = source_config['drug_encoding'], source_config['target_encoding']
            print('Processing the dataset...')
            train, val, test = utils.data_process(X_drugs, X_targets, y,
                                        drug_encoding, target_encoding, 
                                        split_method=split_method,frac=[0.7,0.1,0.2],
                                        random_seed = 1)
            print('Done! ')
        
            config = utils.generate_config(drug_encoding = drug_encoding, 
                                    target_encoding = target_encoding,  
                                    cuda_id=str(cuda_id),
                                    wandb_project_name = target_wandb_project_name,
                                    wandb_dir = wandb_dir,
                                    num_workers=int(num_workers),
                                    parent_wandb_id = run.id,
                                    dataset_name = dataset_name
            )
            
            config.update({k: v for k,v in source_config.items() if k not in ['device', 'cuda_id', 'wandb_dir', 'num_workers', 'dataset_name', 'wandb_project_name']})
            
            model = models.model_initialize(**config)
            print(str(model.model))
            print(str(model.config))
            model.train(train, val, test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepPurpose DTI example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda_id", help="the id of the GPU that will be used for training")
    parser.add_argument("--num_workers", help="the number of workers that will be used by the dataloaders")
    parser.add_argument("--source_wandb_project_name", help="name of the source wandb project from which I will select configurations to extend")
    parser.add_argument("--target_wandb_project_name", help="name of the target wandb project where I will save the extended configurations")
    parser.add_argument("--source_validation_setting", help="validation_setting")
    parser.add_argument("--target_validation_setting", help="validation_setting")

    parser.add_argument("--dataset_name", help="dataset_name")

    args = parser.parse_args()
    config = vars(args)
    
    main(config['cuda_id'], config['num_workers'], config['source_wandb_project_name'], config['target_wandb_project_name'], config['source_validation_setting'], config['target_validation_setting'], config['dataset_name'])

