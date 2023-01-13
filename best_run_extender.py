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

def main(cuda_id, num_workers, source_wandb_project_name, target_wandb_project_name, wandb_dir='/data/gent/vo/000/gvo00048/vsc43483'):
    
    update_file = "reserved_wandb_ids.txt"

    # check if the file exists
    if not os.path.exists(update_file):
        # create the file if it doesn't exist
        open(update_file, "w").close()
        print(f"{update_file} has been created.")
    else:
        print(f"{update_file} already exists.")
    
    
    api = wandb.Api()
    runs = api.runs('diliadis/' + source_wandb_project_name) 

    data = {'id': [], 'validation_setting': [], 'dataset_name': [], 'general_architecture_version': [],'best_val_loss': [], 'test_MSE': [], 'epoch': []}

    for run in tqdm(runs):
        data['id'].append(run.id)
        data['validation_setting'].append(run.config['validation_setting'])
        data['dataset_name'].append(run.config['dataset_name'])
        data['general_architecture_version'].append(run.config['general_architecture_version'])
        data['epoch'].append(run.summary['epoch'])
        data['test_MSE'].append(run.summary['test_MSE'])
        data['best_val_loss'].append(run.summary['best_val_loss'])
        
    df = pd.DataFrame(data)
    
    # Group the dataframe by the specified columns
    grouped = df.groupby(['general_architecture_version', 'dataset_name', 'validation_setting'])
    # Select the rows with the smallest 'best_val_loss' value from each group
    result = grouped.apply(lambda x: x.nsmallest(1, 'best_val_loss'))
    # Reset the index
    result.reset_index(drop=True, inplace=True)

    for id in tqdm(result['id'].to_list()):
        
        run = api.run('diliadis/' + source_wandb_project_name + '/' + id) 
        
        # extract the max epoch the configuration achieved during training
        max_epoch = run.summary._json_dict['epoch']
        source_config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        
        print(run.id+': '+str(max_epoch)+', '+str(source_config.get('reserved'))+' ) ==========================================================================================================')
        
        if max_epoch == 99 and not source_config.get('reserved'):

            file_lock = threading.Lock()
            
            print('Getting lock....')
            file_lock.acquire()
            print('Got it!!!')
            # Open the file in read mode
            print('Reading file...')
            with open(update_file, "r") as f:
                updates = f.read()
            print('Done.')
            
            is_reserved = run.id in updates
            
            print(run.id+' is in '+updates+' : '+str(is_reserved)) 
            
            if not is_reserved:
                
                with open(update_file, "a") as f:
                    # Write the current time to the file as an update
                    f.write(str(run.id)+"\n")
                file_lock.release()
            # if max_epoch == 99 and not source_config.get('reserved', False): # the config should be eligible for extension to more epochs and it should not be reserved by any other script that may be running at the same time
            
                run.config['reserved'] = True
                run.update()
                
                if source_config['validation_setting'] == 'B':
                    split_method = 'cold_drug'
                if source_config['validation_setting'] == 'C':
                    split_method = 'cold_protein'
                if source_config['validation_setting'] == 'A':
                    split_method = 'random'
                    
                dataset_name = source_config['dataset_name']
                
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
                                        train_epoch = 200, 
                                        cuda_id=str(cuda_id),
                                        wandb_project_name = target_wandb_project_name,
                                        wandb_dir = wandb_dir,
                                        num_workers=int(num_workers),
                                        parent_wandb_id = run.id,
                )
                
                config.update({k: v for k,v in source_config.items() if k not in ['device', 'cuda_id', 'wandb_dir', 'num_workers', 'train_epoch', 'wandb_project_name']})
                
                model = models.model_initialize(**config)
                print(str(model.model))
                print(str(model.config))
                model.train(train, val, test)
            else:
                print('This run is actually reserved in the .txt file... ')
            
        else:
            print('Not a config that needs to be extended')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepPurpose DTI example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda_id", help="the id of the GPU that will be used for training")
    parser.add_argument("--num_workers", help="the number of workers that will be used by the dataloaders")
    parser.add_argument("--source_wandb_project_name", help="name of the source wandb project from which I will select configurations to extend")
    parser.add_argument("--target_wandb_project_name", help="name of the target wandb project where I will save the extended configurations")

    args = parser.parse_args()
    config = vars(args)
    
    main(config['cuda_id'], config['num_workers'], config['source_wandb_project_name'], config['target_wandb_project_name'])

