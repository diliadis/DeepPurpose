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


def main(num_samples, val_setting, cuda_id, num_workers, dataset_name, performance_threshold=1.0, wandb_dir='/data/gent/vo/000/gvo00048/vsc43483'):
    num_samples = int(num_samples)
    
    split_method = 'random'
    if str(val_setting) == 'B':
        split_method = 'cold_drug'
    elif str(val_setting) == 'C':
        split_method = 'cold_protein'
    elif str(val_setting) == 'A':
        split_method = 'random'

    wandb_project_name = 'DeepPurpose_final_equal_embeddings_simple'
    wandb_project_entity = 'diliadis'
    general_architecture_version = 'mlp'

    if dataset_name.lower() == 'davis':
        X_drugs, X_targets, y = dataset.load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30)
    elif dataset_name.lower() == 'kiba':
        X_drugs, X_targets, y = dataset.load_process_KIBA(path = './data/', binary=False)
    else:
        raise AttributeError('invalid dataset name passed.')
    
    drug_encoding, target_encoding = 'Morgan', 'AAC'
    print('Processing the dataset...')
    train, val, test = utils.data_process(X_drugs, X_targets, y,
                                drug_encoding, target_encoding, 
                                split_method=split_method,frac=[0.7,0.1,0.2],
                                random_seed = 1)
    print('Done! ')


    ranges_dict = {
        'learning_rate': [0.01, 0.001, 0.0001, 0.00001, 0.000001],
        'hidden_dim_drug': [4, 8, 16, 32, 64, 128, 256, 512],
        # 'hidden_dim_protein': [4, 8, 16, 32, 64, 128, 256, 512],

        'mlp_drug_depth': [1,2,3,4],
        'mlp_drug_nodes_per_layer': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        
        'mlp_target_depth': [1,2,3,4],
        'mlp_target_nodes_per_layer': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],

        'cls_depth': [1, 2, 3],
        'cls_hidden_size': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    }

    api = wandb.Api()
    entity, project = wandb_project_entity, wandb_project_name  # set to your entity and project 
    runs = api.runs(entity + "/" + project) 
    completed_param_combinations = {param_name: [] for param_name in ranges_dict.keys()}
    print(str(completed_param_combinations))
    for run in tqdm(runs):
        if run.state == "finished":
            if ((run.config['general_architecture_version'] == general_architecture_version) and (run.config['dataset_name'] == dataset_name) and (run.config['validation_setting'] == val_setting)):
                for param_name in ranges_dict.keys():
                    if param_name == 'learning_rate':
                        completed_param_combinations[param_name].append(run.config['LR'])
                    elif param_name in 'cls_depth':
                        completed_param_combinations[param_name].append(len(run.config['cls_hidden_dims']))
                    elif param_name == 'cls_hidden_size':
                        completed_param_combinations[param_name].append(run.config['cls_hidden_dims'][0])
                    elif param_name in 'mlp_drug_depth':
                        completed_param_combinations[param_name].append(len(run.config['mlp_hidden_dims_drug']))
                    elif param_name == 'mlp_drug_nodes_per_layer':
                        completed_param_combinations[param_name].append(run.config['mlp_hidden_dims_drug'][0])
                    elif param_name in 'mlp_target_depth':
                        completed_param_combinations[param_name].append(len(run.config['mlp_hidden_dims_target']))
                    elif param_name == 'mlp_target_nodes_per_layer':
                        completed_param_combinations[param_name].append(run.config['mlp_hidden_dims_target'][0])
                    else:
                        completed_param_combinations[param_name].append(run.config[param_name][0] if isinstance(run.config[param_name], list) else run.config[param_name])
                    
    # dataframe with configurations already tested and logged to wandb
    completed_param_combinations_df = pd.DataFrame(completed_param_combinations)
    print('completed configs df: '+str(completed_param_combinations_df))
    
    num_remaining_configs = np.prod([len(v) for k, v in ranges_dict.items()]) - len(completed_param_combinations_df) 
    
    if num_remaining_configs != 0:
        if num_samples > num_remaining_configs:
            num_samples = num_remaining_configs
            print('I will actually run '+str(num_samples)+' different configurations')
    
    for experiment_id in range(num_samples):
        
        unseen_config_found = False
        temp_config = {}
        while not unseen_config_found:
            temp_config.update({param_name: random.sample(vals, 1)[0] for param_name, vals in ranges_dict.items()}) 
            if completed_param_combinations_df[
                (completed_param_combinations_df['learning_rate'] == temp_config['learning_rate']) & 
                (completed_param_combinations_df['hidden_dim_drug'] == temp_config['hidden_dim_drug']) & 
                # (completed_param_combinations_df['hidden_dim_protein'] == temp_config['hidden_dim_protein']) & 

                (completed_param_combinations_df['cls_depth'] == temp_config['cls_depth']) & 
                (completed_param_combinations_df['cls_hidden_size'] == temp_config['cls_hidden_size']) &
                (completed_param_combinations_df['mlp_drug_depth'] == temp_config['mlp_drug_depth']) & 
                (completed_param_combinations_df['mlp_drug_nodes_per_layer'] == temp_config['mlp_drug_nodes_per_layer']) &
                (completed_param_combinations_df['mlp_target_depth'] == temp_config['mlp_target_depth']) & 
                (completed_param_combinations_df['mlp_target_nodes_per_layer'] == temp_config['mlp_target_nodes_per_layer'])
            ].empty:
                completed_param_combinations_df = completed_param_combinations_df.append(temp_config, ignore_index=True)
                print('NEW CONFIG FOUND: '+str(temp_config))
                print('The dataframe now containts: '+str(completed_param_combinations_df))
                unseen_config_found = True 

        print('testing the following config: '+str(temp_config))
        config = utils.generate_config(drug_encoding = drug_encoding, 
                                target_encoding = target_encoding, 
                                cls_hidden_dims = int(temp_config['cls_depth']) * [int(temp_config['cls_hidden_size'])], 
                                train_epoch = 100, 
                                LR = temp_config['learning_rate'], 
                                batch_size = 256,
                                hidden_dim_drug = int(temp_config['hidden_dim_drug']),
                                hidden_dim_protein = int(temp_config['hidden_dim_drug']),
                                mlp_hidden_dims_drug = int(temp_config['mlp_drug_depth']) * [int(temp_config['mlp_drug_nodes_per_layer'])],
                                mlp_hidden_dims_target = int(temp_config['mlp_target_depth']) * [int(temp_config['mlp_target_nodes_per_layer'])],
                                
                                general_architecture_version = general_architecture_version,
                                cuda_id=str(cuda_id),
                                wandb_project_name = wandb_project_name,
                                wandb_project_entity = wandb_project_entity,
                                wandb_dir = wandb_dir,
                                use_early_stopping = True,
					            patience = 30,
					            delta = 0.001,
					            metric_to_optimize_early_stopping = 'loss',
                                num_workers=int(num_workers),
                                performance_threshold = {'metric_name':'MSE', 'value': performance_threshold, 'direction': 'min', 'max_epochs_allowed': 30},
                                validation_setting=val_setting,
                                dataset_name = dataset_name.upper()
                                )
        config['protein_mode_coverage'] = 'extended'

        model = models.model_initialize(**config)
        print(str(model.model))
        print(str(model.config))
        model.train(train, val, test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepPurpose DTI example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_configs", help="number of different configuration that will be trained and tested")
    parser.add_argument("--val_setting", help="the validation setting that will be used to split the data")
    parser.add_argument("--cuda_id", help="the id of the GPU that will be used for training")
    parser.add_argument("--num_workers", help="the number of workers that will be used by the dataloaders")
    parser.add_argument("--dataset_name", help="the name of the dataset that will be used. (DAVIS and KIBA are the current valid options)")
    parser.add_argument("--performance_threshold", help="performance threshold checked before epoch 30")
    args = parser.parse_args()
    config = vars(args)
    
    main(config['num_configs'], config['val_setting'], config['cuda_id'], config['num_workers'], config['dataset_name'], performance_threshold=float(config['performance_threshold']))

