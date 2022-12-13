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


def main(num_samples, cuda_id, num_workers):
    num_samples = int(num_samples)

    wandb_project_name = 'DeepPurpose_repeat_2'
    wandb_project_entity = 'diliadis'
    general_architecture_version = 'kronecker'

    X_drugs, X_targets, y = dataset.load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30)
    drug_encoding, target_encoding = 'MPNN', 'CNN'
    print('Processing the dataset...')
    train, val, test = utils.data_process(X_drugs, X_targets, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2],
                                random_seed = 1)
    print('Done! ')


    ranges_dict = {
        'learning_rate': [0.01, 0.001, 0.0001, 0.00001, 0.000001],
        'hidden_dim_drug': [4, 8, 16, 32, 64, 128, 256, 512],
        'mpnn_depth': [1, 2, 3],
        
        'hidden_dim_protein': [4, 8, 16, 32, 64, 128, 256, 512],
        'cnn_target_filters': [16, 32, 64, 128],
        'cnn_target_kernels': [4, 8, 12, 16],

    }

    api = wandb.Api()
    entity, project = wandb_project_entity, wandb_project_name  # set to your entity and project 
    runs = api.runs(entity + "/" + project) 
    completed_param_combinations = {param_name: [] for param_name in ranges_dict.keys()}
    print(str(completed_param_combinations))
    for run in tqdm(runs):
        if run.state == "finished":
            if run.config['general_architecture_version'] == general_architecture_version:
                for param_name in ranges_dict.keys():
                    if param_name == 'learning_rate':
                        completed_param_combinations[param_name].append(run.config['LR'])
                    elif 'cnn_target_filters' in param_name:
                        completed_param_combinations[param_name].append(run.config['cnn_target_filters'])
                        # completed_param_combinations[param_name].append(run.config['cnn_target_filters'][int(param_name.split('_')[-1])]  if int(param_name.split('_')[-1]) < len(run.config['cnn_target_filters']) else -1)
                    elif 'cnn_target_kernels' in param_name:
                        completed_param_combinations[param_name].append(run.config['cnn_target_kernels'])
                        # completed_param_combinations[param_name].append(run.config['cnn_target_kernels'][int(param_name.split('_')[-1])] if int(param_name.split('_')[-1]) < len(run.config['cnn_target_kernels']) else -1)
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
            temp_config.update({param_name: random.sample(vals, 1)[0] for param_name, vals in ranges_dict.items() if param_name not in ['cnn_target_filter', 'cnn_target_kernel']}) 
            cnn_num_layers = random.randint(1, 3)
            temp_config['cnn_target_filters'] = random.sample(ranges_dict['cnn_target_filters'], cnn_num_layers)
            temp_config['cnn_target_kernels'] = random.sample(ranges_dict['cnn_target_kernels'], cnn_num_layers)
            
            if completed_param_combinations_df[
                (completed_param_combinations_df['learning_rate'] == temp_config['learning_rate']) & 
                (completed_param_combinations_df['hidden_dim_drug'] == temp_config['hidden_dim_drug']) & 
                (completed_param_combinations_df['hidden_dim_protein'] == temp_config['hidden_dim_protein']) & 

                (completed_param_combinations_df['cnn_target_filters'].apply((temp_config['cnn_target_filters']).__eq__)) &
                (completed_param_combinations_df['cnn_target_kernels'].apply((temp_config['cnn_target_kernels']).__eq__)) &
                (completed_param_combinations_df['mpnn_depth'] == temp_config['mpnn_depth'])
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
                                hidden_dim_protein = int(temp_config['hidden_dim_protein']),
                                mpnn_depth = int(temp_config['mpnn_depth']),
                                mpnn_hidden_size = 50,
                                cnn_target_filters = temp_config['cnn_target_filters'],
                                cnn_target_kernels = temp_config['cnn_target_kernels'],
                                
                                general_architecture_version = general_architecture_version,
                                cuda_id=str(cuda_id),
                                wandb_project_name = wandb_project_name,
                                wandb_project_entity = wandb_project_entity,
                                use_early_stopping = True,
					            patience = 5,
					            delta = 0.001,
					            metric_to_optimize_early_stopping = 'loss',
                                num_workers=int(num_workers),
                                performance_threshold = {'metric_name':'MSE', 'value': 1, 'direction': 'min', 'max_epochs_allowed': 30}
                                )
        config['protein_mode_coverage'] = 'extended'

        model = models.model_initialize(**config)
        print(str(model.model))
        print(str(model.config))
        model.train(train, val, test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepPurpose DTI example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_configs", help="number of different configuration that will be trained and tested")
    parser.add_argument("--cuda_id", help="the id of the GPU that will be used for training")
    parser.add_argument("--num_workers", help="the number of workers that will be used by the dataloaders")

    args = parser.parse_args()
    config = vars(args)
    
    main(config['num_configs'], config['cuda_id'], config['num_workers'])