from DeepPurpose import utils, dataset
from sklearn.model_selection import train_test_split
from DeepPurpose.utils import *
import itertools
import torch
import numpy as np
from AutoEncoder import AutoEncoder
from torch.utils.data import DataLoader, Dataset
from torch import nn
import random
import wandb
from tqdm import tqdm


def generate_combinations(ranges_per_param_name_dict, output_file_dir=None):
    # generate all combinations and store in dataframe
    df = pd.DataFrame(itertools.product(*ranges_per_param_name_dict.values()), columns=ranges_per_param_name_dict.keys())
    # shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    if output_file_dir:
        df.to_csv(output_file_dir+'.csv', index=False)
    return df

def main(num_samples):
    
    branch_model_to_use = 'protein'
    drug_encoding = 'MPNN'
    target_encoding = 'CNN'

    wandb_project_name = 'Protein_autoencoder_with_linear_embedding'
    wandb_project_entity = 'diliadis'
    
    X_drugs, X_targets, y = dataset.load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30) # http://staff.cs.utu.fi/~aatapa/data/DrugTarget/
    drug_encoding, target_encoding = drug_encoding, target_encoding
    print('Processing the dataset...')
    train, _, _ = utils.data_process(X_drugs, X_targets, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2],
                                random_seed = 1)
    print('Done! ')

    # get the feature representations of the unique drugs or proteins
    data = train.drop_duplicates('SMILES' if branch_model_to_use=='drug' else 'Target Sequence', ignore_index=True)['SMILES' if branch_model_to_use=='drug' else 'target_encoding']

    frac = {'train': 0.8, 'val': 0.1, 'test': 0.2}
    print('Splitting the dataset...')
    # split to train, val, test
    train, test = train_test_split(data, test_size=frac['test'], random_state=42)
    train, val = train_test_split(train, test_size=frac['val']/(1-frac['test']), random_state=42)
    print('Done! ')

    ranges_dict = {
        'learning_rate': [0.01, 0.001, 0.0001, 0.00001, 0.000001],
        # 'embedding_size': [4, 8, 16, 32, 64, 128, 256, 512],
        # 'mpnn_depth': [1, 2, 3],
        
        # 'hidden_dim_protein': [4, 8, 16, 32, 64, 128, 256, 512],
        'cnn_filters': [4, 8, 16, 32, 64, 128],
        'cnn_kernels': [2, 4, 8, 12, 16],
        'embedding_size': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        # 'cls_depth': [1, 2, 3],
        # 'cls_hidden_size': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    }
    
    
    api = wandb.Api()
    entity, project = wandb_project_entity, wandb_project_name  # set to your entity and project 
    runs = api.runs(entity + "/" + project) 
    completed_param_combinations = {param_name: [] for param_name in ranges_dict.keys()}
    for run in tqdm(runs):
        if run.state == "finished":
            for param_name in ranges_dict.keys():
                if param_name == 'learning_rate':
                    completed_param_combinations[param_name].append(run.config['LR'][0] if isinstance(run.config['LR'], list) else run.config['LR'])
                elif 'cnn_filters' in param_name:
                    completed_param_combinations[param_name].append(run.config['cnn_filters'])
                    # completed_param_combinations[param_name].append(run.config['cnn_target_filters'][int(param_name.split('_')[-1])]  if int(param_name.split('_')[-1]) < len(run.config['cnn_target_filters']) else -1)
                elif 'cnn_kernels' in param_name:
                    completed_param_combinations[param_name].append(run.config['cnn_kernels'])
                elif 'embedding_size' in param_name:
                    completed_param_combinations[param_name].append(run.config['embedding_size'])
                    
    # dataframe with configurations already tested and logged to wandb
    completed_param_combinations_df = pd.DataFrame(completed_param_combinations) 
    
    num_remaining_configs = np.prod([len(v) for k, v in ranges_dict.items()]) - len(completed_param_combinations_df) 
    
    if num_remaining_configs != 0:
        if num_samples > num_remaining_configs:
            num_samples = num_remaining_configs
            print('I will actually run '+str(num_samples)+' different configurations')
    
        for experiment_id in range(num_samples):    
            unseen_config_found = False
            while not unseen_config_found:
                temp_config = {param_name: random.sample(vals, 1)[0] for param_name, vals in ranges_dict.items() if param_name not in ['cnn_filter', 'cnn_kernel']} 
                cnn_num_layers = random.randint(1, 3)
                temp_config['cnn_filters'] = random.sample(ranges_dict['cnn_filters'], cnn_num_layers)
                temp_config['cnn_kernels'] = random.sample(ranges_dict['cnn_kernels'], cnn_num_layers)
                
                if completed_param_combinations_df[
                    (completed_param_combinations_df['learning_rate'] == temp_config['learning_rate']) & 
                    (completed_param_combinations_df['cnn_filters'].apply((temp_config['cnn_filters']).__eq__)) &
                    (completed_param_combinations_df['cnn_kernels'].apply((temp_config['cnn_kernels']).__eq__)) & 
                    (completed_param_combinations_df['embedding_size'].apply((temp_config['embedding_size']).__eq__))
                ].empty:
                    unseen_config_found = True 
            
                print('testing the following config: '+str(temp_config))
            
            config = {
                'wandb_project_name': wandb_project_name,
                'wandb_project_entity': 'diliadis',
                
                'drug_encoding': 'MPNN',
                'target_encoding': 'CNN',
                
                'cuda_id': '0',
                'num_workers': 8,
                
                # 'experiment_name': 'autoencoder_'+branch_model_to_use+'_',
                'experiment_name': None,
                'result_folder': './results/',
                
                'decay': 0,
                'LR': 0.001,
                
                'batch_size': 32,
                'train_epoch': 100,
                'test_every_X_epoch': 5,
                
                'cnn_filters': [32, 16, 8],
                'cnn_kernels': [3, 3, 2],
                
                'embedding_size': 2,
                
                'use_early_stopping': True,
                'patience': 5,
                'delta': 0.0005,
                'metric_to_optimize_early_stopping': 'loss',
                'metric_to_optimize_best_epoch_selection': 'loss',
                
                'save_model': True
            }
            
            config.update(temp_config)
        
            # inialize the model
            model = AutoEncoder(config)
            model.train(train, val, test)
    
    else:
        print('There are no more configurations left to be run')
        
if __name__ == "__main__":
    n = int(sys.argv[1])
    main(n)