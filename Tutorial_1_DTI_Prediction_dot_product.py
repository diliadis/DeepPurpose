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

def generate_combinations(ranges_per_param_name_dict, output_file_dir=None):
    # generate all combinations and store in dataframe
    df = pd.DataFrame(itertools.product(*ranges_per_param_name_dict.values()), columns=ranges_per_param_name_dict.keys())
    # shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    if output_file_dir:
        df.to_csv(output_file_dir+'.csv', index=False)
    return df

def main(num_samples):

    wandb_project_name = 'DeepPurpose_repeat'
    wandb_project_entity = 'diliadis'
    general_architecture_version = 'dot_product'
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
        'embedding_size': [4, 8, 16, 32, 64, 128, 256, 512],
        'mpnn_depth': [1, 2, 3],
        
        # 'hidden_dim_protein': [4, 8, 16, 32, 64, 128, 256, 512],
        'cnn_target_filters': [16, 32, 64, 128],
        'cnn_target_kernels': [4, 8, 12, 16],

        'cls_depth': [1, 2, 3],
        'cls_hidden_size': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    }

    api = wandb.Api()
    entity, project = wandb_project_entity, wandb_project_name  # set to your entity and project 
    runs = api.runs(entity + "/" + project) 
    completed_param_combinations = {param_name: [] for param_name in ranges_dict.keys()}
    for run in tqdm(runs):
        if run.state == "finished":
            if run.config['general_architecture_version'] == general_architecture_version:
                for param_name in ranges_dict.keys():
                    if param_name == 'learning_rate':
                        completed_param_combinations[param_name].append(run.config['LR'])
                    elif param_name == 'embedding_size':
                        completed_param_combinations[param_name].append(run.config['hidden_dim_drug'])
                    else:
                        completed_param_combinations[param_name].append(run.config[param_name][0] if isinstance(run.config[param_name], list) else run.config[param_name])
    # dataframe with configurations already tested and logged to wandb
    completed_param_combinations_df = pd.DataFrame(completed_param_combinations)
    
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
                (completed_param_combinations_df['embedding_size'] == temp_config['embedding_size']) & 

                (completed_param_combinations_df['cls_depth'] == temp_config['cls_depth']) & 
                (completed_param_combinations_df['cls_hidden_size'] == temp_config['cls_hidden_size']) &
                (completed_param_combinations_df['cnn_target_filters'].apply((temp_config['cnn_target_filters']).__eq__)) &
                (completed_param_combinations_df['cnn_target_kernels'].apply((temp_config['cnn_target_kernels']).__eq__)) &
                (completed_param_combinations_df['mpnn_depth'] == temp_config['mpnn_depth'])
            ].empty:
                print('NEW CONFIG FOUND: '+str(temp_config))
                unseen_config_found = True 

        print('testing the following config: '+str(temp_config))
        config = utils.generate_config(drug_encoding = drug_encoding, 
                                target_encoding = target_encoding, 
                                # cls_hidden_dims = [1024,1024,512], 
                                train_epoch = 100, 
                                LR = temp_config['learning_rate'], 
                                batch_size = 256,
                                hidden_dim_drug = int(temp_config['embedding_size']),
                                hidden_dim_protein = int(temp_config['embedding_size']),
                                mpnn_depth = int(temp_config['mpnn_depth']),
                                
                                cnn_target_filters = temp_config['cnn_target_filters'],
                                cnn_target_kernels = temp_config['cnn_target_kernels'],
                                
                                general_architecture_version = general_architecture_version,
                                cuda_id='0',
                                wandb_project_name = wandb_project_name,
                                wandb_project_entity = wandb_project_entity,
                                use_early_stopping = True,
                                patience = 5,
                                delta = 0.001,
                                metric_to_optimize_early_stopping = 'loss',
                                num_workers=4,
                                )
        config['protein_mode_coverage'] = 'extended'
        
        model = models.model_initialize(**config)
        print(str(model.config))
        model.train(train, val, test)

if __name__ == "__main__":
    n = int(sys.argv[1])
    main(n)