from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models
from DeepPurpose.utils import *

import pandas as pd
import wandb
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")


def main(run_id):
    
    general_architecture_version = 'dot_product'
    
    wandb_project_name = 'DeepPurpose_repeat'
    wandb_project_entity = 'diliadis'
    
    # load the config of the requested run from wandb
    api = wandb.Api()
    run = api.run(path=wandb_project_entity+'/'+wandb_project_name+'/'+run_id)
    best_config = run.config
    best_config['cuda_id'] = 0
    
    # load and split dataset
    X_drugs, X_targets, y = dataset.load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30) # http://staff.cs.utu.fi/~aatapa/data/DrugTarget/
    drug_encoding, target_encoding = best_config['drug_encoding'], best_config['target_encoding']
    print('Processing the dataset...')
    train, val, test = utils.data_process(X_drugs, X_targets, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2],
                                random_seed = 1)
    print('Done! ')
    config = {}
    '''
    config = utils.generate_config(drug_encoding = drug_encoding, 
                            target_encoding = target_encoding, 
                            train_epoch = 100, 
                            LR = 0.001, 
                            batch_size = 256,
                            hidden_dim_drug = 64,
                            hidden_dim_protein = 64,
                            mpnn_depth = 3,
                            
                            cnn_target_filters = [32,64,96],
                            cnn_target_kernels = [4,8,12],
                            
                            general_architecture_version = general_architecture_version,
                            cuda_id='4',
                            wandb_project_name = wandb_project_name,
                            wandb_project_entity = wandb_project_entity,
                            use_early_stopping = True,
                            patience = 5,
                            delta = 0.001,
                            metric_to_optimize_early_stopping = 'loss',
                            num_workers=4,
                            experiment_name='best_'+general_architecture_version+'model'
                            )
    '''
    # updating the dummy config with the dictionary loaded from wandb
    config.update(best_config)
    # initialize the model
    model = models.model_initialize(**config)
    print(str(model.model))
    print(str(model.config))
    # start training, validating, testing
    model.train(train, val, test)
    
    
if __name__ == "__main__":
    id = str(sys.argv[1])
    main(id)