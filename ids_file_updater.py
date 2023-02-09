import wandb
import argparse
import pandas as pd
from tqdm import tqdm

def main(source_wandb_project_name, reserved_run_ids_file_name, top_k):
    
    # Read the text file with the ids
    with open(reserved_run_ids_file_name+'.txt') as f:
        ids = f.readlines()
        
    # Strip newline characters and convert to a set
    ids = set(id.strip() for id in ids)
    
    print('Number of ids in the .txt file: '+str(len(ids)))
    

    api = wandb.Api()
    runs = api.runs('diliadis/' + source_wandb_project_name) 
    
    # Get a set of all the run ids
    run_ids = set(run.id for run in runs)
    
    print('Number of ids in the wandb project: '+str(len(run_ids)))
    
    ids = ids.intersection(run_ids)
    
    print(str(ids))
    
    print('Re-writing the file...')
    # Write the updated ids back to the text file
    with open(reserved_run_ids_file_name+'.txt', 'w') as f:
        for id in tqdm(ids):
            f.write(f"{id}\n")
    print('Done')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepPurpose DTI example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source_wandb_project_name", help="name of the source wandb project from which I will select configurations to extend")
    parser.add_argument("--reserved_run_ids_file_name", help="name of file that wll be used to store reserved ids")
    parser.add_argument("--top_k", help="select the number of top performing configs to be extended")
    
    args = parser.parse_args()
    config = vars(args)
    
    main(config['source_wandb_project_name'], config['reserved_run_ids_file_name'], config['top_k'])

