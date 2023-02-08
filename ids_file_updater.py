import wandb
import argparse
import pandas as pd

def main(source_wandb_project_name, reserved_run_ids_file_name, top_k):
    
    # Read the text file with the ids
    with open(reserved_run_ids_file_name+'.txt') as f:
        ids = f.readlines()
        
    # Strip newline characters and convert to a set
    ids = set(id.strip() for id in ids)
    
    print('Number of ids in the .txt file: '+str(len(ids)))
    

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
    result = grouped.apply(lambda x: x.nsmallest(int(top_k), 'best_val_loss'))
    # Reset the index
    result.reset_index(drop=True, inplace=True)
    
    run_ids = set(result['id'].to_list())
    
    # Get a set of all the run ids
    # run_ids = set(run.id for run in runs)
    
    print('Number of ids in the wandb project: '+str(len(run_ids)))
    
    ids = ids.intersection(run_ids)
    
    print('Re-writing the file...')
    # Write the updated ids back to the text file
    with open('file_name.txt', 'w') as f:
        for id in ids:
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

