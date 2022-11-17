import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import os
from DeepPurpose import utils, dataset
from datetime import datetime
import copy
import wandb
from DeepPurpose.utils import *


class EarlyStopping:
    '''Early stops the training if validation loss doesn't improve after a given patience.'''

    def __init__(self, use_early_stopping, patience=7, delta=0.0, metric_to_track='loss', verbose=False):
        '''
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        '''
        self.use_early_stopping = use_early_stopping
        self.patience = patience
        self.delta = delta
        self.metric_to_track = metric_to_track
        self.verbose = verbose

        self.early_stop_flag = False
        self.counter = 0

        self.best_score = None
        self.best_epoch = None
        self.best_performance_results = None

        if True in [
            m in self.metric_to_track.lower()
            for m in [
                'auroc',
                'aupr',
                'recall',
                'f1_score',
                'precision',
                'accuracy',
                'R2',
            ]
        ]:
            print('Early stopping detected metric: '+str(self.metric_to_track))
            self.fac = 1
        elif True in [
            m in self.metric_to_track.lower()
            for m in ['hamming_loss', 'RMSE', 'MSE', 'MAE', 'RRMSE', 'loss']
        ]:
            print('Early stopping detected metric: '+str(self.metric_to_track))
            self.fac = -1
        else:
            AttributeError(
                'Invalid metric name used for early stopping: '
                + str(self.metric_to_track)
            )


    def __call__(
        self,
        performance_results,
        epoch,
    ):

        score = performance_results['val_' + self.metric_to_track] * self.fac
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_performance_results = performance_results
            # self.save_checkpoint(val_loss, model)

        elif (score <= self.best_score + self.delta) and self.use_early_stopping:
            self.counter += 1
            if self.verbose:
                print(
                    f'-----------------------------EarlyStopping counter: {self.counter} out of {self.patience}---------------------- best epoch currently {self.best_epoch}'
                )
            if self.counter >= self.patience:
                self.early_stop_flag = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_performance_results = performance_results
            self.counter = 0


class AutoEncoder_model(nn.Module):
    def __init__(self, **config):
        super(AutoEncoder_model, self).__init__()
        
        self.cnn_filters = [26] + config['cnn_filters']
        self.cnn_kernels = config['cnn_kernels']
        
        self.encoder_list = []
        self.decoder_list = []
        
        for i in range(len(self.cnn_filters)-1):
            self.encoder_list.append(nn.Conv1d(in_channels=self.cnn_filters[i], out_channels=self.cnn_filters[i+1], kernel_size=self.cnn_kernels[i]))
            self.encoder_list.append(nn.ReLU(True))
            
        for i in reversed(range(len(self.cnn_filters)-1)):
            self.decoder_list.append(nn.ConvTranspose1d(in_channels=self.cnn_filters[i+1], out_channels=self.cnn_filters[i], kernel_size=self.cnn_kernels[-i]))
            self.decoder_list.append(nn.ReLU(True))
        self.decoder_list.append(nn.Tanh())
        
        self.encoder = nn.Sequential(*self.encoder_list)
        self.decoder = nn.Sequential(*self.decoder_list)

        self.encoder = self.encoder.double()
        self.decoder = self.decoder.double()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class AutoEncoder_model_with_linear_embedding(nn.Module):
    def __init__(self, **config):
        super(AutoEncoder_model, self).__init__()
        
        self.cnn_filters = [26] + config['cnn_filters']
        self.cnn_kernels = config['cnn_kernels']
        
        self.output_sizes = [1000]
        
        self.encoder_list = []
        self.decoder_list = []
        
        for i in range(len(self.cnn_filters)-1):
            self.output_sizes.append(self.calc_output_size(self.output_sizes[-1], self.cnn_kernels[i], 0, 1))
            self.encoder_list.append(nn.Conv1d(in_channels=self.cnn_filters[i], out_channels=self.cnn_filters[i+1], kernel_size=self.cnn_kernels[i]))
            self.encoder_list.append(nn.ReLU(True))
        self.encoder_list.append(nn.Flatten())
        self.encoder_list.append(nn.Linear(self.output_sizes[-1]*self.cnn_filters[-1], config['embedding_size']))
        
        self.decoder_list.append(nn.Linear(config['embedding_size'], self.output_sizes[-1]*self.cnn_filters[-1]))
        self.decoder_list.append(nn.Unflatten(1, (self.cnn_filters[-1], self.output_sizes[-1])))
        for i in reversed(range(len(self.cnn_filters)-1)):
            self.decoder_list.append(nn.ConvTranspose1d(in_channels=self.cnn_filters[i+1], out_channels=self.cnn_filters[i], kernel_size=self.cnn_kernels[-i]))
            self.decoder_list.append(nn.ReLU(True))
        self.decoder_list.append(nn.Tanh())
        
        self.encoder = nn.Sequential(*self.encoder_list)
        self.decoder = nn.Sequential(*self.decoder_list)

        self.encoder = self.encoder.double()
        self.decoder = self.decoder.double()
    
    def calc_output_size(self, W, K, P, S):
        # w: input volume
        # K: kernel size
        # P: padding size
        # S: stride size
        return int(((W-K+2*P)/S)+1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CustomDataset(Dataset):
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return protein_2_embed(self.data.iloc[index])


class AutoEncoder:
    
    def __init__(self, config):
        
        self.wandb_run = None
        
        if 'cuda_id' in config:
            if config['cuda_id'] is None:
                self.device = torch.device('cpu')
            else:
                self.device = torch.device('cuda:' + str(config['cuda_id']) if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        config['device'] = self.device

        print('Using the following device: '+str(self.device))
        
        # initialize a model
        if 'embedding_size' in config:
            self.model = AutoEncoder_model_with_linear_embedding(**config)
        else:
            self.model = AutoEncoder_model(**config)
        
        self.config = config
        
        self.result_folder = self.config['result_folder']
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)
        
        # create a custom folder for every experiment inside the resutls directory. This way, you don't have to lose your results every time you run an experiment.
        if self.config['experiment_name'] is None:
            self.experiment_dir = self.result_folder+datetime.now().strftime('%d_%m_%Y__%H_%M_%S')
        else:
            self.experiment_dir = self.result_folder+config['experiment_name']
        self.config['experiment_dir'] = self.experiment_dir
        os.mkdir(self.experiment_dir)
            
        self.binary = False
        if 'num_workers' not in self.config.keys():
            self.config['num_workers'] = 0
        if 'decay' not in self.config.keys():
            self.config['decay'] = 0
            
            
        self.early_stopping = EarlyStopping(
            use_early_stopping=self.config['use_early_stopping'],
            patience=self.config['patience'],
            delta=self.config['delta'],
            verbose=True,
            metric_to_track=self.config['metric_to_optimize_early_stopping'] if self.config['use_early_stopping'] else self.config['metric_to_optimize_best_epoch_selection']
        )
   
   
    def test_(self, data_generator, model):
        model.eval()
        loss_history = []
        i=0
        for data in data_generator:
            data = data.double().to(self.device) 
            output = self.model(data)
            loss_fct = torch.nn.MSELoss()
            # n = torch.squeeze(output, 1)
            loss = loss_fct(output, data)
            loss_history.append(loss.item())
            i += 1
        return np.mean(loss_history)
    
    
    def train(self, train, val, test, verbose=True):
        max_MSE = 10000
        BATCH_SIZE = self.config['batch_size']
        test_every_X_epoch = self.config['test_every_X_epoch'] if 'test_every_X_epoch' in self.config.keys() else 40
        loss_history = []

        self.model = self.model.to(self.device)
        
        # wandb logging
        if self.config['wandb_project_name'] is not None and self.config['wandb_project_entity'] is not None:
            self.wandb_run = wandb.init(project=self.config['wandb_project_name'], entity=self.config['wandb_project_entity'], reinit=True)
            self.wandb_run.watch(self.model)
            self.wandb_run.config.update(self.config)
        
  
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config['LR'], weight_decay=self.config['decay'])
        criterion = torch.nn.MSELoss()
        
        params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': self.config['num_workers'],
                'drop_last': False}
        train_generator = DataLoader(CustomDataset(train), **params)
        print('Num train batches: '+str(len(train_generator)))
        
        if val is not None:
            params['shuffle'] = True
            validation_generator = DataLoader(CustomDataset(val), **params)
        print('Num validation batches: '+str(len(validation_generator)))
        
        if test is not None:
            params_test = {'batch_size': BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': self.config['num_workers'],
                    'drop_last': False}
        
            test_generator = DataLoader(CustomDataset(test), **params_test)
            print('Num test batches: '+str(len(test_generator)))

        for epoch in range(self.config['train_epoch']):
            total_loss = 0
            # train loop
            for data in train_generator:
                data = data.double().to(self.device) 
                # ===================forward=====================
                output = self.model(data)
                loss = criterion(output, data)
                
                # ===================backward====================
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.data
            
            epoch_loss = total_loss.cpu().detach().numpy()/len(train_generator)
            print('Epoch '+str(epoch)+ ' train_loss: '+str(epoch_loss))
            
            if self.wandb_run is not None: self.wandb_run.log({'train_loss': epoch_loss, 'epoch': epoch})
            
            # validation loop
            if val is not None:
                with torch.set_grad_enabled(False):
                    loss = self.test_(validation_generator, self.model)
                    if loss < max_MSE:
                        model_max = copy.deepcopy(self.model)
                        max_MSE = loss
                    print('      '+str(epoch)+ ' val_loss: '+str(loss))

                if self.wandb_run is not None:
                    # self.wandb_run.log({'val_MSE': mse, 'val_pearson_correlation': r2, 'val_concordance_index': CI, 'epoch': epo})
                    self.wandb_run.log({'val_loss': loss, 'epoch': epoch}, commit=True)
                
                # update early stopping and keep track of best model 
                self.early_stopping(
                    {'val_loss': loss}, epoch
                )    
                
                if self.early_stopping.early_stop_flag and self.config['use_early_stopping']:
                    print('Early stopping criterion met. Training stopped!!!')
                    break
        
        self.wandb_run.log({'best_loss': max_MSE})
        
        self.model = model_max
        
        if test is not None:
            loss = self.test_(test_generator, model_max)
            print('TEST_loss: '+str(loss))
            if self.wandb_run is not None:
                # self.wandb_run.log({'val_MSE': mse, 'val_pearson_correlation': r2, 'val_concordance_index': CI, 'epoch': epo})
                self.wandb_run.log({'test_loss': loss})
                
                
        if self.wandb_run is not None:
            self.wandb_run.finish()
            
        if self.config['save_model']:
            torch.save(self.model.state_dict(), self.experiment_dir + '/model.pt')
            save_dict(self.experiment_dir, self.config)
            
    
    def load_pretrained(self, path, device):
        if not os.path.exists(path):
            os.makedirs(path)

        state_dict = torch.load(path, map_location = device)
        
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)
