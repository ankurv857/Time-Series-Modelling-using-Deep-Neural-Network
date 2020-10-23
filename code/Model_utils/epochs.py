import pandas as pd
import numpy as np
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset , DataLoader
import utils
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Model_utils.loss_func import loss_func_seq , loss_func_non_seq
from Config.arguments import get_args ; args = get_args()
from Experiment.experiments import Experiment


def network_epoch(epoch,loader, model,model_level,optimizer,data_index,index_group,max_log_y ,tag , time_steps, exp = None):
    # Cycle through the data loader for num_batches equal to __len__ of the dataset
    #check if there is a gpu in case sets it as device
    # cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if cuda else "cpu")

    if tag == 'train':
        model.train()
        loader = loader
    else:
        model.eval()
        loader = loader

    cum_loss = 0 ; exp_start = time.time()
    for i, data in enumerate(loader):
        #print('enetered enumerate loader')
        target_data , mlp_stat_data, mlp_tmp_data , emb_stat_data, emb_tmp_data , dataframe_idxs = [arr for arr in data]
        #forward pass
        optimizer.zero_grad() #removes gradients from the optimizer, default is to accumulate
        pred = model(mlp_stat_data, mlp_tmp_data , emb_stat_data, emb_tmp_data)  #forward pass, it outputs a 2xseq_len predictions
        if model_level[2] in ['Neural_Network_mlp']:
            loss = loss_func_non_seq(pred ,target_data , args.seq_len) 
        else :
            loss = loss_func_seq(pred ,target_data , args.seq_len, time_steps)
        if tag ==  'train':
            loss.backward()
            optimizer.step()
        cum_loss += loss.item()

        predicted = pd.DataFrame(pred[:,].contiguous().view(-1).detach().cpu().numpy())   ; predicted.columns = ['predicted']
        actual = pd.DataFrame(target_data[:,].contiguous().view(-1).detach().cpu().numpy())  ; actual.columns = ['actual']
        result = pd.concat([actual , predicted] , axis = 1)
        
        result['actual'] = np.exp(result['actual'] * max_log_y) - 1 ; result['predicted'] = np.exp(result['predicted'] * max_log_y) - 1
        result['MAPE'] =  abs((result['actual'] - result['predicted'])/(result['actual']))* 100
        result['MAPE'] = np.where(result['MAPE'] == np.inf , 100, result['MAPE'])
        #result_backseqlen = result.iloc[:args.seq_len]
        result = result.iloc[args.seq_len:]
        if tag == 'train':
            result_append = result
        else:
            if i == 0:
                result_append = result.reset_index(drop = True)
            else:
                result_append = result_append.append(result , ignore_index = True)
            
        mape = np.mean(result_append['MAPE'])
        
        if exp:
            exp.log(i , dataframe_idxs.detach().cpu().numpy() , pred.detach().cpu().numpy(), target_data.detach().cpu().numpy() , loss.item())
    
    dt = time.time() - exp_start
    if exp:
        if tag != 'train':
            exp.save(epoch , tag ,model_level , dt)

    return loss , cum_loss , mape , result_append