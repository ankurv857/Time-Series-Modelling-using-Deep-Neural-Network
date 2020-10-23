import pandas as pd
import numpy as np
import numpy
import os
from os.path import join, exists
import sys
import warnings
import copy
warnings.filterwarnings("ignore")
import datetime
from datetime import time , date
import time

from dateutil.relativedelta import relativedelta
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset , DataLoader
import utils
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.manual_seed(0)


from Data_Preparation.prepare import data_prep
from Data_Preparation.reader import data_read
from DataLoader.loader import data_loader_with_offset 
from Model.model import  Neural_Network_lstmfused , Neural_Network_lstmcell , Neural_Network_wavenet , Neural_Network_cnnlstm , Neural_Network_mlp
from Config.arguments import get_args ; args = get_args()
from Model_utils.EarlyStopping import EarlyStopping, save_checkpoint
from Model_utils.epochs import network_epoch
from Config.config import config
from Model_utils.dynamic_batch_size import dynamic_batch_size
from Experiment.experiments import Experiment


if __name__ == '__main__':
    print('YO!!! The journey begins! ML automation with Pytorch <Neural Network Workflows>!')
    #check if there is a gpu in case sets it as device
    # cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if cuda else "cpu")

    code_start_time = time.time() 

    #Start new experiment log if an experiment name is provided
    exp = None
    if args.exp_name != '':
        exp = Experiment(args.exp_name)

    # #prepare the raw datasets
    data_prep = data_prep(config['data_list_dataprep'],config['target1'] , config['aggregation_list'] , config['causal_columns1'] , config['causal_columns2'])
    
    #Call the class data_read()
    data = data_read(config['data_list_dataread'], config['date_list'] , config['target'], config['idx'] , config['multiclass_discontinuous'] , config['text'] , config['remove_list'] )

    final_results_train = pd.DataFrame(columns= config['final_cols'] )
    final_results_val = pd.DataFrame(columns=config['final_cols'])
    final_results_test = pd.DataFrame(columns=config['final_cols'])
    final_results_future = pd.DataFrame(columns=config['final_cols'])

    for level_index ,level in enumerate(config['level_list']):
        for loader_num_index , loader_num in enumerate(config['loader_class_list']):
            for model_num_index ,model_num in enumerate(config['model_class_list']):
                for ts in config['timestep_list']:
                    time_steps = ts
                    for objects in data.df_list[0][level].unique():
                        model_level = [level ,loader_num , model_num , ts, objects]
                        df_list1 = data.df_list[0][(data.df_list[0][level] == objects)]
                        max_log_y = np.max(np.log(df_list1[config['target1']]+1)) + np.std(np.log(df_list1[config['target1']]+1)) 
                        df_list1[config['target1']] = np.log(df_list1[config['target1']]+1)/max_log_y #; print('df_list1' , df_list1)
                        batch_size = dynamic_batch_size(df_list1.index) #; print('batch_size' , batch_size)

                        if loader_num_index == 0 :
                            dataset_train = data_loader_with_offset([df_list1] , data.dtype_list , config['granularity'] ,config['static_list'] ,args.seq_len , None ,'train', config['train_val_date_list'],config['index'])
                            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size= batch_size , shuffle=True)
                            dataset_val = data_loader_with_offset([df_list1] , data.dtype_list , config['granularity'] ,config['static_list'] ,args.seq_len  , None,'val', config['train_val_date_list'],config['index'] )
                            val_loader = torch.utils.data.DataLoader(dataset_val, batch_size= 1  , shuffle=False)
                            dataset_test = data_loader_with_offset([df_list1] , data.dtype_list , config['granularity'] ,config['static_list'] ,args.seq_len  , None,'test' ,config['train_val_date_list'],config['index'] )
                            test_loader = torch.utils.data.DataLoader(dataset_test, batch_size= 1  , shuffle=False)
                            dataset_future = data_loader_with_offset([df_list1] , data.dtype_list , config['granularity'] ,config['static_list'] ,args.seq_len  , None,'future' ,config['train_val_date_list'],config['index'] )
                            future_loader = torch.utils.data.DataLoader(dataset_future, batch_size= 1  , shuffle=False)
                            _ = dataset_train[0] ; _ = dataset_val[0] ; _ = dataset_test[0] ; _ = dataset_future[0]

                        elif loader_num_index == 1 :
                            dataset_train = data_loader_wo_offset([df_list1] , data.dtype_list , config['granularity'] ,config['static_list'] ,args.seq_len , None ,'train', config['train_val_date_list'], config['index'])
                            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size= batch_size , shuffle=True , num_workers = 48) 
                            dataset_val = data_loader_wo_offset([df_list1] , data.dtype_list , config['granularity'] ,config['static_list'] ,args.seq_len  , None,'val', config['train_val_date_list'],config['index'] )
                            val_loader = torch.utils.data.DataLoader(dataset_val, batch_size= 1  , shuffle=False)
                            dataset_test = data_loader_wo_offset([df_list1] , data.dtype_list , config['granularity'] ,config['static_list'] ,args.seq_len  , None,'test' ,config['train_val_date_list'],config['index'] )
                            test_loader = torch.utils.data.DataLoader(dataset_test, batch_size= 1  , shuffle=False)
                            dataset_future = data_loader_wo_offset([df_list1] , data.dtype_list , config['granularity'] ,config['static_list'] ,args.seq_len  , None,'future' ,config['train_val_date_list'],config['index'] )
                            future_loader = torch.utils.data.DataLoader(dataset_future, batch_size= 1  , shuffle=False)
                            _ = dataset_train[0] ; _ = dataset_val[0] ; _ = dataset_test[0] ; _ = dataset_future[0]

                        #call the class Neural_Network()
                        if model_num_index == 0:
                            model = Neural_Network_lstmfused(dataset_train.emb_tmpl_list_inp, dataset_train.emb_tmpl_list_out, dataset_train.emb_static_list_inp, dataset_train.emb_static_list_out, dataset_train.mlp_tmpl_id, dataset_train.mlp_static_id)#.to(device)
                        elif model_num_index == 1:
                            model = Neural_Network_wavenet(dataset_train.emb_tmpl_list_inp, dataset_train.emb_tmpl_list_out, dataset_train.emb_static_list_inp, dataset_train.emb_static_list_out, dataset_train.mlp_tmpl_id, dataset_train.mlp_static_id)#.to(device)
                        elif model_num_index == 2:
                            model = Neural_Network_cnnlstm(dataset_train.emb_tmpl_list_inp, dataset_train.emb_tmpl_list_out, dataset_train.emb_static_list_inp, dataset_train.emb_static_list_out, dataset_train.mlp_tmpl_id, dataset_train.mlp_static_id)#.to(device)
                        elif model_num_index == 3:
                            model = Neural_Network_mlp(dataset_train.emb_tmpl_list_inp, dataset_train.emb_tmpl_list_out, dataset_train.emb_static_list_inp, dataset_train.emb_static_list_out, dataset_train.mlp_tmpl_id, dataset_train.mlp_static_id)#.to(device)
                        elif model_num_index == 4:
                            model = Neural_Network_lstmcell(dataset_train.emb_tmpl_list_inp, dataset_train.emb_tmpl_list_out, dataset_train.emb_static_list_inp, dataset_train.emb_static_list_out, dataset_train.mlp_tmpl_id, dataset_train.mlp_static_id)#.to(device)            
                   
                        # criterion = torch.nn.MSELoss(size_average=True)
                        optimizer = torch.optim.RMSprop(model.parameters(),lr=args.lr,  alpha=.9, momentum=0.9)
                        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = args.lr, max_lr = 1e-8)
                        # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
                        early_stopper = EarlyStopping(patience=5)
                        cur_best = None

                        for epoch in range(args.epochs):
                            e_start = time.time() ; start = time.time()
                            loss_train_iter , cum_loss_train_iter , mape_train_iter , result_append_train_iter = network_epoch(epoch,train_loader,model ,model_level ,optimizer,dataset_train.data_index,config['index_group'],max_log_y, 'train', time_steps , exp = exp)
                            with torch.no_grad():
                                loss_val_iter , cum_loss_val_iter , mape_val_iter , result_append_val_iter = network_epoch(epoch,val_loader,model ,model_level ,optimizer,dataset_val.data_index,config['index_group'],max_log_y, 'val', time_steps , exp = None)
                            is_best = not cur_best or mape_val_iter < cur_best
                            if is_best:
                                cur_best = mape_val_iter
                            save_checkpoint(model.state_dict(), is_best, join(args.modelsave_dir, 'checkpoint.tar'),join(args.modelsave_dir, 'best.tar')) #; print(model.state_dict())
                            with torch.no_grad():
                                loss_test_iter , cum_loss_test_iter , mape_test_iter , result_append_test_iter = network_epoch(epoch,test_loader,model ,model_level ,optimizer,dataset_test.data_index,config['index_group'],max_log_y, 'test', time_steps, exp = None)
                                sys.stdout.write('\r')
                                sys.stdout.write('| level_index: %3d loader_num_index %3d model_num_index: %3d ts %3d Epoch [%3d/%3d] MAPE_VAl: %.2f MAPE_TEST: %.2f Time: %.2f' % (level_index , loader_num_index , model_num_index, ts,  epoch + 1, args.epochs  ,mape_val_iter , mape_test_iter , (time.time() - start)))
                                sys.stdout.flush()
                                model.load_state_dict(torch.load(os.path.join(args.modelsave_dir, 'best.tar')))
                                loss_val , cum_loss_val , mape_val , result_append_val = network_epoch(epoch,val_loader,model ,model_level ,optimizer,dataset_val.data_index,config['index_group'],max_log_y, 'val', time_steps, exp = exp)
                                loss_test , cum_loss_test , mape_test , result_append_test = network_epoch(epoch,test_loader,model ,model_level ,optimizer,dataset_test.data_index,config['index_group'],max_log_y, 'test', time_steps, exp = exp)
                                loss_future , cum_loss_future , mape_future , result_append_future = network_epoch(epoch,future_loader,model ,model_level ,optimizer,dataset_future.data_index,config['index_group'],max_log_y, 'future' , time_steps, exp = exp)

                            scheduler.step(mape_val_iter)
                            early_stopper(mape_val_iter, model)
                            if early_stopper.early_stop:
                                break

                        df_list_val = df_list1[(df_list1['train_test'] == 'val')] ; df_list_val = df_list_val[config['index']].reset_index(drop = True) ;result_append_val = pd.concat([df_list_val ,result_append_val ] , axis = 1) ; result_append_val['Object_MAPE'] = result_append_val.groupby(config['index_group'])['MAPE'].transform(np.mean)
                        df_list_test = df_list1[(df_list1['train_test'] == 'test')] ; df_list_test = df_list_test[config['index']].reset_index(drop = True)   ;result_append_test = pd.concat([df_list_test ,result_append_test ] , axis = 1) ; result_append_test['Object_MAPE'] = result_append_test.groupby(config['index_group'])['MAPE'].transform(np.mean)
                        df_list_future = df_list1[(df_list1['train_test'] == 'future')] ; df_list_future = df_list_future[config['index']].reset_index(drop = True)   ;result_append_future = pd.concat([df_list_future ,result_append_future ] , axis = 1) ; result_append_future['Object_MAPE'] = result_append_future.groupby(config['index_group'])['MAPE'].transform(np.mean)
                        
                        result_append_val['model_type'] = model_num ; result_append_val['loader_type'] = loader_num ; result_append_val['timestep'] = ts ; result_append_val['level'] = level ; result_append_val['objects'] = objects
                        result_append_test['model_type'] = model_num ; result_append_test['loader_type'] = loader_num ; result_append_test['timestep'] = ts ; result_append_test['level'] = level ; result_append_test['objects'] = objects
                        result_append_future['model_type'] = model_num ; result_append_future['loader_type'] = loader_num ; result_append_future['timestep'] = ts ; result_append_future['level'] = level ; result_append_future['objects'] = objects
                        
                        final_results_val = pd.concat([final_results_val , result_append_val], axis=0) 
                        final_results_test = pd.concat([final_results_test , result_append_test], axis=0)
                        final_results_future = pd.concat([final_results_future , result_append_future], axis=0)
    
    final_results_val.to_csv(os.path.join(args.dataexp_dir , 'final_results_val.csv') , index = False)
    final_results_test.to_csv(os.path.join(args.dataexp_dir , 'final_results_test.csv') , index = False)
    final_results_future.to_csv(os.path.join(args.dataexp_dir , 'final_results_future.csv') , index = False)
    print('Total run time:', (time.time() - code_start_time)/60)
