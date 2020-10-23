import pandas as pd
import numpy as np
import copy
import datetime
from datetime import time , date 
from dateutil.relativedelta import relativedelta
import torch
import os
from Config.arguments import get_args ; args = get_args()

class data_loader_with_offset():
    def __init__(self,df_list ,dtype_list, granularity, static_list ,seq_delta,cv ,tag ,train_val_date_list ,index):
        self.df_list = df_list ; self.dtype_list = dtype_list ;self.target = self.dtype_list[8] 
        self.granularity = granularity  ; self.seq_delta  = seq_delta ; self.cv =cv ; self.tag = tag
        self.train_val_date_list = train_val_date_list ; self.index = index

        self.dataframe = self._init_emb_dict_(self.df_list) 
        self.dataframe = self._init_emb_insert_([self.dataframe]) ; self.emb_features = self.emb_int 
        self.mlp_static = static_list[0] ; self.emb_static = static_list[1] 
        self.mlp_tmpl = list(set(self.mlp_features) - set(self.mlp_static)) ; self.emb_tmpl = list(set(self.emb_features) - set(self.emb_static))
        self._init_inout_emb_(self.emb_dict) 
        self._init_data_samples_()

    #Strings + Binary + Multiclass(1,3,4) less than equal to 12 are one hot ; Multiclass discontinuous(6) is embedding
    def _init_emb_dict_(self,dataframe):
        emb_dict = {} ; emb_len = {} ; self.mlp_features = []
        for df in dataframe:
            for key in df.keys():
                if key in self.dtype_list[0]:
                    df[key+'_rank_desc'] = df.groupby([self.granularity])[key].rank(ascending=False).astype(int)
                    df[key+'_rank_desc_f'] = df[key+'_rank_desc']/df[key+'_rank_desc'].max()
                    df[key+'_rank_asc'] = df.groupby([self.granularity])[key].rank(ascending=True).astype(int)
                    df[key+'_rank_asc_f'] = df[key+'_rank_asc']/df[key+'_rank_asc'].max()
                    self.mlp_features += [key+'_rank_asc_f']
                
                self.binary_cont = self.dtype_list[3] + self.dtype_list[5] + self.dtype_list[4] + self.dtype_list[6] 
                if key in list(set(self.binary_cont)  - set(['all'])):
                    df[key] = df[key]/df[key].max()
                    self.mlp_features += [key]
                
                self.str_mult_discont = self.dtype_list[1]  + self.dtype_list[4] + self.dtype_list[6] 
                if key in list(set(self.str_mult_discont) - set(['train_test'])):  
                    df[key + '_emb'] = df[key] 
                    emb_dict[key + '_emb'] = np.unique(df[key + '_emb'])
                    emb_len[key + '_emb'] = len(np.unique(df[key + '_emb']))
                    self.emb_dict = emb_dict ; self.emb_len = emb_len  

                if key in self.dtype_list[8]:
                    for lag in [1]:
                        df = df.sort_values([self.granularity , self.dtype_list[0][0] ],ascending=True)
                        df[key + '_lag_mean_p' + str(lag)] = df.groupby([self.granularity])[key].shift(args.seq_len).groupby(df[self.granularity]).rolling(lag*args.seq_len , min_periods = 1).mean().reset_index(level = 0 , drop = True).astype('Float64')
                        df[key + '_lag_mean_s' + str(lag)] = df.groupby([self.granularity])[key].shift(1).groupby(df[self.granularity]).rolling(lag*args.seq_len , min_periods = 1).mean().reset_index(level = 0 , drop = True).astype('Float64')
                        df[key + '_lag_mean' + str(lag)] = np.nan  
                        df[key + '_lag_mean' + str(lag)] = df[key + '_lag_mean' + str(lag)].fillna(df[key + '_lag_mean_p' + str(lag)]).fillna(df[key + '_lag_mean_s' + str(lag)])
                        df[key + '_lag_mean' + str(lag)].fillna(method='bfill', inplace=True)   ; df[key + '_lag_mean' + str(lag)].fillna(0, inplace=True)

                        df[key + '_lag_sd_p' + str(lag)] = df.groupby([self.granularity])[key].shift(args.seq_len).groupby(df[self.granularity]).rolling(lag*args.seq_len , min_periods = 1).std().reset_index(level = 0 , drop = True).astype('Float64')
                        df[key + '_lag_sd_s' + str(lag)] = df.groupby([self.granularity])[key].shift(1).groupby(df[self.granularity]).rolling(lag*args.seq_len , min_periods = 1).std().reset_index(level = 0 , drop = True).astype('Float64')
                        df[key + '_lag_sd' + str(lag)] = np.nan  
                        df[key + '_lag_sd' + str(lag)] = df[key + '_lag_sd' + str(lag)].fillna(df[key + '_lag_sd_p' + str(lag)]).fillna(df[key + '_lag_sd_s' + str(lag)])
                        df[key + '_lag_sd' + str(lag)].fillna(method='bfill', inplace=True)   ; df[key + '_lag_sd' + str(lag)].fillna(0, inplace=True)

                        df[key + '_lag_p' + str(lag)] = df.groupby([self.granularity])[key].shift(args.seq_len).groupby(df[self.granularity]).rolling(1, min_periods = 1).mean().reset_index(level = 0 , drop = True).astype('Float64')
                        df[key + '_lag_s' + str(lag)] = df.groupby([self.granularity])[key].shift(1).groupby(df[self.granularity]).rolling(1 , min_periods = 1).mean().reset_index(level = 0 , drop = True).astype('Float64')
                        df[key + '_lag' + str(lag)] = np.nan  
                        df[key + '_lag' + str(lag)] = df[key + '_lag' + str(lag)].fillna(df[key + '_lag_p' + str(lag)]).fillna(df[key + '_lag_s' + str(lag)])
                        df[key + '_lag' + str(lag)].fillna(method='bfill', inplace=True)   ; df[key + '_lag' + str(lag)].fillna(0, inplace=True)

                        self.mlp_features += [key + '_lag_mean' + str(lag) , key + '_lag_sd' + str(lag) , key + '_lag' + str(lag)]
        return df

    def _init_emb_insert_(self, dataframe):
        self.emb_int = []
        for df in dataframe:
            for key in self.emb_dict.keys():
                if key in df.keys():
                    df[key] = df[key].replace(list(self.emb_dict.get(key)) , list(range(len(self.emb_dict.get(key)))))
                    self.emb_int += [key] 
            return df
    

    def _init_inout_emb_(self,emb_dict):
        self.emb_static_list_inp = [] ; self.emb_static_list_out = [] ; self.emb_tmpl_list_inp = [] ; self.emb_tmpl_list_out = [] 
        self.emb_static_inp_cnt = 0 ; self.emb_tmpl_inp_cnt = 0
        for key in emb_dict.keys():
            if key in self.emb_static:
                if len(emb_dict[key]) > 100:
                    self.emb_static_list_inp.append(len(emb_dict[key]))
                    self.emb_static_inp_cnt += 1
                    self.emb_static_list_out.append(50) 
                else:
                    self.emb_static_list_inp.append(len(emb_dict[key]))
                    self.emb_static_inp_cnt += 1
                    self.emb_static_list_out.append((len(emb_dict[key])//2) + 1 )
            if key in self.emb_tmpl:
                if len(emb_dict[key]) > 100:
                    self.emb_tmpl_list_inp.append(len(emb_dict[key]))
                    self.emb_tmpl_inp_cnt += 1
                    self.emb_tmpl_list_out.append(50)
                else:
                    self.emb_tmpl_list_inp.append(len(emb_dict[key]))
                    self.emb_tmpl_inp_cnt += 1
                    self.emb_tmpl_list_out.append((len(emb_dict[key])//2) + 1 )

    def _init_data_samples_(self):
        """Construct the mapping of sampleable isd's to index in dataframe df.
        Each sample loads temporal data between (day - seq_len) and  (day + seq_len), so it's important that we
        ensure no overlap in days observed between the train and test set.
        """
        self.dataframe = copy.deepcopy(self.dataframe) ; og_idxs = self.dataframe.index

        samples = copy.deepcopy(self.dataframe)
        samples['idx'] = samples.index
        samples = samples.loc[(samples[self.dtype_list[0][0] +  '_rank_asc'] >= args.seq_len + 1 ) & (samples[self.dtype_list[0][0] + '_rank_desc'] >= args.seq_len)]

        if self.tag == 'train':
            # Each sample will be predicted seq_len out from sample day, so ensure there's no overlap between
            samples = samples.loc[(samples['train_test'] == 'train') & (samples[self.dtype_list[0][0] + '_rank_desc'] >= 4*args.seq_len)] #; print('sample_train' , samples)
        elif self.tag == 'val':
            samples = samples.loc[(samples['train_test'] == 'val') & (samples[self.dtype_list[0][0] + '_rank_desc'] == 3*args.seq_len)] #; print('sample_test' , samples)
        elif self.tag == 'test':
            samples = samples.loc[(samples['train_test'] == 'test') & (samples[self.dtype_list[0][0] + '_rank_desc'] == 2*args.seq_len)] #; print('sample_test' , samples)
        elif self.tag == 'future':
            samples = samples.loc[(samples['train_test'] == 'future') & (samples[self.dtype_list[0][0] + '_rank_desc'] == args.seq_len)] #; print('sample_test' , samples)

        assert len(samples) <= len(og_idxs)

        self.samples = samples['idx'].tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self , j):
        dataframe_idx = self.samples[j]  
        data_batch = self.dataframe.ix[dataframe_idx - self.seq_delta : dataframe_idx + self.seq_delta-1] 
        self.data_index = data_batch[self.index].reset_index(drop=True)
        mlp_static_data = np.zeros((2*self.seq_delta, len(self.mlp_static)) , float)
        mlp_tmpl_data = np.zeros((2*self.seq_delta , len(self.mlp_tmpl)) , float)
        emb_static_data = np.zeros((2*self.seq_delta , len(self.emb_static)) , int)
        emb_tmpl_data = np.zeros((2*self.seq_delta, len(self.emb_tmpl)) , int)
        target_data = np.zeros(2*self.seq_delta, float)
        self.mlp_static_id = 0 ; self.mlp_tmpl_id = 0 ; self.emb_static_id = 0 ; self.emb_tmpl_id = 0
        for key in data_batch.keys():
            if key in self.target:
                target_data[:] = data_batch[key]
            if key in self.mlp_static:
                mlp_static_data[ :, self.mlp_static_id] = data_batch[key]
                self.mlp_static_id += 1
            if key in self.mlp_tmpl:
                mlp_tmpl_data[ :, self.mlp_tmpl_id] = data_batch[key]
                self.mlp_tmpl_id += 1
            if key in self.emb_static:
                emb_static_data[ :, self.emb_static_id] = data_batch[key]
                self.emb_static_id += 1
            if key in self.emb_tmpl:
                emb_tmpl_data[: , self.emb_tmpl_id] = data_batch[key]
                self.emb_tmpl_id += 1
        target_data = torch.tensor(target_data , dtype=torch.float32)
        mlp_static_data = torch.tensor(mlp_static_data , dtype=torch.float32)
        mlp_tmpl_data = torch.tensor(mlp_tmpl_data , dtype=torch.float32)
        emb_static_data = torch.LongTensor(emb_static_data)
        emb_tmpl_data = torch.LongTensor(emb_tmpl_data)
        dataframe_idxs = list(range(dataframe_idx- self.seq_delta , dataframe_idx + self.seq_delta-1)) ; dataframe_idxs = torch.tensor(dataframe_idxs)
        return target_data , mlp_static_data, mlp_tmpl_data , emb_static_data, emb_tmpl_data , dataframe_idxs