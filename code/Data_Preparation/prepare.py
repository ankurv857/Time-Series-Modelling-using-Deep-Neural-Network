import pandas as pd
import numpy as np
import os
import datetime
from datetime import time , date
import time
from dateutil.relativedelta import relativedelta
import copy
from Config.arguments import get_args ; args = get_args()

class data_prep():
	def __init__(self,data_list , target , aggregation_list , causal_columns1 , causal_columns2 ):
		self.target = target ; self.aggregation_list = aggregation_list ; self.causal_columns1 = causal_columns1 ; self.causal_columns2 = causal_columns2
		self._init_read_(data_list)


	def _init_read_(self,data_list):
		for df , level , granularity, columns , date_format , reader_columns in data_list:
			data = pd.read_csv(os.path.join(args.rawdata_dir ,df) , na_values = ' ', low_memory=False)
			raw_data = copy.deepcopy(data)
			data.columns = columns 
			data[self.target] = data[self.target].str.replace(',', '')   
			data[granularity] = data[granularity].astype(str).str[:4].astype(str) + '-' + data[granularity].astype(str).str[4:].astype(str) + '-' + '01' 
			data[self.target] = data[self.target].astype(float)
			data[granularity] = pd.to_datetime(data[granularity] , format = date_format)
			data['map'] = data[level[0]].astype(str) +'_split_' +data[level[1]].astype(str) 

			raw_data.columns = columns
			raw_data[self.target] = raw_data[self.target].str.replace(',', '')   
			raw_data[granularity] = raw_data[granularity].astype(str).str[:4].astype(str) + '-' + raw_data[granularity].astype(str).str[4:].astype(str) + '-' + '01'
			raw_data[self.target] = raw_data[self.target].astype(float)
			raw_data[granularity] = pd.to_datetime(raw_data[granularity] , format = date_format)
			raw_data['map'] = raw_data[level[0]].astype(str) +'_split_' +raw_data[level[1]].astype(str) 
			
			data = data.groupby(['map',granularity]).agg(np.sum).reset_index()[[self.target ,'map' ,granularity]] 
			data['map_count'] = data.groupby(['map'])[granularity].transform('size')
			data =  data[(data['map_count'] >= 15)] ; print('Total selected combinations = ',data['map'].nunique())
			del data['map_count']

			data_resampled = pd.DataFrame(columns= [granularity,self.target,'map','train_test'])

			for objects in data['map'].unique():
				base_df = data[(data['map'] == objects)]
				base_df = base_df.sort_values(['map', granularity] , ascending = True)

				base_df = base_df.set_index([granularity]) ; base_df.index = pd.DatetimeIndex(base_df.index) 
				base_df_res = base_df.resample('MS').sum().reset_index()
				base_df_res['map'] = objects

				base_df_res['map_rank'] = base_df_res.groupby(['map'])[granularity].rank(ascending=False).astype(int)
				def tag_alloc(x):
					
					if x >  3*args.seq_len:
						tag = 'train'
					elif (x >  2*args.seq_len)&(x <= 3*args.seq_len) :
						tag = 'val'
					elif (x >  1*args.seq_len)&(x <= 2*args.seq_len) :
						tag = 'test'
					else :
						tag = 'future'	
					return tag

				base_df_res['train_test'] = base_df_res.apply(lambda row: tag_alloc(row['map_rank']), axis=1)
				base_df_res = base_df_res[[granularity,self.target,'map','train_test']]
				data_resampled =  data_resampled.append(base_df_res, ignore_index=True) 

			data_resampled[level[0]] , data_resampled[level[1]] = data_resampled['map'].str.split('_split_', 1).str 

			data_resampled[self.target] = np.where(data_resampled[self.target] < 0,0 ,data_resampled[self.target]) ; data_resampled.fillna(0 , inplace = True)

			data_resampled['all'] = 1
			data_resampled = data_resampled[reader_columns]
			data_resampled.to_csv(os.path.join(args.dataread_dir , 'data.csv') , index = False)
		
