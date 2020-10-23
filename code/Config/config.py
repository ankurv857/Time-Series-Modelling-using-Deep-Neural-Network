import numpy as np
import datetime
from datetime import time , date
import time
import os
from Config.arguments import get_args ; args = get_args()

config = {}
config['data_list_dataprep'] = [('raw_data.csv', ['DC','SKU_ID'] ,'Date' , 
['DC',	'Date'] , "%Y-%m-%d" ,['all','map', 'Date', 'DC', 'SKU_ID' ,'Vol','train_test'])]

config['target1'] = 'Vol'
config['aggregation_list'] = []
config['causal_columns1'] = []
config['causal_columns2'] = []

#For the class data_read()
config['data_list_dataread'] = ['data.csv']  
config['date_list'] = ['Date'] 
config['target'] = ['Vol']  
config['idx'] = []  
config['multiclass_discontinuous'] = [] 
config['text'] = []  
config['remove_list'] = []

#For the classes of dataloaders
config['granularity'] = 'map' 
config['index'] = ['map','DC','SKU_ID' , 'Date']  
config['index_group'] = ['map','DC','SKU_ID']
config['level_list'] = ['all']
config['loader_class_list'] = ['data_loader_with_offset']
config['model_class_list'] = ['Neural_Network_lstmfused' , 'Neural_Network_wavenet' , 'Neural_Network_cnnlstm' , 'Neural_Network_mlp', 'Neural_Network_lstmcell']
config['timestep_list'] = [1,args.seq_len]
config['final_cols'] = ['map', 'DC', 'SKU_ID','Date' ,'actual', 'predicted', 'MAPE', 'Object_MAPE','model_type', 'loader_type', 'timestep', 'level','objects']
config['train_val_date_list'] =  ['Date' , datetime.date(2016 , 1 ,1) , datetime.date(2019 , 3 ,1), datetime.date(2019 , 4 ,1) , datetime.date(2019 , 6 ,1)]  
config['static_list'] = [[] , ['map_emb' , 'DC_emb' , 'SKU_ID_emb']]