import pandas as pd
import numpy as np
import numpy
import os
from os.path import join, exists
import glob
import sys
import warnings
import copy
warnings.filterwarnings("ignore")
import datetime
from datetime import time , date
from datetime import datetime
from arguments import get_args ; args = get_args()


if __name__ == '__main__':
    print('YO!!! The Consolidation of Deep Learning models begins!')

    index = ['map' , 'model_type',	'loader_type',	'timestep',	'level']
    
    final_results_future = pd.read_csv(os.path.join(args.results_dir , 'final_results_future.csv'))
    final_results_test = pd.read_csv(os.path.join(args.results_dir , 'final_results_test.csv'))
    final_results_val = pd.read_csv(os.path.join(args.results_dir , 'final_results_val.csv'))
    
    #Impute NA values to zero
    final_results_future['MAPE'].fillna(0,inplace = True)
    final_results_test['MAPE'].fillna(0,inplace = True)
    final_results_val['MAPE'].fillna(0,inplace = True)
    

    #concat mapes of val and test to calculate best mape from the average predictions of both
    best_mape_cal = pd.concat([final_results_test ,final_results_val ] , axis = 0).reset_index(drop = True) 
    best_mape_cal = best_mape_cal[['map' , 'model_type',	'loader_type',	'timestep',	'level' , 'MAPE' ,'Object_MAPE']].reset_index(drop = True) 
    best_mape_cal['avg_mape'] = best_mape_cal.groupby(index)['MAPE'].transform(np.mean) 
    best_mape_cal['std_mape'] = best_mape_cal.groupby(index)['MAPE'].transform(np.std) 
    best_mape_cal['ranker'] = best_mape_cal['avg_mape'] + best_mape_cal['std_mape']
    best_mape_cal['Best_Predictions'] = best_mape_cal.groupby(['map'])['ranker'].rank(method= 'min', ascending=True).astype(int) 
    best_mape_cal = best_mape_cal[['map' , 'model_type',	'loader_type',	'timestep',	'level' , 'avg_mape'	,'std_mape'	,'ranker'	,'Best_Predictions']].reset_index(drop = True) 
    best_mape_cal = best_mape_cal[(best_mape_cal['Best_Predictions'] < 65 + 1)]
    best_mape_cal = best_mape_cal.drop_duplicates()

    best_mape_cal.to_csv(os.path.join(args.consol_dir , 'best_mape_cal.csv') , index = False)


    final_results_future['mape_future'] = final_results_future['Object_MAPE']  
    final_results_future = final_results_future[['map' , 'model_type',	'loader_type',	'timestep',	'level','Date' , 'actual',	'predicted'	,'MAPE' , 'mape_future']]
    final_results_test['mape_test'] = final_results_test['Object_MAPE']  
    final_results_test = final_results_test[['mape_test']]
    final_results_val['mape_val'] = final_results_val['Object_MAPE']  
    final_results_val = final_results_val[['mape_val']]

    # # #concat future data with the object mapes of val and test
    all_combined = pd.concat([final_results_future ,final_results_val ,final_results_test] , axis = 1) 

    # #merge the fufure mape cal data with the best mape cal
    consolidation = pd.merge(all_combined , best_mape_cal , on = ['map' , 'model_type',	'loader_type',	'timestep',	'level'] , how = 'inner')
    consolidation = consolidation.sort_values(['map', 'Date'] , ascending = True)  ; print('consolidation' , consolidation.head(2))
    consolidation['predicted'] = np.where(consolidation['ranker'] == 100 , 0 , consolidation['predicted'])
    result = consolidation.groupby(['map','Date']).agg({'actual' : 'mean' , 'predicted' : 'mean'}).reset_index() ; print('result1' , result.head(2))
    result['MAPE'] =  abs((result['actual'] - result['predicted'])/(result['actual']))* 100 
    result['MAPE'] = np.where(result['MAPE'] == np.inf , 100, result['MAPE'])
    result['MAPE'].fillna(0,inplace = True)
    print('result2' , result.head(2))

    result['abs_error'] =  abs((result['actual'] - result['predicted']))
#    result['BP'] , result['Cluster.Code'] = result['map'].str.split('_split_', 1).str
#    result['tdp_of_year'] = result.apply(lambda row: str(row['Date'].timetuple().tm_yday//10).zfill(2), axis = 1)
#    result['Year'] = result.apply(lambda row: row['Date'].year, axis =1)
#    result['TDP'] = result['Year'].astype(str) + result['tdp_of_year']
    
    weighted_mape = np.sum(result['actual']*result['MAPE'])/np.sum(result['actual'])
    abs_error = sum(result['abs_error'])/sum(result['actual'])*100
    print('average_mape',result['MAPE'].mean() , 'weighted_mape' , weighted_mape , 'absolute_error' , abs_error)

    consolidation.to_csv(os.path.join(args.consol_dir , 'consolidation.csv') , index = False)
    result.to_csv(os.path.join(args.consol_dir , 'result.csv') , index = False)

