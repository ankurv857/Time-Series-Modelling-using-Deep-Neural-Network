from glob import glob
import os
import json
from Config.arguments import get_args ; args = get_args()

class Experiment(object):

    def __init__(self, name , log_dir = args.explog_dir):
        self.name = name
        self.log_dir = os.path.abspath(os.path.join(log_dir , name))

        #create viable directory for experiment logs
        if os.path.exists(self.log_dir):
            n = 1
            path = '{}_{}'.format(self.log_dir , n)
            while os.path.exists(path):
                n += 1
                path = '{}_{}'.format(self.log_dir , n)
            self.log_dir = path
        os.makedirs(self.log_dir)

        #Batch-level log cache, flushed to disk at every epoch
        self.logs = []

    def log(self, batch , dataframe, pred_g , g , loss):
        """ Log data by batch. dataframe is a tensor containing a list of dataframe_indices per sample"""
        self.logs.append({
            'batch':batch,
            'dataframe_idxs':dataframe.data.tolist(),
            'pred_g':pred_g.data.tolist(),
            'g':g.data.tolist(),
            'loss':loss,
        }) 

    def save(self,epoch , tag ,model_level , dt):
        """Save list of batch logs for an epoch"""
        f_name = '{}_{}_{}_{}_{}_{}'.format(epoch , tag ,model_level[0],model_level[1] , model_level[2],model_level[3])
        path = os.path.join(self.log_dir, f_name + '.json')
        with open(path, 'w') as f:
            f.write(json.dumps({
                'epoch':epoch,
                'tag':tag,
                'model_level0':model_level[0],
                'model_level1':model_level[1],
                'model_level2':model_level[2],
                'model_level3':model_level[3],
                'batches':self.logs,
                'dt':dt
            }))

        self.logs = [] # clear log cache

    def load(self):
        pass
