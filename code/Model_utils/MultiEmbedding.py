from collections import OrderedDict, defaultdict
import os
from os.path import join
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np



class MultiEmbedding(nn.Module):
    """Module containing multiple embedding layers of different input-output size
    expects two lists of integers for input/output sizes. sequenced=True if inputs to module are sequences.
    First dimmension of input (x) must always be batch_size.
    Sequenced: input (batch_size, sample_len, num_features)
    Non-sequenced: input (batch_size, num_features)
    """
    def __init__(self, in_sizes, out_sizes, sequenced=False):
        super().__init__()
        self.sequenced = sequenced
        self.emb =  nn.ModuleList([])
        #loops through the list of input/output sizes and create a new embedding layer of the respective size
        for in_s, out_s in zip(in_sizes, out_sizes):
            self.emb.append(nn.Embedding(in_s, out_s))
            #print('self.emb',self.emb)

    def forward(self, x):
        #x batch_size,num_embeddings

        # Check input
        if self.sequenced:
            assert len(x.size()) == 3
        else:
            assert len(x.size()) == 2

        emb_list=[]
        # Loops through the feature dimmension of x and applies the ith embedding to the ith feature
        for i in range(x.size()[-1]):
            if len(x.size()) == 2: 
                emb_list.append(self.emb[i](x[:, i])) #; print('emb_list2', emb_list , np.shape(emb_list))
            elif len(x.size()) == 3:
                emb_list.append(self.emb[i](x[:, :, i])) #; print('emb_list3', emb_list ,np.shape(emb_list) )
                #print('emb_list_seq',np.shape(emb_list))
        #print('lenght of the tensor' , len(x.size()))
        return torch.cat(emb_list, dim=len(x.size()) - 1)