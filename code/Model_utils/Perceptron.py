from collections import OrderedDict, defaultdict
import os
from os.path import join
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


def Perceptron(in_dim, out_dim, nlayers, mid_dim=None, disc_input=False):
    # with 0 layers it is either embedding or a embedding layer
    if nlayers == 0:
        if disc_input is False:
            return nn.Linear(in_dim,out_dim)
        else:
            return nn.Embedding(in_dim + 1, out_dim, padding_idx=in_dim) #the embedding layer is create with an extra (padded) dimension
    else:
        if mid_dim is None: #if mid_dim is not specified it uses two times the in_dim size
            mid_dim = in_dim * 2
        if disc_input is False: #creates the first layer as linear or embedding with output size mid_dim
            layers = [('linear1', nn.Linear(in_dim, mid_dim))]
        else:
            layers = [('embedding1', nn.Embedding(in_dim + 1, mid_dim, padding_idx=in_dim))]
        for i in range(nlayers - 1): #left-over layers are mid_dim --> mid_dim with a ReLU
            layers.append(('relu{}'.format(i + 1), nn.ReLU()))
            layers.append(('linear{}'.format(i + 2), nn.Linear(mid_dim, mid_dim)))
        layers.append(('relu{}'.format(nlayers), nn.ReLU()))
        #last layer with output of size out_dim
        layers.append(('linear{}'.format(nlayers + 1), nn.Linear(mid_dim, out_dim)))
        return nn.Sequential(OrderedDict(layers))