import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from Config.arguments import get_args ; args = get_args()

def loss_func_seq(pred_g, g, warm_up, time_steps):
    """should take prediction of size 2xseq_len-1,1 and  g of size 2xseq_len,1
     and compute the mean squared error note that we are only doing next_step predictions"""
    #extracts the predictions of warm_up+1 till end and flattens them, note that predictions are done at the previous step
    #so :,warm_up,: stores the predicted value for warm_up+1
    # y_hat = pred_g[:, warm_up - time_steps : - time_steps].contiguous().view(-1)
    y_hat = pred_g[:, warm_up:].contiguous().view(-1)
    #extracts the ground truth targets from warm_up+1 till the end and flattens them
    y = g[:, warm_up:].contiguous().view(-1)
    
    # computes loss for predictions from network
    epsilon = 0.0001 #; print('loss_func_seq')
    # loss = torch.mean((abs(y_hat - y))/(y + epsilon))
    # loss = torch.mean((2*abs(y_hat - y))/(y_hat + y + epsilon))
    # loss = F.mse_loss(y_hat, y, reduction='mean') #; print('loss' , loss)
    loss = F.l1_loss(y_hat, y, reduction='mean') #; print('loss' , loss)
    return loss


def loss_func_non_seq(pred_g, g, warm_up):
    """should take prediction of size 2xseq_len-1,1 and  g of size 2xseq_len,1
     and compute the mean squared error note that we are only doing next_step predictions"""
    #extracts the predictions of warm_up+1 till end and flattens them, note that predictions are done at the previous step
    #so :,warm_up,: stores the predicted value for warm_up+1
    y_hat = pred_g[:, warm_up:].contiguous().view(-1)
    #extracts the ground truth targets from warm_up+1 till the end and flattens them
    y = g[:, warm_up:].contiguous().view(-1)
    #computes loss for predictions from network
    epsilon = 0.0001 #; print('loss_func_non_seq')
    # loss = torch.mean((abs(y_hat - y))/(y + epsilon))
    # loss = torch.mean((2*abs(y_hat - y))/(y_hat + y + epsilon))
    # loss = F.mse_loss(y_hat, y, reduction='mean')
    loss = F.l1_loss(y_hat, y, reduction='mean')
    return loss