#@author - Dunzo

import argparse
def get_args(args_string=None):
    parser = argparse.ArgumentParser(description='AUTO ML for Demand Prediction')
    parser.add_argument('--rawdata-dir', type=str,default='../data/raw', help='folder for storing raw data')
    parser.add_argument('--dataread-dir', type=str,default='../data/prepared', help='folder for storing prepared data')
    parser.add_argument('--dataexp-dir', type=str,default='../data/experiments/exp_1', help='folder for storing experimental data')
    parser.add_argument('--modelsave-dir', type=str,default='../data/modelsave', help='folder for storing models')
    parser.add_argument('--explog-dir', type=str,default='../data/experimentlog', help='folder for storing experiment logs')
    parser.add_argument('--exp-name', type=str,default='', help='name of the experiment')

    #Neural Network
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--rnn-layers', type=int, default=1,help='rnn layers (default: 1)')
    parser.add_argument('--seq-len', type=int, default=3, help='warm_up/prediction window lenght, note that the effective size of the seq_len is going to be 2x')
    args = parser.parse_args(args=args_string)
    return args