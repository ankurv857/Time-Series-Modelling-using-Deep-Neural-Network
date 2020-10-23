#@author - Dunzo

import argparse
def get_args(args_string=None):
    parser = argparse.ArgumentParser(description='AUTO ML for Demand Prediction')
    parser.add_argument('--results-dir', type=str,default='../exp_26', help='folder for storing results data')
    parser.add_argument('--consol-dir', type=str,default='../exp_26', help='folder for storing consloidated results data')
    args = parser.parse_args(args=args_string)
    return args
