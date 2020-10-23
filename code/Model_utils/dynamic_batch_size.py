import pandas as pd
import numpy as np
from Config.arguments import get_args ; args = get_args()


def dynamic_batch_size(data_index):
    num_batches = len(data_index) - 2*args.seq_len + 1
    if num_batches <= 100:
        batch_size = 1
    elif num_batches >= 25600:
        batch_size = 256
    else:
        batch_size = num_batches//100
    return batch_size
