from utilities import *
from data import get_data, get_loaders
from tokenizer import get_tokenizer

import neptune
from accelerate import Accelerator, DistributedType
import torch
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import gc



def sample_batch(CFG, 
                 df, 
                 sample_size = 5, 
                 seq_size    = 15, 
                 batch_idx   = 0):

    '''
    Display sample training and validation batch
    '''
    
    # tests
    assert isinstance(CFG, dict),         'CFG has to be a dict with parameters'
    assert isinstance(df,  pd.DataFrame), 'df has to be a pandas dataframe'
    assert sample_size > 0,               'sample_size has to be positive'
    assert seq_size > 0,                  'seq_size has to be positive'
    assert batch_idx >= 0,                'batch_idx has to be an integer'

    ##### PREPARATIONS

    # initialize accelerator
    accelerator = Accelerator(device_placement = True,
                              fp16             = CFG['use_fp16'],
                              split_batches    = False)
    accelerator.state.device = torch.device('cpu')

    # sample indices
    idx_start = batch_idx * sample_size
    idx_end   = (batch_idx + 1) * sample_size

    # data sample
    df_sample = pd.concat((df.iloc[idx_start:idx_end],
                           df.iloc[idx_start:idx_end]), axis = 0)
    df_sample['fold0'] = np.concatenate((np.zeros(sample_size), np.ones(sample_size)))

    # get data
    df_trn, df_val = get_data(df          = df_sample,
                              rep         = 0,
                              fold        = 0,
                              CFG         = CFG,
                              accelerator = accelerator,
                              silent      = True,
                              debug       = False)

    # get data loaders
    trn_loader, val_loader = get_loaders(df_trn,
                                         df_val,
                                         CFG,
                                         accelerator,
                                         labeled   = True,
                                         return_sd = True,
                                         silent    = True)

    # get tokenizer
    tokenizer = get_tokenizer(CFG)

    # set seed
    seed_everything(CFG['seed'])


    ##### TRAIN DATA

    # display train data
    batch_time = time.time()
    for batch_idx, (inputs, masks, token_type_ids, labels, sds) in enumerate(trn_loader):

        # feedback
        inputs_shape = inputs.shape
        load_time    = time.time() - batch_time

        # save inputs
        train_inputs = inputs
        break


    ##### VALID DATA

    # display valid data
    batch_time = time.time()
    for batch_idx, (inputs, masks, token_type_ids, labels, sds) in enumerate(val_loader):
        break

    # feedback
    print('- loading time: {:.4f} vs {:.4f} seconds'.format(load_time, (time.time() - batch_time)))
    print('- inputs shape: {} vs {}'.format(inputs_shape, inputs.shape))

    # display texts
    print('-' * 100)
    for i in range(len(train_inputs)):
        print('{:<75} | {:>22}'.format(
            ', '.join(str(txt) for txt in train_inputs[i][0:seq_size].tolist()),
            str(labels[i].tolist())))
    print('-' * 100)
    for i in range(len(inputs)):
        print('{:<75} | {:>22}'.format(
            str(tokenizer.decode(inputs[i][0:seq_size])),
            str(labels[i].tolist())))
    print('-' * 100)

    return (inputs, masks, token_type_ids)
