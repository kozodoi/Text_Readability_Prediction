import torch
import torch.nn as nn
import torch.functional as T


def get_losses(CFG):
    
    '''
    Get loss function
    '''
    
    # tests
    assert isinstance(CFG, dict), 'CFG has to be a dict with parameters'

    # define training loss
    if CFG['loss_fn'] == 'MSE' or CFG['loss_fn'] == 'RMSE':
        train_criterion = torch.nn.MSELoss()
        
    # define valid loss
    valid_criterion = torch.nn.MSELoss()

    return train_criterion, valid_criterion