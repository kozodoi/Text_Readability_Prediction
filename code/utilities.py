import os 
import numpy as np
import torch
import random
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error


# competition metric
def get_score(y_true, y_pred):
    score = np.sqrt(mean_squared_error(y_true, y_pred))
    return score


# random sequences
def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)


# device-aware printing
def smart_print(expression, accelerator = None):
    if accelerator is None:
        print(expression)
    else:
        accelerator.print(expression)

        
# randomness
def seed_everything(seed, accelerator = None, silent = False):
    assert isinstance(seed, int), 'seed has to be an integer'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    if not silent:
        smart_print('- setting random seed to {}...'.format(seed), accelerator)
    
    
# torch random fix
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
    
# simple ensembles
def compute_blend(df, preds, blend, CFG, weights = None):
    
    # checks
    if weights is not None:
        assert len(preds) == len(weights), 'Weights and preds are not the same length'
    
    # equal weights
    if weights is None:
        weights = np.ones(len(preds)) / len(preds)
        
    # compute blend
    if blend == 'amean':
        out = np.sum(df[preds].values * weights, axis = 1)
    elif blend == 'median':
        out = df[preds].median(axis = 1)
    elif blend == 'gmean':
        out = np.prod(np.power(df[preds].values, weights), axis = 1)
    elif blend == 'pmean':
        out = np.sum(np.power(df[preds].values, CFG['power']) * weights, axis = 1) ** (1 / CFG['power'])
    elif blend == 'rmean':
        out = np.sum(df[preds].rank(pct = True).values * weights, axis = 1)
        
    return out