from utilities import *

import numpy as np
import timm
from timm.utils import *
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType



def test_epoch(loader,
               model,
               CFG,
               accelerator):

    '''
    Test epoch
    '''

    ##### PREPARATIONS
    
    # tests
    assert isinstance(CFG, dict), 'CFG has to be a dict with parameters'

    # switch regime
    model.eval()

    # placeholders
    PREDS = []

    # progress bar
    pbar = tqdm(range(len(loader)), disable = not accelerator.is_main_process)


    ##### INFERENCE LOOP

    # loop through batches
    with torch.no_grad():
        for batch_idx, (inputs, masks, token_type_ids) in enumerate(loader):

            # forward pass
            preds = model(inputs, masks, token_type_ids)
            preds = preds['logits'].squeeze(-1)

            # store predictions
            PREDS.append(accelerator.gather(preds).detach().cpu())

            # feedback
            pbar.update()

    # transform predictions
    return np.concatenate(PREDS)
