from utilities import *
from model import get_model
from train_fold import train_fold
from data import get_data
from plot_results import plot_results

import gc
import neptune
from accelerate import Accelerator, DistributedType
import pandas as pd
import numpy as np
import torch



def run_training(df, 
                 CFG):
    
    '''
    Run cross-validation loop
    '''
    
    # tests
    assert isinstance(CFG, dict),         'CFG has to be a dict with parameters'
    assert isinstance(df,  pd.DataFrame), 'df has to be a pandas dataframe'
    
    # placeholder
    oof_score = []
    
    # outer repetitions loop
    for rep in range(CFG['num_reps']):
        
        # placeholder
        rep_oof_score = []

        # cross-validation loop
        for fold in range(CFG['num_folds']):
            
            # initialize accelerator
            accelerator = Accelerator(device_placement = True,
                                      fp16             = CFG['use_fp16'],
                                      split_batches    = False)
            if CFG['num_devices'] == 1 and CFG['device'] == 'GPU':
                accelerator.state.device = torch.device('cuda:{}'.format(CFG['device_index']))

            # feedback
            accelerator.print('-' * 55)
            accelerator.print('REP {:d}/{:d} | FOLD {:d}/{:d}'.format(
                rep + 1, CFG['num_reps'], fold + 1, CFG['num_folds']))    
            accelerator.print('-' * 55) 

            # update seed
            seed_everything(CFG['seed'] + fold, accelerator, silent = True)
            
            # get model
            model = get_model(CFG)

            # get data
            df_trn, df_val = get_data(df, rep, fold, CFG, accelerator)  

            # run single fold
            trn_losses, val_losses, val_scores = train_fold(rep         = rep,
                                                            fold        = fold, 
                                                            df_trn      = df_trn,
                                                            df_val      = df_val, 
                                                            CFG         = CFG, 
                                                            model       = model, 
                                                            accelerator = accelerator)
            rep_oof_score.append(np.min(val_scores))

            # feedback
            accelerator.print('-' * 55)
            accelerator.print('Best: score = {:.4f} (epoch {})'.format(
                np.min(val_scores), np.argmin(val_scores) + 1))
            accelerator.print('-' * 55)

            # plot loss dynamics
            if accelerator.is_local_main_process:
                plot_results(trn_losses, val_losses, val_scores, rep, fold, CFG)      
                
            # clear memory
            del accelerator
            gc.collect()
            
        # save performance
        oof_score.append(np.mean(rep_oof_score))
        if CFG['tracking']:
            neptune.send_metric('oof_score_{}'.format(int(rep)), np.mean(rep_oof_score))

        # feedback
        print('-' * 55)
        print('REP {:d}/{:d} | OOF score = {:.4f}'.format(
                rep + 1, CFG['num_reps'], oof_score[-1]))    
        print('-' * 55)
        print('')
        
    # feedback
    print('-' * 55)
    print('Mean OOF score = {:.4f}'.format(np.mean(oof_score)))
    print('-' * 55)
