from utilities import *
from model import get_model
from data import get_data, get_loaders
from test_epoch import test_epoch

import gc
import neptune
from accelerate import Accelerator, DistributedType
import pandas as pd
import numpy as np



def run_inference(df, 
                  df_test, 
                  CFG):
    
    '''
    Run inference loop
    '''

    # tests
    assert isinstance(CFG, dict),         'CFG has to be a dict with parameters'
    assert isinstance(df,  pd.DataFrame), 'df has to be a pandas dataframe'
    
    # placeholders
    oof = None
    sub = None
    
    # inference loop
    for rep in range(CFG['num_reps']):
        for fold in range(CFG['num_folds']):
            
            # initialize accelerator
            accelerator = Accelerator(device_placement = True,
                                      fp16             = CFG['use_fp16'],
                                      split_batches    = False)
            if CFG['device'] == 'GPU':
                accelerator.state.device = torch.device('cuda:{}'.format(CFG['device_index']))

            # feedback
            accelerator.print('-' * 55)
            accelerator.print('REP {:d}/{:d} | FOLD {:d}/{:d}'.format(rep + 1, CFG['num_reps'], fold + 1, CFG['num_folds']))
            accelerator.print('-' * 55)   

            # get data
            df_trn, df_val = get_data(df, rep, fold, CFG, accelerator, silent = True)  

            # get test loader
            _, val_loader  = get_loaders(df_trn, df_val,  CFG, accelerator, labeled = False, return_sd = False, silent = True) 
            _, test_loader = get_loaders(df_trn, df_test, CFG, accelerator, labeled = False, return_sd = False, silent = True) 

            # prepare model
            model = get_model(CFG, pretrained = CFG['out_path'] + 'weights_rep{}_fold{}.pth'.format(int(rep), int(fold)))

            # handle device placement
            model, val_loader, test_loader = accelerator.prepare(model, val_loader, test_loader)

            # inference for validation data
            if CFG['predict_oof']:

                # produce OOF preds
                val_preds = test_epoch(loader      = val_loader, 
                                       model       = model,
                                       CFG         = CFG,
                                       accelerator = accelerator)

                # store OOF preds
                val_preds_df = pd.DataFrame(val_preds, columns = ['pred_rep' + str(rep)])
                val_preds_df = pd.concat([df_val, val_preds_df], axis = 1)
                oof          = pd.concat([oof,    val_preds_df], axis = 0).reset_index(drop = True)

            # inference for test data
            if CFG['predict_test']:

                # produce test preds
                test_preds = test_epoch(loader      = test_loader, 
                                        model       = model,
                                        CFG         = CFG,
                                        accelerator = accelerator)

                # store test preds
                test_preds_df = pd.DataFrame(test_preds, columns = ['pred_rep_{}_fold{}'.format(int(rep), int(fold))])
                sub           = pd.concat([sub, test_preds_df], axis = 1)

            # clear memory
            del model, val_loader, test_loader
            del accelerator
            gc.collect()
        
    # aggregate and export OOF preds
    if CFG['predict_oof']:
        for rep in range(CFG['num_reps']):
            if rep == 0:
                oof_all = oof.loc[-oof['pred_rep' + str(rep)].isnull()].reset_index(drop = True)
            else:
                del oof_all['pred_rep' + str(rep)]
                oof_rep = oof.loc[-oof['pred_rep' + str(rep)].isnull()].reset_index(drop = True)[['id', 'pred_rep' + str(rep)]]
                oof_all = oof_all.merge(oof_rep, how = 'left', on = 'id')
                oof_all.to_csv(CFG['out_path'] + 'oof.csv', index = False, float_format = '%.10f')
        if CFG['tracking']:
            neptune.send_artifact(CFG['out_path'] + 'oof.csv')
            
    # export test preds
    if CFG['predict_test']:
        sub = pd.concat([df_test['id'], sub], axis = 1)
        sub.to_csv(CFG['out_path'] + 'submission.csv', index = False, float_format = '%.10f')
        if CFG['tracking']:
            neptune.send_artifact(CFG['out_path'] + 'submission.csv')
            
    # checks
    assert len(oof_all) == len(df),  'OOF predictions and training data are not the same length'
    assert len(sub) == len(df_test), 'Test predictions and test data are not the same length'