from utilities import *
from data import get_loaders
from optimizers import get_optimizer
from schedulers import get_scheduler
from losses import get_losses
from train_valid_epoch import train_valid_epoch

import torch
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import time
import gc
import neptune
from accelerate import Accelerator, DistributedType
import pandas as pd



def train_fold(rep, 
               fold, 
               df_trn, 
               df_val, 
               CFG, 
               model, 
               accelerator):
    
    '''
    Run training and validation on a single fold
    '''

    ##### PREPARATIONS
    
    # tests
    assert isinstance(CFG,    dict),         'CFG has to be a dict with parameters'
    assert isinstance(df_trn, pd.DataFrame), 'df_trn has to be a pandas dataframe'
    assert isinstance(df_val, pd.DataFrame), 'df_val has to be a pandas dataframe'
    assert isinstance(rep,    int),          'rep has to be an integer'
    assert isinstance(fold,   int),          'fold has to be an integer'

    # update seed
    seed_everything(CFG['seed'] + rep*10 + fold*100, accelerator, silent = True)
    
    # get data loaders
    trn_loader, val_loader = get_loaders(df_trn, df_val, CFG, accelerator)
        
    # get optimizer 
    optimizer = get_optimizer(CFG, model)
    
    # handle device placement
    model, optimizer, trn_loader, val_loader = accelerator.prepare(model, optimizer, trn_loader, val_loader)
    
    # get scheduler
    scheduler = get_scheduler(CFG, optimizer)
    
    # get losses
    trn_criterion, val_criterion = get_losses(CFG)
    
    # stochastic weight averaging
    if CFG['swa']:
        swa_model     = AveragedModel(model)
        swa_model     = accelerator.prepare(swa_model)
        swa_scheduler = SWALR(optimizer       = optimizer, 
                              swa_lr          = CFG['swa_learning_rate'], 
                              anneal_epochs   = CFG['anneal_epochs'], 
                              anneal_strategy = CFG['anneal_strategy'])
        
    # placeholders
    trn_losses = []
    val_losses = []
    val_scores = []
    lrs        = []

    
    ##### TRAINING AND INFERENCE

    for epoch in range(CFG['num_epochs']):
                        
        # timer
        epoch_start = time.time()
        
        # update seed
        seed_everything(CFG['seed'] + rep*10 + fold*100 + epoch*1000, accelerator, silent = True)

        # training and validation
        accelerator.wait_for_everyone()    
        trn_loss, val_loss, val_score = train_valid_epoch(trn_loader     = trn_loader, 
                                                          val_loader     = val_loader, 
                                                          model          = model, 
                                                          optimizer      = optimizer, 
                                                          scheduler      = scheduler,
                                                          trn_criterion  = trn_criterion, 
                                                          val_criterion  = val_criterion, 
                                                          accelerator    = accelerator,
                                                          epoch          = epoch,
                                                          CFG            = CFG,
                                                          best_val_score = 1 if epoch == 0 else min(val_scores),
                                                          fold           = fold,
                                                          rep            = rep,
                                                          swa_model      = None if not CFG['swa'] else swa_model,
                                                          swa_scheduler  = None if not CFG['swa'] else swa_scheduler)
        
        # save LR and losses
        accelerator.wait_for_everyone()
        lrs.append(scheduler.state_dict()['_last_lr'][0])
        trn_losses.append(trn_loss / len(df_trn) * CFG['num_devices'])
        val_losses.append(val_loss / len(df_val) * CFG['num_devices'])        
        val_scores.append(val_score)
        
        # stochastic weight averaging
        if CFG['swa']:
            update_bn(trn_loader, swa_model, device = accelerator.device)
        
        # feedback
        accelerator.wait_for_everyone()
        accelerator.print('-- epoch {}/{} | lr = {:.6f} | trn_loss = {:.4f} | val_loss = {:.4f} | val_score = {:.4f} | {:.2f} min'.format(
            epoch + 1, CFG['num_epochs'], lrs[epoch],
            trn_losses[epoch], val_losses[epoch], val_scores[epoch],
            (time.time() - epoch_start) / 60))
        
        # send performance to Neptune
        if CFG['tracking'] and accelerator.is_local_main_process:
            neptune.send_metric('trn_lr_{}'.format(int(fold)),   lrs[epoch])
            neptune.send_metric('trn_loss_{}'.format(int(fold)), trn_losses[epoch])
            neptune.send_metric('val_loss{}'.format(int(fold)),  val_losses[epoch])
            
    # clear memory
    del model, optimizer, scheduler, trn_loader, val_loader, trn_criterion, val_criterion
    del trn_loss, val_loss, val_score
    
    return trn_losses, val_losses, val_scores