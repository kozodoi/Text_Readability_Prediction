from utilities import *
from augmentations import shuffle_sentences

import timm
from timm.utils import *
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType
import neptune



def train_valid_epoch(trn_loader,
                      val_loader,
                      model,
                      optimizer,
                      scheduler,
                      trn_criterion,
                      val_criterion,
                      accelerator,
                      epoch,
                      CFG,
                      best_val_score,
                      fold,
                      rep,
                      swa_model     = None,
                      swa_scheduler = None):

    '''
    Training and validation epoch
    '''

    ##### PREPARATIONS
    
    # tests
    assert isinstance(CFG,    dict), 'CFG has to be a dict with parameters'
    assert isinstance(rep,    int),  'rep has to be an integer'
    assert isinstance(fold,   int),  'fold has to be an integer'
    
    # switch regime
    model.train()

    # running loss
    trn_loss = AverageMeter()

    # placeholders
    step_scores = []

    # loader length
    len_loader = CFG['max_batches'] if CFG['max_batches'] else len(trn_loader)

    # progress bar
    pbar = tqdm(range(len_loader), disable = not accelerator.is_main_process)


    ##### TRAINING LOOP

    # loop through batches
    for batch_idx, (inputs, masks, token_type_ids, labels, sds) in enumerate(trn_loader):

        # update scheduler on batch
        if CFG['update_on_batch']:
            if not CFG['swa']:
                scheduler.step(epoch + batch_idx / len_loader)
            elif (epoch + 1) < CFG['swa_start']:
                scheduler.step(epoch + batch_idx / len_loader)

                # shuffle sentences
        if CFG['p_shuffle']:
            do_shuffle = np.random.rand(1)
            if do_shuffle < CFG['p_shuffle']:
                inputs = shuffle_sentences(inputs, accelerator)

        # manipulate target
        if CFG['noise_alpha'] > 0:
            noises = (1 - 2 * torch.rand(labels.size(), device = accelerator.device)) * CFG['noise_alpha']
            labels += noises * sds

        # passes and weight updates
        with torch.set_grad_enabled(True):

            # forward pass 
            preds = model(inputs, masks, token_type_ids)
            preds = preds['logits'].squeeze(-1).to(labels.dtype)

            # compute loss
            loss = trn_criterion(preds, labels)
            if CFG['loss_fn'] == 'RMSE':
                loss = torch.sqrt(loss)
            loss = loss / CFG['accum_iter']

            # backward pass
            accelerator.backward(loss)

            # gradient clipping
            if CFG['grad_clip']:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), CFG['grad_clip'])

            # update weights
            if ((batch_idx + 1) % CFG['accum_iter'] == 0) or ((batch_idx + 1) == len_loader):
                optimizer.step()
                optimizer.zero_grad()

        # update loss
        trn_loss.update(loss.item() * CFG['accum_iter'], inputs.size(0))

        # feedback
        pbar.update()

        # clear memory
        del loss, preds
        del inputs, masks, token_type_ids, labels, sds


        ###### NESTED VALIDATION LOOP

        # whether to run validation
        run_validation = False
        if CFG['eval_step']:
            if ((batch_idx + 1) % CFG['eval_step'] == 0) or ((batch_idx + 1) == len_loader):
                run_validation = True
        elif not CFG['eval_step']:
            if ((batch_idx + 1) == len_loader):
                run_validation = True

        # validation loop
        if run_validation:

            # placeholders
            PREDS  = []
            LABELS = []

            # running loss
            val_loss = AverageMeter()

            # switch regime
            model.eval()
            if CFG['swa']:
                swa_model.eval()

            # loop through batches
            with torch.no_grad():
                for val_batch_idx, (inputs, masks, token_type_ids, labels, sds) in enumerate(val_loader):

                    # forward pass
                    if not CFG['swa'] or (epoch + 1) < CFG['swa_start']:
                        preds = model(inputs, masks, token_type_ids)
                    else:
                        preds = swa_model(inputs, masks, token_type_ids)
                    preds = preds['logits'].squeeze(-1).to(labels.dtype)

                    # compute loss
                    loss = val_criterion(preds, labels)
                    val_loss.update(loss.item(), inputs.size(0))

                    # store predictions
                    PREDS.append(accelerator.gather(preds.detach().cpu()))
                    LABELS.append(accelerator.gather(labels.detach().cpu()))

                    # clear memory
                    del loss, preds
                    del inputs, masks, token_type_ids, labels, sds

            # switch regime
            model.train()
            if CFG['swa']:
                swa_model.train()

            # checks
            assert len(LABELS) == len(PREDS),  'Labels and predictions are not the same length'

            # feedback
            step_scores.append(get_score(np.concatenate(LABELS), np.concatenate(PREDS)))
            if CFG['batch_verbose']:
                if (batch_idx > 0) and (batch_idx % CFG['batch_verbose'] == 0):
                    accelerator.print('-- batch {} | lr = {:.6f} | trn_loss = {:.4f} | val_score = {:.4f}'.format(
                        batch_idx, scheduler.state_dict()['_last_lr'][0], trn_loss.avg, step_scores[-1]))
            if CFG['tracking']:
                neptune.send_metric('val_score_{}'.format(int(fold)), step_scores[-1])

            # export weights if global minimum
            if step_scores[-1] <= best_val_score:
                best_val_score  = step_scores[-1]
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), CFG['out_path'] + 'weights_rep{}_fold{}.pth'.format(rep, fold))

            # save loss values if local minimum
            if step_scores[-1] <= min(step_scores):
                best_epoch_val_loss  = val_loss
                best_epoch_val_score = step_scores[-1]

        # early stop
        if (batch_idx == len_loader):
            break

    # update scheduler on epoch
    if not CFG['update_on_batch']:
        if not CFG['swa']:
            scheduler.step()
        elif (epoch + 1) < CFG['swa_start']:
            scheduler.step()

    # update SWA scheduler
    if CFG['swa'] and (epoch + 1) >= args['swa_start']:
        swa_model.update_parameters(model)
        swa_scheduler.step()

    return trn_loss.sum, best_epoch_val_loss.sum, best_epoch_val_score
