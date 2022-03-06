import torch.optim as optim
from transformers import AdamW
from adamp import AdamP
from madgrad import MADGRAD



def get_optimizer(CFG, 
                  model):
    
    '''
    Get optimizer
    '''
    
    ##### PREPARATIONS

    # tests
    assert isinstance(CFG, dict), 'CFG has to be a dict with parameters'
    
    # list of layers with no decay
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']
        
    # list of backbone layers
    backbone_layers = ['embeddings'] + ['layer.' + str(l) + '.' for l in range(model.backbone.config.num_hidden_layers)]
    backbone_layers.reverse()
    
    # params in network head
    parameters = [{'params':       [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad],
                   'weight_decay': 0.0,
                   'lr':           CFG['lr'] * 10}]
    
    # params in backbone layers
    layer_lr = CFG['lr']
    for layer in backbone_layers:
        parameters += [{'params':       [p for n, p in model.named_parameters() if layer in n and not any(nd in n for nd in no_decay) and p.requires_grad],
                        'weight_decay': CFG['decay'],
                        'lr':           layer_lr},
                       {'params':       [p for n, p in model.named_parameters() if layer in n and any(nd in n for nd in no_decay) and p.requires_grad],
                        'weight_decay': 0.0,
                        'lr':           layer_lr}]
        layer_lr *= CFG['lr_layer_decay'] 
        
        
    ##### INITIALIZE OPTIMIZER

    if CFG['optim'] == 'Adam':
        optimizer = optim.Adam(parameters, 
                               lr           = CFG['lr'], 
                               weight_decay = CFG['decay'])
    elif CFG['optim'] == 'AdamW':
        optimizer = AdamW(parameters, 
                          lr           = CFG['lr'], 
                          weight_decay = CFG['decay'],
                          correct_bias = CFG['adamw_bias'])
    elif CFG['optim'] == 'AdamP':
        optimizer = AdamP(parameters, 
                          lr           = CFG['lr'], 
                          weight_decay = CFG['decay'])
    elif CFG['optim'] == 'madgrad':
        optimizer = MADGRAD(parameters, 
                            lr           = CFG['lr'], 
                            weight_decay = CFG['decay']) 

    return optimizer