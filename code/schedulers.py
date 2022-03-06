from transformers import (  
    get_constant_schedule_with_warmup, 
    get_cosine_schedule_with_warmup, 
    get_linear_schedule_with_warmup,
)



def get_scheduler(CFG, 
                  optimizer):
    
    '''
    Get scheduler
    '''
    
    # tests
    assert isinstance(CFG, dict), 'CFG has to be a dict with parameters'
    
    # constant
    if CFG['scheduler'] == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimzier        = optimizer, 
                                                      num_warmup_steps = CFG['warmup'])
        
    # linear
    if CFG['scheduler'] == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer          = optimizer, 
                                                    num_warmup_steps   = CFG['warmup'], 
                                                    num_training_steps = CFG['num_epochs'])
    
    # cosine annealing
    if CFG['scheduler'] == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer          = optimizer, 
                                                    num_warmup_steps   = CFG['warmup'], 
                                                    num_training_steps = CFG['num_epochs'])

    return scheduler