import torch
import transformers
from transformers import AutoTokenizer



def get_tokenizer(CFG):
    
    '''
    Get tokenizer
    '''
    
    # tests
    assert isinstance(CFG, dict), 'CFG has to be a dict with parameters'

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG['backbone'])
    return tokenizer