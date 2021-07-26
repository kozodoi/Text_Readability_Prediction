import torch
import transformers
from transformers import AutoTokenizer



def get_tokenizer(CFG):
    
    '''
    Get tokenizer
    '''

    tokenizer = AutoTokenizer.from_pretrained(CFG['backbone'])
    
    return tokenizer