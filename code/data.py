from utilities import *
from tokenizer import get_tokenizer

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import transformers
import numpy as np
import pandas as pd



class TextData(Dataset):
    
    '''
    Text dataset class
    '''
    
    def __init__(self, 
                 df, 
                 tokenizer,
                 max_len,
                 padding,
                 p_translate = 0,
                 labeled     = True,
                 return_sd   = False):
        self.df          = df
        self.tokenizer   = tokenizer
        self.max_len     = max_len
        self.padding     = padding
        self.p_translate = p_translate
        self.labeled     = labeled
        self.return_sd   = return_sd
        self.languages   = ['excerpt_de_en', 'excerpt_es_en', 'excerpt_fr_en', 'excerpt_it_en', 'excerpt_pt_en', 'excerpt_ru_en', 'excerpt_tr_en']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        # import text
        if self.p_translate == 0:
            text = self.df.loc[idx, 'excerpt']
        else: 
            do_translate = np.random.rand(1)
            if do_translate < self.p_translate:
                language_id = np.random.randint(7)
                text = self.df.loc[idx, self.languages[language_id]]
            else:
                text = self.df.loc[idx, 'excerpt']
        
        # tokenize text
        text = self.tokenizer(text                  = text,
                              truncation            = True,
                              add_special_tokens    = True, 
                              max_length            = self.max_len,
                              padding               = self.padding,
                              return_token_type_ids = True,
                              return_attention_mask = True)
        
        # labels
        if self.labeled:
            text['labels'] = self.df.loc[idx, 'target']
            
        # SDs
        if self.return_sd:
            text['sds'] = self.df.loc[idx, 'standard_error']
            
        # output
        return text
    
    

def get_data(df, 
             rep, 
             fold, 
             CFG, 
             accelerator, 
             debug  = None, 
             silent = False):
    
    '''
    Get training and validation data
    '''
    
    # tests
    assert isinstance(df,    pd.DataFrame), 'df has to be a pandas dataframe'
    assert isinstance(rep,   int),          'rep has to be an integer'
    assert isinstance(fold,  int),          'fold has to be an integer'
    assert isinstance(CFG,   dict),         'CFG has to be a dict with parameters'

    # load splits
    df_train = df.loc[df['fold' + str(rep)] != fold].reset_index(drop = True)
    df_valid = df.loc[df['fold' + str(rep)] == fold].reset_index(drop = True)
    if not silent:
        accelerator.print('- no. observations: train - {}, valid - {}'.format(len(df_train), len(df_valid)))
        
    # checks
    assert len(df_train) + len(df_valid) == len(df), 'Incorrect no. observations'

    # subset for debug mode
    if debug is None:
        debug = CFG['debug']
    if debug:
        df_train = df_train.sample(CFG['batch_size'] * 10, random_state = CFG['seed']).reset_index(drop = True)
        df_valid = df_valid.sample(CFG['batch_size'] * 10, random_state = CFG['seed']).reset_index(drop = True)
        accelerator.print('- subsetting data for debug mode...')
        accelerator.print('- no. onservations: train - {}, valid - {}'.format(len(df_train), len(df_valid)))
    
    return df_train, df_valid



def collate_fn(batch, 
               tokenizer, 
               CFG, 
               labeled   = True, 
               return_sd = False):
    
    '''
    Collate batch inputs with padding
    '''
    
    # padding
    batch = tokenizer.pad(batch,
                          padding        = 'longest' if CFG['dynamic_pad'] else 'max_length',
                          max_length     = CFG['max_len'],
                          return_tensors = 'pt')
    
    # extract elements
    inputs, masks, token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
    
    # output
    if labeled:
        labels = batch['labels']
        if return_sd:
            sds = batch['sds']
            return inputs, masks, token_type_ids, labels, sds
        return inputs, masks, token_type_ids, labels
    return inputs, masks, token_type_ids



def get_loaders(df_train, 
                df_valid, 
                CFG, 
                accelerator, 
                labeled     = True, 
                return_sd   = True, 
                silent      = False):
    
    '''
    Get training and validation dataloaders
    '''
    
    # tests
    assert isinstance(df_train, pd.DataFrame), 'df_train has to be a pandas dataframe'
    assert isinstance(df_valid, pd.DataFrame), 'df_valid has to be a pandas dataframe'
    assert isinstance(CFG, dict),              'CFG has to be a dict with parameters'


    ##### DATASETS
    
    # tokenizer
    tokenizer = get_tokenizer(CFG)

    # datasets
    train_dataset = TextData(df          = df_train, 
                             tokenizer   = tokenizer,
                             max_len     = CFG['max_len'],
                             padding     = False,
                             p_translate = CFG['p_translate'],
                             labeled     = labeled,
                             return_sd   = return_sd)
    valid_dataset = TextData(df          = df_valid, 
                             tokenizer   = tokenizer,
                             max_len     = CFG['max_len'],
                             padding     = False,
                             p_translate = 0,
                             labeled     = labeled,
                             return_sd   = return_sd)

        
    ##### DATA LOADERS
    
    # samplers
    train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)
    
    # data loaders
    train_loader = DataLoader(dataset        = train_dataset, 
                              batch_size     = CFG['batch_size'], 
                              sampler        = train_sampler,
                              collate_fn     = lambda b: collate_fn(b, tokenizer, CFG, labeled, return_sd),
                              num_workers    = CFG['cpu_workers'],
                              drop_last      = False, 
                              worker_init_fn = worker_init_fn,
                              pin_memory     = False)
    valid_loader = DataLoader(dataset     = valid_dataset, 
                              batch_size  = CFG['valid_batch_size'], 
                              sampler     = valid_sampler,
                              collate_fn  = lambda b: collate_fn(b, tokenizer, CFG, labeled, return_sd),
                              num_workers = CFG['cpu_workers'],
                              drop_last   = False,
                              pin_memory  = False)
    
    return train_loader, valid_loader