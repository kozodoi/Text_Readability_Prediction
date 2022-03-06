from utilities import *

import transformers
from transformers import AutoModel
import torch
import torch.nn as nn
import gc


def get_model(CFG, 
              pretrained = None, 
              silent     = False, 
              name       = None):

    '''
    Get transformer model
    '''
    
    # tests
    assert isinstance(CFG, dict), 'CFG has to be a dict with parameters'

    # transformer class
    class TransformerModel(nn.Module):

        # init
        def __init__(self,
                     model_path,
                     model_name        = None,
                     hidden_size       = 768,
                     initializer_range = None,
                     pooling           = 'mean',
                     pooling_layer     = -1,
                     concat_layers     = 4,
                     layer_norm_eps    = 1e-7,
                     hidden_dropout    = 0.0,
                     head_dropout      = 0.0):

            super(TransformerModel, self).__init__()
            
            # checks
            assert hidden_dropout >= 0, 'dropout has to be greater than zero'
            assert head_dropout   >= 0, 'dropout has to be greater than zero'
            assert concat_layers  > 0,  'concat_layers has to be positive'
            assert hidden_size    > 0,  'hidden_size has to be positive'
            
            # parameters
            if model_name is None:
                self.model_name = model_path
            else:
                self.model_name = model_name
            self.model_path     = model_path
            self.hidden_size    = hidden_size
            self.init_range     = initializer_range
            self.pooling        = pooling
            self.pooling_layer  = pooling_layer
            self.concat_layers  = concat_layers
            self.layer_norm_eps = layer_norm_eps
            self.hidden_dropout = hidden_dropout
            self.head_dropout   = head_dropout
            self.feature_size   = hidden_size * self.concat_layers
            if self.pooling == 'meanmax':
                self.feature_size *= 2

            # layers
            if 'funnel' in self.model_name:
                self.backbone = AutoModel.from_pretrained(pretrained_model_name_or_path = self.model_path,
                                                          hidden_dropout                = self.hidden_dropout,
                                                          layer_norm_eps                = self.layer_norm_eps,
                                                          output_hidden_states          = True)
            elif 'distilbert' in self.model_name or 'xlnet' in self.model_name or 'bart' in self.model_name:
                self.backbone = AutoModel.from_pretrained(pretrained_model_name_or_path = self.model_path,
                                                          dropout                       = self.hidden_dropout,
                                                          output_hidden_states          = True)
            else:
                self.backbone = AutoModel.from_pretrained(pretrained_model_name_or_path = self.model_path,
                                                          hidden_dropout_prob           = self.hidden_dropout,
                                                          layer_norm_eps                = self.layer_norm_eps,
                                                          output_hidden_states          = True)
            self.dropout   = nn.Dropout(self.head_dropout)
            self.regressor = nn.Linear(self.feature_size, 1)

            # weights
            if self.init_range is not None:
                self._init_weights(self.regressor)


        # weight initialization
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean = 0.0, std = self.init_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean = 0.0, std = self.init_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


        # forward function
        def forward(self,
                    input_ids      = None,
                    attention_mask = None,
                    token_type_ids = None):

            # backbone output
            if 'distilbert' in self.model_name or 'xlnet' in self.model_name or 'bart' in self.model_name:
                outputs = self.backbone(input_ids      = input_ids,
                                        attention_mask = attention_mask)
            else:
                outputs = self.backbone(input_ids      = input_ids,
                                        attention_mask = attention_mask,
                                        token_type_ids = token_type_ids)

            # pooling transformer outputs
            if self.pooling == 'default':
                features = outputs['pooler_output']
            else:
                if 'bart' in self.model_name:
                    hidden_states = torch.stack(outputs['encoder_hidden_states'])
                elif 'funnel' in self.model_name:
                    hidden_states = torch.stack(outputs['hidden_states'][-3:])
                else:
                    hidden_states = torch.stack(outputs['hidden_states'])
                for l in range(self.concat_layers):
                    if l == 0:
                        hidden_state = hidden_states[self.pooling_layer, :, :]
                    else:
                        hidden_state = torch.cat((hidden_state, hidden_states[self.pooling_layer - l]), -1)
                if self.pooling == 'cls':
                    features = hidden_state[:, 0]
                elif self.pooling == 'mean':
                    mask_expanded  = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
                    sum_embeddings = torch.sum(hidden_state * mask_expanded, 1)
                    sum_mask       = mask_expanded.sum(1)
                    sum_mask       = torch.clamp(sum_mask, min = 1e-9)
                    features       = sum_embeddings / sum_mask
                elif self.pooling == 'max':
                    mask_expanded  = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
                    hidden_state[mask_expanded == 0] = -1e9
                    features = torch.max(hidden_state, 1)[0]
                elif self.pooling == 'meanmax':
                    mask_expanded  = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
                    sum_embeddings = torch.sum(hidden_state * mask_expanded, 1)
                    sum_mask       = mask_expanded.sum(1)
                    sum_mask       = torch.clamp(sum_mask, min = 1e-9)
                    mean_features  = sum_embeddings / sum_mask
                    hidden_state[mask_expanded == 0] = -1e9
                    max_features   = torch.max(hidden_state, 1)[0]
                    features       = torch.cat((mean_features, max_features), 1)

            # dropout
            if self.head_dropout > 0:
                features = self.dropout(features)

            # regressor
            logits = self.regressor(features)

            # output
            return {'logits': logits}


    # initialize model
    model = TransformerModel(model_path        = CFG['backbone'],
                             model_name        = name,
                             hidden_size       = CFG['hidden_size'],
                             initializer_range = CFG['init_range'],
                             pooling           = CFG['pooling'],
                             pooling_layer     = CFG['pooling_layer'],
                             concat_layers     = CFG['concat_layers'],
                             layer_norm_eps    = CFG['layer_norm_eps'],
                             hidden_dropout    = CFG['hidden_dropout'],
                             head_dropout      = CFG['head_dropout'])

    # load pre-trained weights
    if pretrained is None:
        pretrained = CFG['pretrained']
    if pretrained is not True:
        model.load_state_dict(torch.load(pretrained, map_location = torch.device('cpu')))
        if not silent:
            print('-- loaded custom weights')

    # freezing deep layers
    if CFG['freeze_embed']:
        for param in model.backbone.embeddings.parameters():
            param.requires_grad = False
    if CFG['freeze_layers'] > 0:
        for name, child in model.backbone.encoder.layer.named_children():
            if int(name) < CFG['freeze_layers']:
                for param in child.parameters():
                    param.requires_grad = False

    return model
