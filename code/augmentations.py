import torch



def shuffle_sentences(inputs, 
                      accelerator, 
                      end_tokens = torch.tensor([4, 116, 328, 322]), # . ? ! ).
                      pad_token  = 1):
    
    '''
    Shuffle sentence order
    '''
    
    # send to device
    end_tokens = end_tokens.to(accelerator.device)

    # loop through texts
    for i in range(inputs.shape[0]):
        
        # find sentence borders
        sentence_idx = torch.stack([t == inputs[i, :] for t in end_tokens]).sum(0).bool().nonzero(as_tuple = False)
        sentence_num = len(sentence_idx)
        
        # run if text is long enough 
        if sentence_num >= 3:

            # new sentence order
            new_sentence_order = torch.cat((torch.tensor([0], device = accelerator.device), 
                                            torch.randperm(sentence_num - 2, device = accelerator.device) + 1,
                                            torch.tensor([sentence_num - 1], device = accelerator.device)))

            # set new sentence borders
            sentence_idx = torch.cat((torch.zeros(1, 1, dtype = torch.int64, device = accelerator.device), sentence_idx))
            sentence_idx = torch.cat((sentence_idx + 1, torch.roll(sentence_idx, shifts = -1)), axis = 1)[:-1, :]
            sentence_idx[0, 0] -= 1
            sentence_idx[sentence_idx.shape[0] - 1, sentence_idx.shape[1] - 1] += 1
            new_sentence_idx = sentence_idx[new_sentence_order, :]

            # add incomplete sentence
            if torch.max(new_sentence_idx) + 1 < len(inputs[i, :]):
                new_sentence_idx = torch.cat((new_sentence_idx, 
                                              torch.tensor([[torch.max(new_sentence_idx) + 1, len(inputs[i, :]) - 1]], 
                                                           device = accelerator.device)))
                sentence_num += 1

            # shuffle sentences 
            for j in range(sentence_num):
                sentence_id = torch.range(new_sentence_idx[j, 0], new_sentence_idx[j, 1], dtype = torch.int64, device = accelerator.device)
                sentence = inputs[i, :].gather(0, sentence_id)
                if j == 0:
                    new_input = sentence
                else:
                    new_input = torch.cat((new_input, sentence))

            # save new text
            inputs[i, :] = new_input 

    return inputs
