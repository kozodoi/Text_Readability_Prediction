# libraries
import gc
import os
import pickle
import sys
import urllib.request
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# custom libraries
sys.path.append('code')  
from model import get_model
from tokenizer import get_tokenizer

# download with progress bar
mybar = None
def show_progress(block_num, block_size, total_size):
    global mybar
    if mybar is None:
        mybar = st.progress(0.0)
    downloaded = block_num * block_size / total_size
    if downloaded <= 1.0:
        mybar.progress(downloaded)
    else:
        mybar.progress(1.0)

# title
st.title('Text Readability Prediction')

# image cover
image = Image.open(requests.get('https://i.postimg.cc/hv6yfMYz/cover-books.jpg', stream = True).raw)
st.image(image)

# description
st.write('This app allows estimating the reading complexity of a custom text with one of the two transfomer models. The models are part of my top-9% solution to the CommonLit Readability Kaggle competition. [Click here](https://github.com/kozodoi/Kaggle_Readability) to see the code and read more about the training pipeline.')

# model selection
model_name = st.selectbox(
    'Which model would you like to use?',
     ['DistilBERT', 'DistilRoBERTa'])

# input text
input_text = st.text_area('Which text would you like to rate?', 'Please enter the text in this field.')

# compute readability
if st.button('Compute readability'):
    
    # specify paths
    if model_name == 'RoBERTa':
        folder_path = 'output/v52/'
        weight_path = 'https://github.com/kozodoi/Kaggle_Readability/releases/download/0e96d53/weights_v52.pth'
    elif model_name == 'DistilBERT':
        folder_path = 'output/v59/'
        weight_path = 'https://github.com/kozodoi/Kaggle_Readability/releases/download/0e96d53/weights_v59.pth'
    elif model_name == 'DistilRoBERTa':
        folder_path = 'output/v47/'
        weight_path = 'https://github.com/kozodoi/Kaggle_Readability/releases/download/0e96d53/weights_v47.pth'

    # download model weights
    if not os.path.isfile(folder_path + 'pytorch_model.bin'):
        with st.spinner('Downloading model weights. This is done once and can take a minute...'):
            urllib.request.urlretrieve(weight_path, folder_path + 'pytorch_model.bin', show_progress)

    # compute predictions
    with st.spinner('Computing prediction...'):

        # load config
        config = pickle.load(open(folder_path + 'configuration.pkl', 'rb'))  
        config['backbone'] = folder_path

        # initialize model
        model = get_model(config, name = model_name.lower(), pretrained = folder_path + 'pytorch_model.bin')
        model.eval()    

        # load tokenizer
        tokenizer = get_tokenizer(config)

        # tokenize text
        text = tokenizer(text                 = input_text,
                        truncation            = True,
                        add_special_tokens    = True, 
                        max_length            = config['max_len'],
                        padding               = False,
                        return_token_type_ids = True,
                        return_attention_mask = True,
                        return_tensors        = 'pt')
        inputs, masks, token_type_ids = text['input_ids'], text['attention_mask'], text['token_type_ids']

        # compute prediction
        if input_text != '':
            prediction = model(inputs, masks, token_type_ids)
            prediction = prediction['logits'].detach().numpy()[0][0]

        # model output
        st.write('**Predicted readability score:** ', np.round(prediction, 4))

        # clear memory
        del model, tokenizer, config, text, inputs, masks, token_type_ids
        gc.collect()


# about the scores
st.write('**Note:** readability scores vary in [-4, 2]. A higher score indicates that the text is easier to read. More details on the used reading complexity metric are available [here](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/240886).')

# example texts
my_expander = st.beta_expander(label = 'Show example texts')
with my_expander:
    st.table(pd.DataFrame({
        'Text':  ['A dog sits on the floor. A cat sleeps on the sofa.',  'This app does text readability prediction. How cool is that?', 'Pre-training of deep bidirectional transformers for language understanding.'],
        'Score': [1.5571, -0.0100, -2.5071],
    }))