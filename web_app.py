##### PREPARATIONS

# libraries
import gc
import os
import pickle
import sys
import urllib.request
import requests
import numpy as np
import pandas as pd
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
        
        
##### CONFIG

# page config
st.set_page_config(page_title            = "Readability prediction", 
                   page_icon             = ":books:", 
                   layout                = "centered", 
                   initial_sidebar_state = "collapsed", 
                   menu_items            = None)


##### HEADER

# title
st.title('Text readability prediction')

# image cover
image = Image.open(requests.get('https://i.postimg.cc/hv6yfMYz/cover-books.jpg', stream = True).raw)
st.image(image)

# description
st.write('This app uses deep learning to estimate the reading complexity of a custom text. Enter your text below, and we will run it through one of the two transfomer models and display the result.')


##### PARAMETERS

# title
st.header('How readable is your text?')

# model selection
model_name = st.selectbox(
    'Which model would you like to use?',
    ['DistilBERT', 'DistilRoBERTa'])

# input text
input_text = st.text_area('Which text would you like to rate?', 'Please enter the text in this field.')


##### MODELING

# compute readability
if st.button('Compute readability'):

    # specify paths
    if model_name == 'DistilBERT':
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

        # clear memory
        gc.collect()

        # load config
        config = pickle.load(open(folder_path + 'configuration.pkl', 'rb'))
        config['backbone'] = folder_path

        # initialize model
        model = get_model(config, name = model_name.lower(), pretrained = folder_path + 'pytorch_model.bin')
        model.eval()

        # load tokenizer
        tokenizer = get_tokenizer(config)

        # tokenize text
        text = tokenizer(text                  = input_text,
                         truncation            = True,
                         add_special_tokens    = True,
                         max_length            = config['max_len'],
                         padding               = False,
                         return_token_type_ids = True,
                         return_attention_mask = True,
                         return_tensors        = 'pt')
        inputs, masks, token_type_ids = text['input_ids'], text['attention_mask'], text['token_type_ids']

        # clear memory
        del tokenizer, text, config
        gc.collect()

        # compute prediction
        if input_text != '':
            prediction = model(inputs, masks, token_type_ids)
            prediction = prediction['logits'].detach().numpy()[0][0]
            prediction = 100 * (prediction + 4) / 6 # scale to [0,100]

        # clear memory
        del model, inputs, masks, token_type_ids
        gc.collect()

        # print output
        st.metric('Readability score:', '{:.2f}%'.format(prediction, 2))
        st.write('**Note:** readability scores are scaled to [0, 100%]. A higher score means that the text is easier to read.')
        st.success('Success! Thanks for scoring your text :)')


##### DOCUMENTATION

# header
st.header('More information')

# example texts
with st.expander('Show example texts'):
    st.table(pd.DataFrame({
        'Text':  ['A dog sits on the floor. A cat sleeps on the sofa.',  'This app does text readability prediction. How cool is that?', 'Training of deep bidirectional transformers for language understanding.'],
        'Score': ['92.62%', '66.50%', '26.62%'],
    }))
    
# models
with st.expander('Read about the models'):
    st.write("Both transformer models are part of my top-9% solution to the CommonLit Readability Kaggle competition. The pre-trained language models are fine-tuned on 2834 text snippets. [Click here](https://github.com/kozodoi/Kaggle_Readability) to see the source code and read more about the training pipeline.")
    
# metric
with st.expander('Read about the metric'):
    st.write("The readability metric is calculated on the basis of a Bradley-Terry analysis of more than 111,000 pairwise comparisons between excerpts. Teachers spanning grades 3-12 (a majority teaching between grades 6-10) served as the raters for these comparisons. The raw scores vary in [-4, 2] and are scaled to [0, 100%] for convenience. More details on the used reading complexity metric are available [here](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/240886).")
    
    
##### CONTACT

# header
st.header("Contact")

# website link
st.write("Check out [my website](https://kozodoi.me) for ML blog, academic publications, Kaggle solutions and more of my work.")

# profile links
st.write("[![Linkedin](https://img.shields.io/badge/-LinkedIn-306EA8?style=flat&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/kozodoi/)](https://www.linkedin.com/in/kozodoi/) [![Twitter](https://img.shields.io/badge/-Twitter-4B9AE5?style=flat&logo=Twitter&logoColor=white&link=https://www.twitter.com/n_kozodoi)](https://www.twitter.com/n_kozodoi) [![Kaggle](https://img.shields.io/badge/-Kaggle-5DB0DB?style=flat&logo=Kaggle&logoColor=white&link=https://www.kaggle.com/kozodoi)](https://www.kaggle.com/kozodoi) [![GitHub](https://img.shields.io/badge/-GitHub-2F2F2F?style=flat&logo=github&logoColor=white&link=https://www.github.com/kozodoi)](https://www.github.com/kozodoi) [![Google Scholar](https://img.shields.io/badge/-Google_Scholar-676767?style=flat&logo=google-scholar&logoColor=white&link=https://scholar.google.com/citations?user=58tMuD0AAAAJ&amp;hl=en)](https://scholar.google.com/citations?user=58tMuD0AAAAJ&amp;hl=en) [![ResearchGate](https://img.shields.io/badge/-ResearchGate-59C3B5?style=flat&logo=researchgate&logoColor=white&link=https://www.researchgate.net/profile/Nikita_Kozodoi)](https://www.researchgate.net/profile/Nikita_Kozodoi) [![Tea](https://img.shields.io/badge/-Buy_me_a_tea-yellow?style=flat&logo=buymeacoffee&logoColor=white&link=https://www.buymeacoffee.com/kozodoi)](https://www.buymeacoffee.com/kozodoi)")

# copyright
st.text("© 2022 Nikita Kozodoi")