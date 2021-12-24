# Text Readability Prediction

Top-9% solution to the [CommonLit Readability Prize](https://www.kaggle.com/c/commonlitreadabilityprize) Kaggle competition on text complexity prediction and interactive web app for evaluating your own text.

![sample](https://i.postimg.cc/hv6yfMYz/cover-books.jpg)


## Summary

Estimating text complexity and readability is a crucial task for school teachers. Offering students text passages at the right level of challenge is important for facilitating a fast development of reading skills. The existing tools to estimate text complexity rely on weak proxies and heuristics, which results in a suboptimal accuracy. This project uses deep learning to predict the readability scores of text passages.

My solution is an ensemble of eight transformer models, including BERT, RoBERTa and others. All transformers are implemented in `PyTorch` and feature a custom regression head that uses a concatenated output of multiple hidden layers. The modeling pipeline implements text augmentations such as sentence order shuffle, backtranslation and injecting target noise. The table below summarizes the main architecture and training parameters. The solution places in the top-9% of the Kaggle competition leaderboard.

![models](https://i.postimg.cc/JnY0HdWQ/read-params.jpg)


## Demo app

The project includes [an interactive web app](https://share.streamlit.io/kozodoi/text_readability_prediction/main/web_app.py). The app allows estimating reading complexity of a custom text using one of the trained transformer models. The app is built in Python using the Streamlit package.

![web_app](https://i.postimg.cc/fTBCX3LY/ezgif-com-gif-maker-6.gif)


## Project structure

The project has the following structure:
- `codes/`: `.py` main scripts with data, model, training and inference modules
- `notebooks/`: `.ipynb` Colab-friendly notebooks with model training and emsembling
- `input/`: input data, including raw texts, backtranslated texts and meta-features
- `output/`: model configurations, weights and figures exported from the notebooks


## Working with the repo

### Environment

To work with the repo, I recommend creating a virtual Conda environment from the `project_environment.yml` file:
```
conda env create --name read --file project_environment.yml
conda activate read
```

The file `requirements.txt` provides the list of dependencies for the web app.


### Reproducing solution

The solution can be reproduced in the following steps:
1. Run training notebooks `01_model_v36.ipynb` - `08_model_v69.ipynb` to obtain weights of base models.
2. Run the feature engineering notebook `09_meta_features.ipynb` to calculate text meta-features.
3. Run the ensembling notebook `10_ensembling.ipynb` to obtain the final predictions.

All training notebooks have the same structure and differ in model/data parameters. Different versions are included to ensure reproducibility. To understand the training process, it is sufficient to go through the `codes/` folder and inspect one of the modeling notebooks.

More details are provided in the documentation within the scripts & notebooks.
