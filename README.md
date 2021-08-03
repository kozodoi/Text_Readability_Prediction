# CommonLit Readability Prediction

Top-9% solution to the [CommonLit Readability Prize](https://www.kaggle.com/c/commonlitreadabilityprize) Kaggle competition on text complexity prediction.

![sample](https://i.postimg.cc/hv6yfMYz/cover-books.jpg)


## Summary

Estimating text complexity and readability is a crucial task for teachers. Offering students text passages at the right level of challenge is important for facilitating a fast development of reading skills. The existing tools to estimate text complexity rely in weak proxies and heuristics, which results in a suboptimal quality.

This projects uses deep learning to predict readability score of text passages. My solution is an ensemble of eight transformer models, including Bert, Roberta and others. All transformers are implemented in `PyTorch`. The table below summarizes the main architecture and training parameters. The solution places in the top-9% of the Kaggle competition leaderboard.

![models](https://i.postimg.cc/JnY0HdWQ/read-params.jpg)


## Project structure

The project has the following structure:
- `codes/`: `.py` main scripts with data, model, training and inference modules
- `notebooks/`: `.ipynb` Colab-friendly notebooks for running model training
- `input/`: input data
- `output/`: model configurations, weights and figures exported from the notebooks


## Working with the repo

### Environment

To work with the repo, I recommend to create a virtual Conda environment from the `environment.yml` file:
```
conda env create --name read --file environment.yml
conda activate read
```

### Reproducing solution

The solution can then be reproduced in the following steps:
1. Run training notebooks `01_model_v36.ipynb` - `08_model_v69.ipynb` to obtain weights of base models.
2. Run the ensembling notebook `09_ensembling.ipynb` to obtain the final predictions.

All training notebooks have the same structure and differ in model/data parameters. Different versions are included to ensure reproducibility. To understand the training process, it is sufficient to go through the `codes/` folder and inspect one of the modeling notebooks.

More details are provided in the documentation within the scripts & notebooks.
