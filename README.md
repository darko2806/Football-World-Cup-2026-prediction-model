# World Cup Match Prediction

This repository contains a set of neural network experiments for predicting the outcome of international football matches.

The problem is treated as a three-class classification task:

- home win
- draw
- away win

The models are implemented in PyTorch and use features based on FIFA rankings, Elo ratings, and recent team form.

## Project Structure

```text
DataProcessing/                     Data preparation and feature engineering
data/                               Raw and processed datasets
experiment_1/                       Initial experiments with ranking features
experiments/                        SMA, EMA, and combined-feature experiments
Experiments_2_and_3_EMA_SMA_points/ Updated form-based experiments
Experiment_block_3_LeakyRelu_GELU/  Activation function comparison
results/                            Experiment summaries
```

## Experiments

The repository includes experiments with:

- FIFA ranking and Elo rating features
- Simple Moving Average (SMA) form
- Exponential Moving Average (EMA) form
- differences between home and away team features
- combined ranking and form features
- strength-adjusted goal margin
- ReLU, LeakyReLU, and GELU activation functions

Each experiment is stored in a separate Jupyter notebook. Shared training and evaluation functions are located in the corresponding `experiment_utils.py` files.

## Evaluation

The models are evaluated using:

- accuracy
- precision
- recall
- specificity
- F1-score

The dataset is split chronologically to ensure that future matches are not included in the training data.

## Results

The main experiment summaries are stored in the `results/` directory as CSV files.

## Technologies

- Python
- PyTorch
- pandas
- NumPy
- scikit-learn
- Jupyter Notebook