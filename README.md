# Machine Learning Project

Classify LoL ranked games outcome by looking at the first 10min worth of data. [League of Legends Diamond Ranked Games (10 min)](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min)


# Instructions for the Machine Learning project

The project notebook has 8 main sections. Each of these sections is in a different section of the notebook. The corresponding sections and sections are detailed below:

### 1. Introduction

Introduction to the project and its theme.

### 2. Data

Exploratory analysis of data and dataset variables. Explanation of the variables and their meaning. Terminology used in the project.

### 3. Preprocessing

Data preprocessing. Elimination of variables, transformation of categorical variables, normalization of numerical variables, etc.

### 4. Models

Training of classification models. Use of different models for data classification. Comparison of the results obtained. Discriminative models and generative models.

### 5. Evaluation

Evaluation of the models. Use of different metrics for the evaluation of the models. Comparison of the results obtained.

### 6. Remodeled

Discarded section, as explained in the project report.

Re-modeling of the data. Use of re-modeling techniques to improve the results obtained. Use of remodeling techniques to improve model performance.

### 7. Hyperparameter Optimization

Optimization of the hyperparameters of the models. Use of hyperparameter optimization techniques to improve the results obtained.

### Test validation

Validation of the candidate model. Use of validation techniques to check the generalization of the candidate model.


For the replication of the results of the project, the following points must be taken into account:

## Libraries used

1. You must have installed the Python libraries and the corresponding versions that are at the beginning of the notebook. Otherwise, there might be compatibility issues with some functions or methods.

```python
!pip install pandas==1.5.3
!pip install numpy==1.24.2
!pip install seaborn==0.12.2
!pip install sklearn==1.0.2
!pip install matplotlib==3.5.1
!pip install plotly==5.6.0
```

## Dataset

2. The `league_dataset.csv` dataset must be downloaded in the same folder where the project notebook is located. This dataset can be found at the following link: [Kaggle](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min)

## Running the notebook

3. Once the compatibility requirements are met, all that remains is to execute the code. To do this, it is recommended that you run the notebook in a development environment such as Visual Studio Code, Jupyter Notebook, or Google Colab. It is recommended to always run all the preprocessing and then go to the sections of interest. If you want to re-execute any cell, it is recommended to repeat the execution process from the beginning.

## Helper functions

4. The different auxiliary functions used in the notebook must be downloaded. These functions are found in the `Resampling` folder.

### `resampling.py`

The `model_performance` function calculates the performance of a model using different resampling methods (training error, simple validation, Monte Carlo cross validation, k-fold cross validation). It takes as input the following parameters:

- `library`: The library used to build the model (eg sklearn, statsmodels, etc.).
- `method`: the method of the model to be used for fitting and prediction.
- `X`: a pandas DataFrame containing the data features.
- `y`: a pandas Array containing the target variable.
- `repeats` (optional): the number of repetitions of the experiment (default is 10).
- `k` (optional): the number of splits in the k-fold cross-validation (default is 20).
- `model` (optional): the name of the model to return (default is "single").
- `rand` (optional): the random seed used in the process (default is 88).

The function calculates the performance of the model using different resampling methods and stores the results in a DataFrame called `results_df` with the following columns: 'Resampling', 'Accuracy', 'Precision Macro', 'Recall Macro' and 'F1 Macro' .

The main steps of the function are described below:

1. Training the model with the entire training data set and calculating the training error.
2. Simple validation: The data set is divided into training and validation sets, the model is trained on the training set, and performance is calculated on the validation set.
3. Monte Carlo Cross Validation: A Monte Carlo cross validation is performed with a given number of iterations and the average performance of the model is calculated.
4. K-fold cross validation: A k-fold cross validation is performed with a given number of splits and the average performance of the model is calculated.
5. Depending on the value of `model`, the function returns the `results_df` DataFrame along with the corresponding trained model.

### `monte_carlo.py`

The `monte_carlo_cv` function implements Monte Carlo cross-validation to evaluate the performance of a model. This technique consists of performing multiple random splits of the data set into training and validation sets, fitting the model on the training set, and evaluating its performance on the validation set. The function takes the following parameters:

- `library`: The library used to build the model (eg sklearn, sklearn.naive_bayes, sklearn.tree, sklearn.ensemble).
- `method`: The model-specific method to use (eg GaussianNB, DecisionTreeClassifier, RandomForestClassifier).
- `X`: the characteristics of the data set.
- `y`: the target variable of the data set.
- `n`: the number of iterations, that is, the number of random divisions to perform.
- `rand` (optional): the random seed used in the process (default is 88).

Within the function, the following procedure is performed:

1. Empty lists are initialized to store performance metrics (accuracy, precision, recall, and F1 score) at each iteration.
2. A `for` loop is executed that performs `n` iterations.
3. At each iteration, the data set is split into training and validation sets using `train_test_split`, with a test ratio of 33% and the specified random seed.
4. An instance of the model specified by `method` is created.
5. Fit the model on the training set using `fit`.
6. Performance metrics (accuracy, precision, recall and F1 score) are calculated by comparing the actual labels in the validation set (`y_val`) with the model predictions in the validation set (`model.predict(X_val)` ).
7. The performance metrics for each iteration are added to the corresponding lists (`acc`, `prec`, `rec`, `f1`).
8. At the end of the iterations, the performance metrics are averaged using `np.mean`.
9. In addition to the performance metrics, the function returns the model trained on the last iteration.

### `k_fold.py`

The `k_fold_cv` function implements k-fold cross-validation to evaluate the performance of a model. This technique divides the data set into k folds, uses k-1 folds to train the model, and evaluates its performance on the remaining fold. The function takes the following parameters:

- `library`: The library used to build the model (eg sklearn, sklearn.naive_bayes, sklearn.tree, sklearn.ensemble).
- `method`: The model-specific method to use (eg GaussianNB, DecisionTreeClassifier, RandomForestClassifier).
- `X`: the characteristics of the data set.
- `y`: the target variable of the data set.
- `k`: the number of folds in the k-fold cross-validation.
- `rand` (optional): the random seed used in the process (default is 88).

Within the function, the following procedure is performed:

1. An instance of `KFold` is created with the specified number of folds (`n_splits=k`), the option to shuffle the data (`shuffle=True`) and the random seed (`random_state=rand`).
2. Empty lists are initialized to store performance metrics (accuracy, precision, recall, and F1 score) on each fold.
3. A `for` loop is executed that iterates over the indices of the folds generated by `KFold.split(X)`.
4. At each iteration, the data set is divided into training and validation sets using the current fold indices.
5. A numeric version of the training and validation sets (`X_learn`, `y_learn`, `X_val`, `y_val`) is created to ensure that they are NumPy arrays.
6. An instance of the model specified by `method` is created.
7. The model is fitted to the training set using `fit`.
8. Performance metrics (accuracy, precision, recall and F1 score) are calculated by comparing the actual labels in the validation set (`y_val`) with the model predictions in the validation set (`model.predict(X_val)` ).
9. The performance metrics for each fold are added to the corresponding lists (`acc`, `prec`, `rec`, `f1`).
10. Upon completion of the folds, the performance metrics are averaged using `np.mean`.
11. In addition to the performance metrics, the function returns the model trained on the last fold.

In short, the `k_fold_cv` function performs k-fold cross-validation to assess the performance of a model using specific slices of the data set. Returns the average performance metrics and the model trained at the last fold.
