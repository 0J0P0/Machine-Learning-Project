import numpy as np
import pandas as pd
import plotly.figure_factory as ff
from Resampling.kfold import k_fold_cv
from sklearn.metrics import confusion_matrix
from Resampling.monte_carlo import monte_carlo_cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def confusion_matrix_plot(y_real, y_pred):
    """
    This function shows a confusion matrix using plotly.

    Parameters:
    -----------
    y_real: pandas Series, real target
    y_pred: pandas Series, predicted target
    """
    cm = confusion_matrix(y_real, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = ff.create_annotated_heatmap(cm, colorscale='Viridis')
    fig.update_layout(title='Confusion Matrix using Single validation',
                    xaxis_title='Predicted',
                    yaxis_title='Real',
                    width=500,
                    height=500,
                    showlegend=False)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 12
        fig.layout.annotations[i].text = str(round(float(fig.layout.annotations[i].text), 2))
    
    fig.show()


def compute_metrics(y_real,y_pred):
    """
    This function computes the accuracy, precision, recall and f1 score of a model.

    Parameters:
    -----------
    y_real: pandas Series, real target
    y_pred: pandas Series, predicted target

    Returns:
    --------
    accuracy: float, accuracy
    precison_macro: float, precision
    recall_macro: float, recall
    f1_macro: float, f1 score
    """
    accuracy = accuracy_score(y_real,y_pred)
    precison_macro =precision_score(y_real,y_pred,  average='macro')
    recall_macro =recall_score(y_real,y_pred,  average='macro')
    f1_macro =f1_score(y_real,y_pred, average='macro')
    return [accuracy, precison_macro, recall_macro, f1_macro]


def model_performance(library, method, X, y, repeats=10, k=20, model="single", rand=88):
    """
    This function computes the performance of a model using different resampling methods (training error, single validation, monte carlo cross-validation, k-fold cross-validation).

    Parameters:
    -----------
    library: sklearn, statsmodels, etc.
    method:
    X: pandas DataFrame, features
    y: pandas Series, target
    repeats: int, number of times to repeat the experiment
    N: int, number of times to repeat the train/test split
    model: string, name of the model to return
    rand: int, random seed

    Returns:
    --------
    results_df: pandas DataFrame, performance of the model using different resampling methods
    model: sklearn model, model trained with the whole dataset
    """
    cols = ['Resampling', 'Accuracy', 'Precision Macro', 'Recall Macro', 'F1 Macro']
    results_df = pd.DataFrame(index=[], columns= cols)
    # 1. Training error
    tr_mod = method
    tr_mod.fit(X=X, y=y)
    results_df.loc[0] = ['Training error'] + compute_metrics(y,tr_mod.predict(X))


    # 2. Single Validation
    X_learn, X_val, y_learn, y_val = train_test_split(X, y, test_size=0.33, random_state=rand)
    sv_mod = method
    sv_mod.fit(X_learn, y_learn)
    results_df.loc[1] = ['Single Validation'] + compute_metrics(y_val,sv_mod.predict(X_val))
    confusion_matrix_plot(y_val,sv_mod.predict(X_val))
    

    # 3. Monte carlo cross-val (with k=1 up to 'repeats' repetitions)
    mc_acc, mc_prec, mc_rec, mc_f1, mc_mod = monte_carlo_cv(library, method, X, y, repeats, rand)
    results_df.loc[2] = ['Monte Carlo'] + [mc_acc, mc_prec, mc_rec, mc_f1]
    
    # 4. k-fold cross-validation
    kf_acc, kf_prec, kf_rec, kf_f1, kf_mod = k_fold_cv(library, method, X, y, k, rand)
    results_df.loc[3] = ['K-fold'] + [kf_acc, kf_prec, kf_rec, kf_f1]

    if model == "train":
        return results_df, tr_mod
    elif model == "single":
        return results_df, sv_mod
    elif model == "monte":
        return results_df, mc_mod
    elif model == "kfold":
        return results_df, kf_mod
