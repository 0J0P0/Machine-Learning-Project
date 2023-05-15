import pandas as pd
from sklearn import metrics
from kfold import k_fold_cv
from monte_carlo import monte_carlo_cv
from sklearn.model_selection import train_test_split


def model_performance(library, method, X, y, repeats=10, k=20, model="single"):
    """
    This function computes the performance of a model using different resampling methods (training error, single validation, monte carlo cross-validation, k-fold cross-validation).

    Parameters:
    -----------
    library: sklearn, statsmodels, etc.
    method: string, name of the method to use
    X: pandas DataFrame, features
    y: pandas Series, target
    repeats: int, number of times to repeat the experiment
    N: int, number of times to repeat the train/test split
    model: string, name of the model to return

    Returns:
    --------
    results_df: pandas DataFrame, performance of the model using different resampling methods
    model: sklearn model, model trained with the whole dataset
    """
    results_df = pd.DataFrame(index=[], columns= ['Accuracy', 'Precision Macro', 'Recall Macro', 'F1 Macro'])

    # 1. Training error
    tr_mod = getattr(library, method)()
    tr_mod.fit(X, y)
    # save accuracy, precision, recall, f1 in a dataframe and add it as a row to the results_df
    results_df = results_df.append(pd.DataFrame([[metrics.accuracy_score(y, tr_mod.predict(X)),
                                                metrics.precision_score(y, tr_mod.predict(X), average='macro'),
                                                metrics.recall_score(y, tr_mod.predict(X), average='macro'),
                                                metrics.f1_score(y, tr_mod.predict(X), average='macro')]],
                                                columns= ['Accuracy', 'Precision Macro', 'Recall Macro', 'F1 Macro']),
                                                ignore_index=True)

    # 2. Single Validation
    X_learn, X_val, y_learn, y_val = train_test_split(X, y, test_size=0.33, random_state=88)
    sv_mod = getattr(library, method)()
    sv_mod.fit(X_learn, y_learn)
    results_df = results_df.append(pd.DataFrame([[metrics.accuracy_score(y_val, sv_mod.predict(X_val)),
                                                metrics.precision_score(y_val, sv_mod.predict(X_val), average='macro'),
                                                metrics.recall_score(y_val, sv_mod.predict(X_val), average='macro'),
                                                metrics.f1_score(y_val, sv_mod.predict(X_val), average='macro')]],
                                                columns= ['Accuracy', 'Precision Macro', 'Recall Macro', 'F1 Macro']),
                                                ignore_index=True)
    
    

    # 3. Monte carlo cross-val (with k=1 up to 'repeats' repetitions)
    mc_acc, mc_prec, mc_rec, mc_f1, mc_mod = monte_carlo_cv(library, method, X, y, repeats)
    results_df = results_df.append(pd.DataFrame([[mc_acc, mc_prec, mc_rec, mc_f1]],
                                                columns= ['Accuracy', 'Precision Macro', 'Recall Macro', 'F1 Macro']),
                                                ignore_index=True)
    
    # 4. k-fold cross-validation
    kf_acc, kf_prec, kf_rec, kf_f1, kf_mod = k_fold_cv(library, method, X, y, k)
    results_df = results_df.append(pd.DataFrame([[kf_acc, kf_prec, kf_rec, kf_f1]],
                                                columns= ['Accuracy', 'Precision Macro', 'Recall Macro', 'F1 Macro']),
                                                ignore_index=True)

    if model == "train":
        return results_df, tr_mod
    elif model == "single":
        return results_df, sv_mod
    elif model == "monte":
        return results_df, mc_mod
    elif model == "kfold":
        return results_df, kf_mod
