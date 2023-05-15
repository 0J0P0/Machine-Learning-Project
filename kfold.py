import numpy as np
# import pandas as pd

from sklearn import metrics
from sklearn.model_selection import KFold


def k_fold_cv(library, method, X, y, k):
    """
    library: sklearn, sklearn.naive_bayes, sklearn.tree, sklearn.ensemble
    method: GaussianNB, DecisionTreeClassifier, RandomForestClassifier
    X: features
    y: target
    k: number of folds
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=88)
    acc = []; prec = []; rec = []; f1 = []
    for learn_index, val_index in kf.split(X):
        X_learn = X.values[learn_index]
        y_learn = y.values[learn_index]
        X_val = X.values[val_index,:]
        y_val = y.values[val_index]
        model = getattr(library, method)()
        model.fit(X_learn, y_learn)
        acc.append(metrics.accuracy_score(y_val, model.predict(X_val)))
        prec.append(metrics.precision_score(y_val, model.predict(X_val)))
        rec.append(metrics.recall_score(y_val, model.predict(X_val)))
        f1.append(metrics.f1_score(y_val, model.predict(X_val)))
    return np.mean(acc), np.mean(prec), np.mean(rec), np.mean(f1), model