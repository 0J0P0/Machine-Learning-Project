import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split


def monte_carlo_cv(library, method, X, y, n, rand=88):
    """
    Monte Carlo Cross Validation

    Parameters:
    -----------
    library: sklearn, sklearn.naive_bayes, sklearn.tree, sklearn.ensemble
    method: GaussianNB, DecisionTreeClassifier, RandomForestClassifier
    X: features
    y: target
    n: number of iterations
    rand: int, random seed

    Returns:
    --------
    acc: accuracy
    prec: precision
    rec: recall
    f1: f1 score
    model: model
    """
    acc = []; prec = []; rec = []; f1 = []
    for _ in range(n):
        # aqui no tiene que haber random o si?
        X_learn, X_val, y_learn, y_val = train_test_split(X, y, test_size=0.33, random_state=rand)
        model = method
        model.fit(X_learn, y_learn)
        
        acc.append(metrics.accuracy_score(y_val, model.predict(X_val)))
        prec.append(metrics.precision_score(y_val, model.predict(X_val)))
        rec.append(metrics.recall_score(y_val, model.predict(X_val)))
        f1.append(metrics.f1_score(y_val, model.predict(X_val)))
    return np.mean(acc), np.mean(prec), np.mean(rec), np.mean(f1), model