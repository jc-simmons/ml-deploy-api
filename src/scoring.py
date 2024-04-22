from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def scorer(estimator, X, y):
    scores = {}

    y_pred = estimator.predict(X).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    auc = roc_auc_score(y,estimator.predict_proba(X)[:,1])

    scores['acc'] = acc
    scores['prec'] = prec
    scores['rec'] = rec
    scores['auc'] = auc

    return scores 

