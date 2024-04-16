from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def scorer():

    scoring = {
        'Accuracy':'accuracy',
        'Precision':'precision',
        'Recall':'recall',
        'AUC':'roc_auc'
        }

    return scoring