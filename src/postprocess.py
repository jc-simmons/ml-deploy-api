import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from src.scoring import scorer
import json
import pathlib
import seaborn as sn
import pickle

class PostProcessor:

    @staticmethod
    def save_evaluate(out_dir, estimator, X_test, y_test):

        scores = scorer(estimator, X_test, y_test)
        with open(pathlib.Path(out_dir + '/scores.json'), 'w') as f:
            json.dump(scores, f)


        y_pred_proba= estimator.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        fig = plt.figure(figsize=(8, 6))
        # Plot the diagonal 50% line
        plt.plot([0, 1], [0, 1], 'k--')
        # Plot the FPR and TPR achieved by our model
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig(pathlib.Path(out_dir + '/roc.png', dpi=1200, bbox_inches = 'tight'))
        plt.clf()


        y_pred = estimator.predict(X_test).astype(int)
        conf_matrix = confusion_matrix(y_test, y_pred)
        sn.heatmap(conf_matrix, annot=True, cmap="Blues")
        plt.savefig(pathlib.Path(out_dir + '/confusion_matrix.png', dpi=1200, bbox_inches = 'tight'))

        return
    

    @staticmethod
    def save_model(out_dir, estimator):

        with open(pathlib.Path(out_dir + '/clfmodel.pkl'), 'wb') as f:
            pickle.dump(estimator, f)

        return
    
    