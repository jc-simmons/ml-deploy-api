import yaml
import importlib
import numpy as np
import pandas as pd
import pathlib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from load_data import data_loader
from preprocess import create_preprocessor
from cv_helper import cv_grid_custom, cv_result_trimmer


with open('config.yaml','r') as conf:
    try:
        cfg = yaml.safe_load(conf)
    except yaml.YAMLError as exc:
        print(exc)


# Data configuration
data_dir = cfg['paths']['data']
data_file = cfg['files']['input_file']
data_path = pathlib.Path(f"{data_dir}/{data_file}")

data = data_loader(data_path)

y = data[['Diabetic']].values.ravel()
X = data.drop('Diabetic', axis=1)


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


preprocessor = create_preprocessor()


pipe = Pipeline([
    ('preprocess', preprocessor),
    ('estimator', None)
])

params = cv_grid_custom(cfg['models'])


#grid_params = [
#    {
#    'estimator' : [RandomForestClassifier()],
#    'estimator__max_depth' : [10,15]
#    }
#]


scoring = {
        'Accuracy':'accuracy',
        'Precision':'precision',
        'Recall':'recall',
        'AUC':'roc_auc'
        }


grid = GridSearchCV(pipe, params, scoring = scoring, refit='AUC') 

grid.fit(X_train, y_train)


results = cv_result_trimmer(pd.DataFrame(grid.cv_results_))




#print(params)
#for key, val in params.items():
#    module = importlib.import_module(val,key)
#    lr = getattr(module, key)
#    lr = model()

#from sklearn.linear_model import LinearRegression
#print(lr)
#test = LinearRegression()

#X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
#y = np.dot(X, np.array([1, 2])) + 3

#reg = lr.fit(X,y)