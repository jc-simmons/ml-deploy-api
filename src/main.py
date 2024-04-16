import yaml
import importlib
import numpy as np
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split

from load_data import data_loader
from preprocess import create_preprocessor
from models import Model


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

ml = Model(preprocessor, cfg['models']).gen()

ml.fit(X_train, y_train)

pd.set_option('display.max_colwidth', 200)
print(ml.chop_results_)

