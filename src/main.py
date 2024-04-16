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
        config = yaml.safe_load(conf)
    except yaml.YAMLError as exc:
        print(exc)


# Data configuration
DATA_PATH = pathlib.Path(f"{config['in']}")
OUT_PATH = pathlib.Path(f"{config['out']}")

MODEL_PARAMS = config['models']
TARGET_VAR = config['target']

data = data_loader(DATA_PATH)

y = data[[TARGET_VAR]].values.ravel()
X = data.drop(TARGET_VAR, axis=1)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

preprocessor = create_preprocessor()

ml = Model(preprocessor, MODEL_PARAMS).gen()

ml.fit(X_train, y_train)



pd.set_option('display.max_colwidth', 200)
print(ml.chop_results_)

