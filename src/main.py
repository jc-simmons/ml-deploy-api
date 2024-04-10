import yaml
import importlib
import numpy as np
import pandas as pd
import pathlib


from load_data import data_loader
from preprocess import create_preprocessor

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