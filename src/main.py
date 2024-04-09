import yaml
import importlib
import numpy as np
import pandas as pd

with open('config.yaml','r') as conf:
    try:
        params = yaml.safe_load(conf)
    except yaml.YAMLError as exc:
        print(exc)

print(params)
for key, val in params.items():
    module = importlib.import_module(val,key)
    lr = getattr(module, key)
    lr = model()

from sklearn.linear_model import LinearRegression
print(lr)
test = LinearRegression()

print(test)

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

reg = lr.fit(X,y)