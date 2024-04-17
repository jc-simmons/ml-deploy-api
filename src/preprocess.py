from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

def create_preprocessor(features):

    prep_pipe = Pipeline([
        ('dropna', Cleaner()),
        ('selector', DropFeatureSelector(features['drop']))
        ])

    numeric_features = features['numeric']

    numeric_transformer = Pipeline([
        ('minmax_scaler', MinMaxScaler())
        ])

    run_pipe =  ColumnTransformer([
            ('num', numeric_transformer, numeric_features)], remainder='passthrough')


    preprocessor = Pipeline([
        ('prep', prep_pipe),
        ('main', run_pipe)
        ])

    return preprocessor



class Cleaner(BaseEstimator, TransformerMixin):
    
    def __init__(self): 
        return 

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        X = X.dropna()
        return X


class DropFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X_dropped = X.drop(self.variables, axis = 1)
        return X_dropped