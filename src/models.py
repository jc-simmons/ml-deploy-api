from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
from scoring import scorer
import importlib

def model(preprocessor, estimators):
        
        est = CustomCVGrid(preprocessor,estimators)

        return est 
    

class CustomCVGrid:
    def __init__(self, preprocessor, models):
        self.preprocessor = preprocessor
        self.models = models

        pipe = Pipeline([
        ('preprocess', self.preprocessor),
        ('estimator', None)
        ])

        params = self.cv_grid_custom()
        self.base = GridSearchCV(pipe, params, scoring = scorer, refit='auc' )

   
    def cv_grid_custom(self) -> list:
        hyper_list = []

        for estimator in self.models:

            est = module_loader(estimator)
            model_dict = {'estimator': [est]}

            for param, vals in estimator['hyperparams'].items():
                model_dict['estimator__' + param] = vals
            
            hyper_list.append(model_dict)

        return hyper_list


    @property
    def chop_results_(self) -> pd.DataFrame: 
        cv_results = pd.DataFrame(self.cv_results_)
        results = cv_results.filter(regex='mean|params')

        for i in results.index:
            results.at[i,'params']['estimator'] = \
            str(type(results.at[i,'params']['estimator'])).split(".")[-1][:-2]

            for param in list(results.at[i,'params']):
                if 'estimator__' in param:
                    trim = param.replace('estimator__','')
                    val = results.at[i,'params'][param]
                    results.at[i,'params'].pop(param,None)
                    results.at[i,'params'][trim] = val


        return results


    def __getattr__(self, name):
        return getattr(self.base, name)


def module_loader(estimator: dict):

    module = importlib.import_module(
        estimator['module'],
        estimator['estimator'])

    model = getattr(module, estimator['estimator'] )

    return model()