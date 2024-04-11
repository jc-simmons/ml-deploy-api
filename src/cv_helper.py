import importlib
import pandas as pd


def custom_cvgrid(models: list) -> list:

    hyper_list = []

    for estimator in models:

        est = module_loader(estimator)

        model_dict = {
            'estimator': [est]
        }

        for param, vals in estimator['hyperparams'].items():


            model_dict['estimator__' + param] = vals
        
        hyper_list.append(model_dict)

    return hyper_list



def module_loader(estimator: dict):

    module = importlib.import_module(
        estimator['module'],
        estimator['estimator']
    )

    model = getattr(module, estimator['estimator'] )

    return model()




def cv_result_trimmer(results: pd.DataFrame) -> pd.DataFrame: 

    results = results.filter(regex='mean|params')

    for i in results.index:
        results.at[i,'params']['estimator'] = str(type(results.at[i,'params']['estimator'])).split(".")[-1][:-2]

    


    return results