import yaml
import pathlib
from sklearn.model_selection import train_test_split


from src.load_data import data_loader
from src.preprocess import create_preprocessor
from src.models import model
from src.postprocess import PostProcessor


def main():

    with open('config.yaml','r') as conf:
        try:
            config = yaml.safe_load(conf)
        except yaml.YAMLError as exc:
            print(exc)


    # Configuration
    DATA_PATH = pathlib.Path(f"{config['in']}")
    OUTPUT_DIR = config['out']
    SAVE_MODEL = config['save_model']
    TARGET_VAR = config['target']
    FEATURE_VAR = config['features']
    MODEL_PARAMS = config['models']

    data = data_loader(DATA_PATH)

    y = data[[TARGET_VAR]].values.ravel()
    X = data.drop(TARGET_VAR, axis=1)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

    preprocessor = create_preprocessor(FEATURE_VAR)

    ml = model(preprocessor, MODEL_PARAMS)

    ml.fit(X_train, y_train)

    PostProcessor.save_evaluate(OUTPUT_DIR, ml, X_test, y_test)

    if SAVE_MODEL:
        PostProcessor.save_model(OUTPUT_DIR, ml)



if __name__ == "__main__":
    main()