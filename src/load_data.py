from pathlib import Path


def load_data(file_path: Path) -> pd.DataFrame:

    data = pd.read_csv('diabetes.csv')

    return data