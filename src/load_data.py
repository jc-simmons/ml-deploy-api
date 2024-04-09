from pathlib import Path
import pandas as pd


def data_loader(file_path: Path) -> pd.DataFrame:

    data = pd.read_csv(file_path)

    return data

