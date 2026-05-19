import pandas as pd
import pathlib as Path
from .data_cleaner import PowerSignalsFixer


class DatasetLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.cleaner = PowerSignalsFixer()

    def load(self):
        all_data = []

        for file in self.data_dir.glob("*.csv"):
            df = pd.read_csv(file)
            df = self.cleaner.fix_colums(df)
            all_data.append(df)

        return pd.concat(all_data, ignore_index=True)