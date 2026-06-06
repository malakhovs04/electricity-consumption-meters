import pandas as pd
from pathlib import Path
from src.data_cleaner import PowerSignalsFixer


class DatasetLoader:

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.cleaner = PowerSignalsFixer()

    def load_all(self):

        all_data = []

        for file in self.data_dir.glob("*.csv"):
            print(f"Loading: {file.name}")
            df = pd.read_csv(file, sep=";")

            df = df.rename(columns={
                "device_id": "meter_id",
                "date": "timestamp",

                "active_plus": "A+",
                "active_minus": "A-",

                "reactive_plus": "R+",
                "reactive_minus": "R-",
            })

            df = self.cleaner.fix_columns(df)
            df["consumer_class"] = file.stem
            all_data.append(df)

        return pd.concat(all_data, ignore_index=True)