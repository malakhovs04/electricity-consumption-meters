import pandas as pd

class PowerSignalsFixer:
    def __init__(self):
        self.mapping = {
            "A+":"R-",
            "A-":"A-",
            "R+":"A+",
            "R-":"R+",
        }

        self.inverse_mapping = {
            "R-":"A+",
            "A-":"A-",
            "A+":"R+",
            "R+":"R-",
        }
    
    def fix_colums(self, df):
        df = df.copy()

        ddf = df.rename(columns={
            "A+":"A+_raw",
            "A-":"A-_raw",
            "R+":"R+_raw",
            "R-":"R-_raw",
        })

        df["A+"] = ddf["A+_raw"]
        df["A-"] = ddf["A-_raw"]
        df["R+"] = ddf["R+_raw"]
        df["R-"] = ddf["R-_raw"]

        return df[['meter_id', 'timestamp', 'A+', 'A-', 'R+', 'R-']]