import pandas as pd


class PowerSignalsFixer:
    def __init__(self):
        pass

    def fix_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        expected_cols = ["A+", "A-", "R+", "R-"]

        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")


        a_plus_raw = df["A+"].copy()
        a_minus_raw = df["A-"].copy()
        r_plus_raw = df["R+"].copy()
        r_minus_raw = df["R-"].copy()

        df["A+"] = r_plus_raw
        df["A-"] = a_minus_raw
        df["R+"] = a_plus_raw
        df["R-"] = r_minus_raw

    
        return df[["meter_id", "timestamp", "A+", "A-", "R+", "R-"]]