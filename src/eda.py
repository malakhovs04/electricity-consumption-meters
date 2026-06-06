import pandas as pd
import numpy as np

class EDAAnalyzer:
    def __init__(self):
        pass

    def basic_info(self, df):
        print("\n===== DATASET INFO =====")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        print("\n===== COLUMN TYPES =====")
        print(df.dtypes)

        print("\n===== MISSING VALUES =====")
        print(df.isna().sum().sort_values(ascending=False))

    def numerical_summary(self, df):
        print("\n===== NUMERICAL SUMMARY =====")
        summary = df.describe().T
        print(summary)

        return summary
    
    def missing_report(self, df):
        missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent":
            100 * df.isna().sum() / len(df)})

        missing = missing.sort_values(
        by="missing_percent",
        ascending=False)

        return missing  
    
    def constant_features(self, df):
        constants = []

        for col in df.columns:
            if df[col].nunique() <= 1:
                constants.append(col)

        return constants
    
    def correlation_matrix(self, df):

        numeric_df = df.select_dtypes(
            include=np.number)
        corr = numeric_df.corr()

        return corr
    
    def class_distribution(self, df):
        print("\n===== CLASS DISTRIBUTION =====")
        distribution = (
            df["consumer_class"]
            .value_counts()
            .sort_index())

        print(distribution)
        print("\n===== PERCENTAGES =====")
        print(round(distribution / distribution.sum() * 100, 2))