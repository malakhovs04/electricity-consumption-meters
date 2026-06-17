import pandas as pd
import numpy as np


class RiskScorer:
    def __init__(self):
        pass

    def calculate_risk(self, anomaly_scores_df, meter_profiles_df, anomaly_features_df):

        scores = anomaly_scores_df.copy()
        profiles = meter_profiles_df.copy()
        features = anomaly_features_df.copy()

        if scores.empty:
            raise ValueError("anomaly_scores_df is empty — нет данных для расчета риска")

        scores['meter_id'] = scores['meter_id'].astype(str).str.strip()
        profiles['meter_id'] = profiles['meter_id'].astype(str).str.strip()
        features['meter_id'] = features['meter_id'].astype(str).str.strip()

        duplicate_cols = [col for col in features.columns if col in scores.columns and col != 'meter_id']
        if duplicate_cols:
            features = features.drop(columns=duplicate_cols)

        duplicate_cols_profiles = [col for col in profiles.columns if col in scores.columns and col != 'meter_id']
        if duplicate_cols_profiles:
            profiles = profiles.drop(columns=duplicate_cols_profiles)

        df = pd.merge(scores, profiles, on='meter_id', how='left')
        df = pd.merge(df, features, on='meter_id', how='left')

        required_cols = ['anomaly_score', 'cv', 'night_day_ratio', 'max_gap_hours', 'anomaly']
        for col in required_cols:
            if col not in df.columns:
                if f"{col}_x" in df.columns:
                    df[col] = df[f"{col}_x"]
                elif f"{col}_y" in df.columns:
                    df[col] = df[f"{col}_y"]
                else:
                    df[col] = 0

        for col in required_cols:
            df[col] = df[col].fillna(0)

        def normalize(series):
            min_v, max_v = series.min(), series.max()
            if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
                return pd.Series(0, index=series.index)
            return (series - min_v) / (max_v - min_v)

        df['norm_anomaly'] = 1 - normalize(df['anomaly_score'])

        df['norm_cv'] = normalize(df['cv'])

        df['norm_nd'] = normalize(df['night_day_ratio'])

        df['norm_gap'] = normalize(df['max_gap_hours'])

        w_anomaly = 0.4   # Форма профиля нагрузки
        w_cv = 0.2        # Нестабильность / хаотичность
        w_nd = 0.2        # Подозрительная ночная активность
        w_gap = 0.2       # Технические манипуляции / разрывы

        df['risk_score'] = (
            w_anomaly * df['norm_anomaly'] +
            w_cv * df['norm_cv'] +
            w_nd * df['norm_nd'] +
            w_gap * df['norm_gap']
        ) * 100

        df['risk_score'] = df['risk_score'].round(2)

        def get_primary_reason(row):
            if row['anomaly'] == 1 and row['risk_score'] < 40:
                return "Нет значимых отклонений"

            if row['anomaly'] == -1 and row['norm_anomaly'] > 0.75:
                return "Критическое искажение формы профиля потребления (Isolation Forest)"

            reasons = {
                "Высокая хаотичность и нестабильность потребления (CV)": row['norm_cv'],
                "Аномально высокий коэффициент ночного потребления (Night/Day)": row['norm_nd'],
                "Длительные подозрительные разрывы в передаче данных": row['norm_gap']
            }
            return max(reasons, key=reasons.get)

        df['primary_reason'] = df.apply(get_primary_reason, axis=1)

        final_cols = ['meter_id', 'risk_score', 'primary_reason', 'anomaly', 'anomaly_score']
        if 'consumer_class' in df.columns:
            final_cols.insert(1, 'consumer_class')

        final_report = df[final_cols].sort_values(by='risk_score', ascending=False)

        return final_report