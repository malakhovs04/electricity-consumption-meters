import pandas as pd
import numpy as np


class RiskScorer:
    def __init__(self):
        pass

    def calculate_risk(self, anomaly_scores_df, meter_profiles_df, anomaly_features_df):
        """
        Сводит все метрики воедино и рассчитывает интегральный Risk Score (0-100).
        Сохранение итогового CSV выполняется на уровне main.py, не здесь.
        """
        scores = anomaly_scores_df.copy()
        profiles = meter_profiles_df.copy()
        features = anomaly_features_df.copy()

        if scores.empty:
            raise ValueError("anomaly_scores_df is empty — нет данных для расчета риска")

        # Приводим meter_id к единому строковому типу во всех таблицах и зачищаем пробелы
        scores['meter_id'] = scores['meter_id'].astype(str).str.strip()
        profiles['meter_id'] = profiles['meter_id'].astype(str).str.strip()
        features['meter_id'] = features['meter_id'].astype(str).str.strip()

        # Устранение дублирующихся колонок перед мерджем (избегаем суффиксов _x/_y)
        duplicate_cols = [col for col in features.columns if col in scores.columns and col != 'meter_id']
        if duplicate_cols:
            features = features.drop(columns=duplicate_cols)

        duplicate_cols_profiles = [col for col in profiles.columns if col in scores.columns and col != 'meter_id']
        if duplicate_cols_profiles:
            profiles = profiles.drop(columns=duplicate_cols_profiles)

        # Последовательное объединение таблиц через left join (scores — ведущая таблица)
        df = pd.merge(scores, profiles, on='meter_id', how='left')
        df = pd.merge(df, features, on='meter_id', how='left')

        # Резервная защита: если какая-то колонка все же раздвоилась, восстанавливаем оригинальное имя
        required_cols = ['anomaly_score', 'cv', 'night_day_ratio', 'max_gap_hours', 'anomaly']
        for col in required_cols:
            if col not in df.columns:
                if f"{col}_x" in df.columns:
                    df[col] = df[f"{col}_x"]
                elif f"{col}_y" in df.columns:
                    df[col] = df[f"{col}_y"]
                else:
                    df[col] = 0

        # NaN в ключевых метриках заменяем на 0, чтобы корректно считались min/max и веса
        for col in required_cols:
            df[col] = df[col].fillna(0)

        # --- Расчет интегрального Risk Score (0-100) ---

        def normalize(series):
            min_v, max_v = series.min(), series.max()
            if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
                return pd.Series(0, index=series.index)
            return (series - min_v) / (max_v - min_v)

        # Скор Isolation Forest (инвертируем: меньший скор = бóльшая аномальность)
        df['norm_anomaly'] = 1 - normalize(df['anomaly_score'])

        # Коэффициент вариации (cv)
        df['norm_cv'] = normalize(df['cv'])

        # Ночной коэффициент (night_day_ratio)
        df['norm_nd'] = normalize(df['night_day_ratio'])

        # Максимальные разрывы во времени (max_gap_hours)
        df['norm_gap'] = normalize(df['max_gap_hours'])

        # Взвешенная сумма для вычисления Risk Score (веса подобраны для баланса факторов)
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

        # Автоматическое определение главной причины риска (Explainable AI для ИТМО)
        def get_primary_reason(row):
            # счётчик нормальный и низкий риск — нет причины для тревоги
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

        # Формируем финальный отчет
        final_cols = ['meter_id', 'risk_score', 'primary_reason', 'anomaly', 'anomaly_score']
        if 'consumer_class' in df.columns:
            final_cols.insert(1, 'consumer_class')

        final_report = df[final_cols].sort_values(by='risk_score', ascending=False)

        return final_report