import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class AnomalyTypeClassifier:
    """
    Кластеризует счетчики, помеченные Isolation Forest как аномальные
    (anomaly == -1), на интерпретируемые подтипы аномалий.

    Подход:
        1. Для каждого аномального счетчика собирается набор признаков,
           описывающих характер отклонения (нестабильность, ночная
           активность, разрывы данных, простои и т.д.).
        2. Признаки масштабируются и кластеризуются методом KMeans.
        3. Каждому кластеру присваивается человекочитаемое название
           на основе того, какой признак сильнее всего отличает
           центроид кластера от среднего по всей выборке аномалий
           (rule-based labeling, объяснимость для ИТМО).

    Названия типов — это гипотезы для дальнейшего ручного разбора,
    а не точный диагноз: модель только подсказывает, в какую сторону
    смотреть инженеру/аналитику.
    """

    # Признаки, по которым кластеризуем аномалии.
    # Берутся из anomaly_features_df (cv, zero_ratio, entropy,
    # peak_factor, load_factor) и meter_profiles_df
    # (night_day_ratio, large_gap_ratio, max_gap_hours).
    CLUSTER_FEATURES = [
        "cv",
        "zero_ratio",
        "entropy",
        "peak_factor",
        "load_factor",
        "night_day_ratio",
        "large_gap_ratio",
        "max_gap_hours",
    ]

    # Правила интерпретации: если у кластера сильнее всего (по z-score)
    # выделяется признак FEATURE в направлении "high"/"low" — присваиваем LABEL.
    # Порядок важен: правила проверяются по очереди, побеждает первое совпадение
    # среди топ-признаков кластера.
    LABEL_RULES = [
        ("large_gap_ratio", "high",
         "Неисправность счетчика / сбои передачи данных"),
        ("max_gap_hours", "high",
         "Неисправность счетчика / сбои передачи данных"),
        ("zero_ratio", "high",
         "Длительные простои / отключение объекта"),
        ("night_day_ratio", "high",
         "Несанкционированное потребление (аномально высокая ночная нагрузка)"),
        ("cv", "high",
         "Нестабильный, скачкообразный режим потребления"),
        ("entropy", "high",
         "Нестабильный, скачкообразный режим потребления"),
        ("peak_factor", "high",
         "Резкие пиковые нагрузки / возможная смена режима работы"),
        ("load_factor", "high",
         "Равномерно заниженное потребление (возможное занижение показаний)"),
        ("load_factor", "low",
         "Резкие пиковые нагрузки / возможная смена режима работы"),
        ("cv", "low",
         "Подозрительно ровный, «идеальный» профиль потребления"),
    ]

    DEFAULT_LABEL = "Нетипичная аномалия (требует ручного разбора)"

    # Признаки, особо склонные к экстремальным выбросам (разрывы
    # в данных, ночное/дневное соотношение при делении на малые числа).
    # Их клипуем по перцентилям перед кластеризацией, чтобы 1-2 выброса
    # не "забивали" всю шкалу после масштабирования.
    WINSORIZE_FEATURES = ["max_gap_hours", "night_day_ratio"]
    WINSORIZE_LOWER_Q = 0.01
    WINSORIZE_UPPER_Q = 0.95

    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    # ------------------------------------------------------------
    # Winsorization (clip outliers)
    # ------------------------------------------------------------

    def _winsorize(self, df):
        df = df.copy()
        for col in self.WINSORIZE_FEATURES:
            if col not in df.columns:
                continue
            lower = df[col].quantile(self.WINSORIZE_LOWER_Q)
            upper = df[col].quantile(self.WINSORIZE_UPPER_Q)
            df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    # ------------------------------------------------------------
    # Подготовка данных
    # ------------------------------------------------------------

    def _prepare_data(self, risk_report, anomaly_features_df, meter_profiles_df):

        risk = risk_report.copy()
        features = anomaly_features_df.copy()
        profiles = meter_profiles_df.copy()

        for d in (risk, features, profiles):
            d["meter_id"] = d["meter_id"].astype(str).str.strip()

        anomalies = risk[risk["anomaly"] == -1].copy()

        if anomalies.empty:
            return anomalies

        profile_cols = ["meter_id", "night_day_ratio", "large_gap_ratio", "max_gap_hours"]
        profile_cols = [c for c in profile_cols if c in profiles.columns]

        merged = anomalies.merge(
            features[["meter_id"] + [c for c in self.CLUSTER_FEATURES if c in features.columns]],
            on="meter_id", how="left"
        )
        merged = merged.merge(
            profiles[profile_cols],
            on="meter_id", how="left"
        )

        for col in self.CLUSTER_FEATURES:
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = merged[col].fillna(0.0)

        return merged

    # ------------------------------------------------------------
    # Подбор числа кластеров (для отчета/анализа)
    # ------------------------------------------------------------

    def suggest_n_clusters(self, X_scaled, k_range=range(2, 7)):
        """
        Возвращает silhouette score для разных k.

        X_scaled должен быть подготовлен так же, как внутри fit_predict:
        winsorize(WINSORIZE_FEATURES) + RobustScaler по CLUSTER_FEATURES.
        Полезно вызвать отдельно при разведочном анализе, чтобы
        обосновать выбор self.n_clusters.
        """
        scores = {}
        n_samples = X_scaled.shape[0]

        for k in k_range:
            if k >= n_samples:
                continue
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(X_scaled)
            if len(set(labels)) < 2:
                continue
            scores[k] = silhouette_score(X_scaled, labels)

        return scores

    # ------------------------------------------------------------
    # Присвоение интерпретируемых названий кластерам
    # ------------------------------------------------------------

    def _label_clusters(self, X_scaled_df, labels):
        """
        Для каждого кластера ищет признак, чей средний z-score
        сильнее всего отклоняется от 0 (от среднего по всем аномалиям),
        и сопоставляет ему текстовый тип из LABEL_RULES.
        """
        X_scaled_df = X_scaled_df.copy()
        X_scaled_df["cluster"] = labels

        cluster_means = X_scaled_df.groupby("cluster")[self.CLUSTER_FEATURES].mean()

        label_map = {}
        for cluster_id, row in cluster_means.iterrows():
            # признаки, отсортированные по убыванию |z-score|
            ranked = row.abs().sort_values(ascending=False)

            assigned = None
            for feature in ranked.index:
                direction = "high" if row[feature] > 0 else "low"
                for rule_feature, rule_dir, label in self.LABEL_RULES:
                    if rule_feature == feature and rule_dir == direction:
                        assigned = label
                        break
                if assigned is not None:
                    break

            label_map[cluster_id] = assigned if assigned is not None else self.DEFAULT_LABEL

        return label_map, cluster_means

    # ------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------

    def fit_predict(self, risk_report, anomaly_features_df, meter_profiles_df):
        """
        Возвращает:
            anomalies_df  — таблица аномальных счетчиков с колонками
                            'cluster' и 'anomaly_type'
            cluster_means — средние z-score признаков по кластерам
                            (для отчета/обоснования меток)
        """
        data = self._prepare_data(risk_report, anomaly_features_df, meter_profiles_df)

        if data.empty:
            empty = data.copy()
            empty["cluster"] = pd.Series(dtype=int)
            empty["anomaly_type"] = pd.Series(dtype=str)
            return empty, pd.DataFrame()

        n_clusters = min(self.n_clusters, len(data))

        data_winsorized = self._winsorize(data)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(data_winsorized[self.CLUSTER_FEATURES])
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.CLUSTER_FEATURES, index=data.index)

        if n_clusters < 2:
            data["cluster"] = 0
            data["anomaly_type"] = self.DEFAULT_LABEL
            return data, pd.DataFrame()

        model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = model.fit_predict(X_scaled)

        data["cluster"] = labels

        label_map, cluster_means = self._label_clusters(X_scaled_df, labels)
        data["anomaly_type"] = data["cluster"].map(label_map)

        # для визуализации сохраняем 2D-проекцию (PCA) прямо здесь, чтобы
        # не пересчитывать масштабирование в визуализаторе
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=self.random_state)
        coords = pca.fit_transform(X_scaled)
        data["pca_x"] = coords[:, 0]
        data["pca_y"] = coords[:, 1]

        return data, cluster_means