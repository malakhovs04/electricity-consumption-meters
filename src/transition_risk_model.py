import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


class TransitionRiskModel:
    """
    Выявляет счётчики, у которых профиль потребления изменился
    между первой и второй половиной наблюдений, и обучает модель,
    которая по признакам первого периода предсказывает вероятность
    того, что второй период окажется аномальным.

    Логика
    ------
    1. label_periods():
        Каждый период каждого счётчика помечается как normal/anomaly
        через Isolation Forest, обученный на признаках всех периодов
        вместе. contamination задаёт ожидаемую долю аномальных
        периодов (по умолчанию 5%).

    2. build_transition_dataset():
        Для каждого счётчика, у которого есть оба периода (0 и 1),
        строится одна обучающая строка:
            X = признаки периода 0 (первая половина)
            y = 1, если период 1 помечен как аномалия (-1 IF), иначе 0

        Это позволяет обучить модель ранней диагностики:
        "по тому, как счётчик работал в первой половине, можно ли
        предсказать, что во второй половине что-то пойдёт не так?"

    3. train():
        RandomForestClassifier (class_weight="balanced") обучается на
        этом датасете. Разбивка train/test производится по meter_id —
        тестовые счётчики целиком не видны при обучении (честная оценка
        обобщения на новые объекты, а не утечка данных).

    4. predict_risk():
        Для счётчиков, у которых есть только период 0 (или у которых
        period_1 помечен как нормальный), возвращает вероятность того,
        что следующий период будет аномальным — "early warning score".
    """

    FEATURE_COLUMNS = [
        "mean_load", "std_load", "peak_factor", "load_factor",
        "cv", "zero_ratio", "entropy",
        "night_day_ratio", "max_gap_hours", "large_gap_ratio",
    ]

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest = None
        self.classifier = None

    # ----------------------------------------------------------
    # Шаг 1: разметка периодов как normal / anomaly
    # ----------------------------------------------------------

    def label_periods(self, period_features_df: pd.DataFrame) -> pd.DataFrame:

        df = period_features_df.copy()
        X = df[self.FEATURE_COLUMNS].fillna(0.0)

        self.iso_forest = IsolationForest(
            n_estimators=300,
            contamination=self.contamination,
            random_state=self.random_state,
        )

        df["period_anomaly"]       = self.iso_forest.fit_predict(X)
        df["period_anomaly_score"] = self.iso_forest.score_samples(X)

        n_anomaly = (df["period_anomaly"] == -1).sum()
        n_total   = len(df)
        print(f"  Аномальных периодов: {n_anomaly} / {n_total} ({100 * n_anomaly / n_total:.1f}%)")

        return df

    # ----------------------------------------------------------
    # Шаг 2: датасет переходов (признаки period_0 → метка period_1)
    # ----------------------------------------------------------

    def build_transition_dataset(self, labeled_df: pd.DataFrame):

        period_0 = labeled_df[labeled_df["period_idx"] == 0].copy()
        period_1 = labeled_df[labeled_df["period_idx"] == 1].copy()

        # оставляем только счётчики, у которых есть оба периода
        common_meters = set(period_0["meter_id"]) & set(period_1["meter_id"])
        print(f"  Счётчиков с обоими периодами: {len(common_meters)}")

        period_0 = period_0[period_0["meter_id"].isin(common_meters)]
        period_1 = period_1[period_1["meter_id"].isin(common_meters)]

        period_1_labels = (
            period_1
            .set_index("meter_id")["period_anomaly"]
        )

        # target = 1 если второй период аномальный
        period_0 = period_0.copy()
        period_0["target"] = (
            period_0["meter_id"]
            .map(period_1_labels)
            .apply(lambda x: 1 if x == -1 else 0)
        )

        X    = period_0[self.FEATURE_COLUMNS].fillna(0.0)
        y    = period_0["target"]
        meta = period_0[["meter_id", "consumer_class", "period_start", "period_end"]].copy()

        print(f"  Датасет переходов: {X.shape[0]} строк")
        print(f"  Переходов в аномалию (target=1): {y.sum()} ({100 * y.mean():.1f}%)")

        return X, y, meta

    # ----------------------------------------------------------
    # Шаг 3: обучение классификатора
    # ----------------------------------------------------------

    def train(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
              test_size: float = 0.25):

        if y.nunique() < 2:
            raise ValueError(
                "В датасете переходов только один класс — "
                "нельзя обучить классификатор. "
                "Попробуйте увеличить contamination или n_periods."
            )

        unique_meters = meta["meter_id"].unique()
        train_meters, test_meters = train_test_split(
            unique_meters, test_size=test_size, random_state=self.random_state
        )

        train_mask = meta["meter_id"].isin(train_meters)
        test_mask  = meta["meter_id"].isin(test_meters)

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        self.classifier = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.classifier.fit(X_train, y_train)

        y_pred  = self.classifier.predict(X_test)
        y_proba = self.classifier.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred)
        auc    = roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else float("nan")

        return {
            "report":       report,
            "roc_auc":      auc,
            "y_test":       y_test,
            "y_proba":      y_proba,
            "X_test":       X_test,
            "meta_test":    meta[test_mask],
            "train_meters": train_meters,
            "test_meters":  test_meters,
        }

    # ----------------------------------------------------------
    # Шаг 4: вероятность риска по первому периоду каждого счётчика
    # ----------------------------------------------------------

    def predict_risk(self, period_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Для каждого счётчика берёт период 0 (первую половину наблюдений)
        и возвращает вероятность того, что вторая половина будет аномальной.
        Работает в том числе для новых счётчиков, которые не были в обучении.
        """
        if self.classifier is None:
            raise RuntimeError("Вызовите train() перед predict_risk().")

        period_0 = (
            period_features_df[period_features_df["period_idx"] == 0]
            .copy()
        )

        X = period_0[self.FEATURE_COLUMNS].fillna(0.0)
        period_0["transition_risk_proba"] = self.classifier.predict_proba(X)[:, 1]

        return (
            period_0[[
                "meter_id", "consumer_class",
                "period_start", "period_end",
                "transition_risk_proba",
            ]]
            .sort_values("transition_risk_proba", ascending=False)
            .reset_index(drop=True)
        )