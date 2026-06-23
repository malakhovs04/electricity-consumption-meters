"""
Юнит-тесты для ключевых модулей проекта.

Покрывают граничные случаи (edge-cases):
    - пустой сигнал / все NaN
    - один счётчик
    - нулевые значения
    - корректность типов и формы выходных данных

Запуск:
    pytest tests/ -v
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureExtractor
from src.anomaly_features import AnomalyFeatureExtractor
from src.consumption_profiles import ConsumptionProfileBuilder
from src.risk_scorer import RiskScorer
from src.data_cleaner import PowerSignalsFixer


# ============================================================
# Фикстуры
# ============================================================

def make_raw_df(n_meters=3, n_measurements=48, seed=42):
    """Синтетический датафрейм с несколькими счётчиками."""
    rng = np.random.default_rng(seed)
    rows = []
    base_time = pd.Timestamp("2024-10-01")
    for i in range(n_meters):
        for j in range(n_measurements):
            rows.append({
                "meter_id":       f"meter_{i:02d}",
                "timestamp":      base_time + pd.Timedelta(hours=j * 0.5),
                "A+":             float(rng.uniform(0.1, 10.0)),
                "A-":             float(rng.uniform(0.0, 1.0)),
                "R+":             float(rng.uniform(0.0, 3.0)),
                "R-":             float(rng.uniform(0.0, 0.5)),
                "consumer_class": f"class_{i % 2}",
            })
    return pd.DataFrame(rows)


def make_anomaly_scores_df(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "meter_id":      [f"meter_{i:02d}" for i in range(n)],
        "anomaly":       rng.choice([-1, 1], size=n),
        "anomaly_score": rng.uniform(-0.8, -0.3, size=n),
    })


def make_profiles_df(n=10):
    rows = []
    for i in range(n):
        row = {"meter_id": f"meter_{i:02d}", "consumer_class": "class_0",
               "n_measurements": 48, "mean_gap_hours": 0.5,
               "max_gap_hours": 1.0, "large_gap_ratio": 0.0,
               "night_day_ratio": float(i * 0.1)}
        for h in range(24):
            row[f"hour_{h}"] = float(i + h * 0.1)
        for d in range(7):
            row[f"weekday_{d}"] = float(i + d * 0.2)
        rows.append(row)
    return pd.DataFrame(rows)


def make_anomaly_features_df(n=10, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "meter_id":    [f"meter_{i:02d}" for i in range(n)],
        "mean_load":   rng.uniform(1, 10, n),
        "std_load":    rng.uniform(0.1, 2, n),
        "peak_factor": rng.uniform(1, 5, n),
        "load_factor": rng.uniform(0.1, 1, n),
        "cv":          rng.uniform(0.1, 2, n),
        "zero_ratio":  rng.uniform(0, 0.3, n),
        "entropy":     rng.uniform(1, 4, n),
    })


# ============================================================
# PowerSignalsFixer
# ============================================================

class TestPowerSignalsFixer:

    def test_columns_swapped_correctly(self):
        df = pd.DataFrame({
            "meter_id":  ["m1"],
            "timestamp": [pd.Timestamp("2024-01-01")],
            "A+": [1.0], "A-": [2.0], "R+": [3.0], "R-": [4.0],
        })
        fixer = PowerSignalsFixer()
        result = fixer.fix_columns(df)
        # A+ должен получить значение из R+ (3.0), R+ — из A+ (1.0)
        assert result["A+"].iloc[0] == 3.0
        assert result["R+"].iloc[0] == 1.0
        assert result["A-"].iloc[0] == 2.0
        assert result["R-"].iloc[0] == 4.0

    def test_missing_column_raises(self):
        df = pd.DataFrame({"meter_id": ["m1"], "timestamp": ["2024-01-01"],
                           "A+": [1.0], "A-": [2.0], "R+": [3.0]})
        fixer = PowerSignalsFixer()
        with pytest.raises(ValueError, match="Missing column"):
            fixer.fix_columns(df)

    def test_output_columns_order(self):
        df = pd.DataFrame({
            "meter_id": ["m1"], "timestamp": [pd.Timestamp("2024-01-01")],
            "A+": [1.0], "A-": [2.0], "R+": [3.0], "R-": [4.0],
        })
        fixer = PowerSignalsFixer()
        result = fixer.fix_columns(df)
        assert list(result.columns) == ["meter_id", "timestamp", "A+", "A-", "R+", "R-"]


# ============================================================
# FeatureExtractor
# ============================================================

class TestFeatureExtractor:

    def test_output_shape(self):
        df = make_raw_df(n_meters=3, n_measurements=48)
        extractor = FeatureExtractor()
        result = extractor.extract(df)
        assert len(result) == 3
        assert "meter_id" in result.columns
        assert "consumer_class" in result.columns

    def test_statistical_features_normal(self):
        extractor = FeatureExtractor()
        signal = [1.0, 2.0, 3.0, 4.0, 5.0]
        feats = extractor.extract_statistical_features(signal, "test")
        assert feats["test_mean"] == pytest.approx(3.0, rel=1e-3)
        assert feats["test_min"] == pytest.approx(1.0)
        assert feats["test_max"] == pytest.approx(5.0)
        assert feats["test_std"] > 0

    def test_statistical_features_empty_signal(self):
        extractor = FeatureExtractor()
        feats = extractor.extract_statistical_features([], "test")
        assert feats["test_mean"] == 0
        assert feats["test_std"] == 0

    def test_statistical_features_all_nan(self):
        extractor = FeatureExtractor()
        feats = extractor.extract_statistical_features([np.nan, np.nan], "test")
        assert feats["test_mean"] == 0

    def test_frequency_features_short_signal(self):
        extractor = FeatureExtractor()
        feats = extractor.extract_frequency_features([1.0], "test")
        assert feats["test_fft_max"] == 0

    def test_electrical_features_empty(self):
        extractor = FeatureExtractor()
        feats = extractor.extract_electrical_features([], [])
        assert feats["apparent_power_mean"] == 0

    def test_no_nan_in_output(self):
        df = make_raw_df(n_meters=5, n_measurements=24)
        extractor = FeatureExtractor()
        result = extractor.extract(df)
        numeric_cols = result.select_dtypes(include=np.number).columns
        assert not result[numeric_cols].isnull().any().any()


# ============================================================
# AnomalyFeatureExtractor
# ============================================================

class TestAnomalyFeatureExtractor:

    def test_output_columns(self):
        df = make_raw_df(n_meters=2, n_measurements=30)
        extractor = AnomalyFeatureExtractor()
        result = extractor.extract(df)
        for col in ["meter_id", "mean_load", "cv", "zero_ratio", "entropy"]:
            assert col in result.columns

    def test_skips_all_nan_meter(self):
        df = make_raw_df(n_meters=2, n_measurements=10)
        df.loc[df["meter_id"] == "meter_00", "A+"] = np.nan
        extractor = AnomalyFeatureExtractor()
        result = extractor.extract(df)
        # счётчик с полным NaN пропускается
        assert "meter_00" not in result["meter_id"].values

    def test_zero_ratio_all_zeros(self):
        df = make_raw_df(n_meters=1, n_measurements=20)
        df["A+"] = 0.0
        extractor = AnomalyFeatureExtractor()
        result = extractor.extract(df)
        assert result["zero_ratio"].iloc[0] == pytest.approx(1.0)


# ============================================================
# ConsumptionProfileBuilder
# ============================================================

class TestConsumptionProfileBuilder:

    def test_output_has_hour_columns(self):
        df = make_raw_df(n_meters=2, n_measurements=96)
        builder = ConsumptionProfileBuilder()
        result = builder.build(df)
        for h in range(24):
            assert f"hour_{h}" in result.columns

    def test_output_has_weekday_columns(self):
        df = make_raw_df(n_meters=2, n_measurements=96)
        builder = ConsumptionProfileBuilder()
        result = builder.build(df)
        for d in range(7):
            assert f"weekday_{d}" in result.columns

    def test_one_row_per_meter(self):
        df = make_raw_df(n_meters=4, n_measurements=48)
        builder = ConsumptionProfileBuilder()
        result = builder.build(df)
        assert len(result) == 4

    def test_night_day_ratio_nonnegative(self):
        df = make_raw_df(n_meters=3, n_measurements=96)
        builder = ConsumptionProfileBuilder()
        result = builder.build(df)
        assert (result["night_day_ratio"] >= 0).all()

    def test_no_nan_in_gap_columns(self):
        df = make_raw_df(n_meters=2, n_measurements=48)
        builder = ConsumptionProfileBuilder()
        result = builder.build(df)
        assert not result["max_gap_hours"].isnull().any()
        assert not result["mean_gap_hours"].isnull().any()


# ============================================================
# RiskScorer
# ============================================================

class TestRiskScorer:

    def test_risk_score_range(self):
        scores = make_anomaly_scores_df(n=10)
        profiles = make_profiles_df(n=10)
        features = make_anomaly_features_df(n=10)
        scorer = RiskScorer()
        result = scorer.calculate_risk(scores, profiles, features)
        assert (result["risk_score"] >= 0).all()
        assert (result["risk_score"] <= 100).all()

    def test_output_has_required_columns(self):
        scores = make_anomaly_scores_df(n=5)
        profiles = make_profiles_df(n=5)
        features = make_anomaly_features_df(n=5)
        scorer = RiskScorer()
        result = scorer.calculate_risk(scores, profiles, features)
        for col in ["meter_id", "risk_score", "primary_reason", "anomaly"]:
            assert col in result.columns

    def test_sorted_by_risk_descending(self):
        scores = make_anomaly_scores_df(n=8)
        profiles = make_profiles_df(n=8)
        features = make_anomaly_features_df(n=8)
        scorer = RiskScorer()
        result = scorer.calculate_risk(scores, profiles, features)
        assert list(result["risk_score"]) == sorted(result["risk_score"], reverse=True)

    def test_empty_input_raises(self):
        scorer = RiskScorer()
        with pytest.raises(ValueError):
            scorer.calculate_risk(
                pd.DataFrame(),
                make_profiles_df(5),
                make_anomaly_features_df(5)
            )

    def test_meter_id_type_coercion(self):
        """meter_id как int в scores и str в profiles — должен смержиться без NaN."""
        scores = make_anomaly_scores_df(n=5)
        scores["meter_id"] = [0, 1, 2, 3, 4]  # int
        profiles = make_profiles_df(n=5)
        profiles["meter_id"] = ["0", "1", "2", "3", "4"]  # str
        features = make_anomaly_features_df(n=5)
        features["meter_id"] = ["0", "1", "2", "3", "4"]
        scorer = RiskScorer()
        result = scorer.calculate_risk(scores, profiles, features)
        assert len(result) == 5
        assert not result["risk_score"].isnull().any()