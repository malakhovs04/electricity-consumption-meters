import json
from typing import Optional
from collections import OrderedDict

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path


app = FastAPI(title="Electricity Consumption Monitor")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

RAW_DATA_DIR = Path("data/raw")

ENERGY_COLUMNS = ["A+", "A-", "R+", "R-"]


# ------------------------------------------------------------------
# Загрузка данных
# ------------------------------------------------------------------

def load_data():
    base = Path("data/processed")

    risk = pd.read_csv(base / "final_risk_report.csv")
    risk["meter_id"] = risk["meter_id"].astype(str).str.strip()

    anomaly_types_path = base / "final_risk_report_with_types.csv"
    if anomaly_types_path.exists():
        risk_types = pd.read_csv(anomaly_types_path)
        risk_types["meter_id"] = risk_types["meter_id"].astype(str).str.strip()
        risk = risk.merge(
            risk_types[["meter_id", "anomaly_type"]],
            on="meter_id", how="left"
        )

    transition_path = base / "transition_risk_current.csv"
    transition = None
    if transition_path.exists():
        transition = pd.read_csv(transition_path)
        transition["meter_id"] = transition["meter_id"].astype(str).str.strip()
        transition["period_start"] = pd.to_datetime(transition["period_start"])
        transition["period_end"]   = pd.to_datetime(transition["period_end"])

    profiles_path = base / "meter_profiles.csv"
    profiles = None
    if profiles_path.exists():
        profiles = pd.read_csv(profiles_path)
        profiles["meter_id"] = profiles["meter_id"].astype(str).str.strip()

    return risk, transition, profiles


RISK_DF, TRANSITION_DF, PROFILES_DF = load_data()


# ------------------------------------------------------------------
# Индекс meter_id -> файл (строится один раз при старте, без чтения
# содержимого файлов целиком — только список meter_id из каждого файла)
# ------------------------------------------------------------------

def build_meter_file_index():
    """
    Возвращает dict {meter_id: Path}.
    Использует только колонку device_id (быстрее, чем читать весь файл),
    поэтому не требует загрузки A+/A-/R+/R- на этом этапе.
    """
    index = {}
    if not RAW_DATA_DIR.exists():
        return index

    for file in RAW_DATA_DIR.glob("*.csv"):
        try:
            ids = pd.read_csv(file, sep=";", usecols=["device_id"])["device_id"]
        except Exception:
            continue
        for meter_id in ids.astype(str).str.strip().unique():
            index[meter_id] = file

    return index


METER_FILE_INDEX = build_meter_file_index()
HAS_RAW_DATA = len(METER_FILE_INDEX) > 0

# Кэш полных сырых временных рядов по счётчику (после первого открытия
# карточки повторные запросы агрегации/энергии/диапазона дат — мгновенные)
_RAW_TIMESERIES_CACHE: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
_RAW_CACHE_MAX_SIZE = 200


def load_meter_raw_timeseries(meter_id: str) -> Optional[pd.DataFrame]:
    """
    Возвращает ВСЕ 4 вида энергии (A+, A-, R+, R-) для счётчика,
    используя индекс meter_id -> файл вместо перебора всех CSV.
    Результат кэшируется (LRU, до _RAW_CACHE_MAX_SIZE счётчиков).
    """
    if meter_id in _RAW_TIMESERIES_CACHE:
        _RAW_TIMESERIES_CACHE.move_to_end(meter_id)
        return _RAW_TIMESERIES_CACHE[meter_id]

    file = METER_FILE_INDEX.get(meter_id)
    if file is None:
        return None

    try:
        df = pd.read_csv(file, sep=";")
    except Exception:
        return None

    df = df.rename(columns={
        "device_id": "meter_id",
        "date": "timestamp",
        "active_plus": "A+", "active_minus": "A-",
        "reactive_plus": "R+", "reactive_minus": "R-",
    })

    meter_rows = df[df["meter_id"].astype(str).str.strip() == meter_id]
    if meter_rows.empty:
        return None

    meter_rows = meter_rows.copy()
    if pd.api.types.is_numeric_dtype(meter_rows["timestamp"]):
        meter_rows["timestamp"] = pd.to_datetime(meter_rows["timestamp"], unit="s")
    else:
        meter_rows["timestamp"] = pd.to_datetime(meter_rows["timestamp"])

    # коррекция A+/R+ — та же логика что в data_cleaner.PowerSignalsFixer
    a_plus = meter_rows["A+"].copy()
    r_plus = meter_rows["R+"].copy()
    meter_rows["A+"] = r_plus
    meter_rows["R+"] = a_plus

    result = meter_rows.sort_values("timestamp")[["timestamp"] + ENERGY_COLUMNS]

    _RAW_TIMESERIES_CACHE[meter_id] = result
    _RAW_TIMESERIES_CACHE.move_to_end(meter_id)
    if len(_RAW_TIMESERIES_CACHE) > _RAW_CACHE_MAX_SIZE:
        _RAW_TIMESERIES_CACHE.popitem(last=False)

    return result


def get_meter_date_bounds(meter_id: str) -> Optional[dict]:
    """Возвращает min/max дату наблюдений для счётчика (для календаря)."""
    raw_df = load_meter_raw_timeseries(meter_id)
    if raw_df is None or raw_df.empty:
        return None
    return {
        "min_date": raw_df["timestamp"].min().strftime("%Y-%m-%d"),
        "max_date": raw_df["timestamp"].max().strftime("%Y-%m-%d"),
    }


# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------

def fig_to_json(fig) -> str:
    """
    Сериализует Plotly figure в JSON для передачи в браузер.

    ВАЖНО: начиная с Plotly 6.x, если в trace передать pandas.Series или
    numpy.ndarray (а не чистый python list) для x/y/и т.д., то даже
    fig.to_plotly_json() возвращает уже ГОТОВЫЙ объект вида
    {"dtype": "f8", "bdata": "<base64>"} вместо обычного списка чисел.
    Наши шаблоны вставляют JSON прямо в <script> через Jinja
    (`var fig = {{ chart | safe }}`) без декодирования base64 на
    клиенте — из-за этого Plotly.js получает мусорные координаты.

    Эта функция рекурсивно обходит fig.to_plotly_json() и:
      - распаковывает любой {"dtype": ..., "bdata": ...} обратно в
        обычный список чисел (через numpy.frombuffer + base64.b64decode);
      - приводит numpy/pandas скаляры и datetime64-массивы к обычным
        python-типам.

    Это работает независимо от того, как создан trace (go.Scatter с
    pandas.Series, px.scatter на DataFrame и т.д.) — последняя линия
    защиты от проблемы с bdata.
    """
    import base64
    import numpy as np

    def decode_bdata(obj):
        """Если obj — это {'dtype': ..., 'bdata': ...}, декодирует в list[float]."""
        raw = base64.b64decode(obj["bdata"])
        arr = np.frombuffer(raw, dtype=obj["dtype"])
        return [None if (isinstance(v, float) and np.isnan(v)) else float(v) for v in arr]

    def convert(obj):
        if isinstance(obj, dict):
            if set(obj.keys()) == {"dtype", "bdata"}:
                return decode_bdata(obj)
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        if isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.datetime64):
                return pd.Series(obj).dt.strftime("%Y-%m-%dT%H:%M:%S.%f").tolist()
            return [convert(v) for v in obj.tolist()]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, (np.datetime64, pd.Timestamp)):
            return pd.Timestamp(obj).strftime("%Y-%m-%dT%H:%M:%S.%f")
        return obj

    fig_dict = convert(fig.to_plotly_json())
    return json.dumps(fig_dict, default=str)


def risk_color(score: float) -> str:
    if score >= 70:
        return "danger"
    if score >= 40:
        return "warning"
    return "success"


PERIOD_LABELS = {
    "all":         "за весь период",
    "last_month":  "за последний месяц",
    "last_week":   "за последнюю неделю",
    "first_month": "за первый месяц",
    "first_week":  "за первую неделю",
    "custom":      "за выбранный период",
}

ENERGY_LABELS = {
    "A+": "Активная энергия, потребление (A+)",
    "A-": "Активная энергия, отдача (A-)",
    "R+": "Реактивная энергия, потребление (R+)",
    "R-": "Реактивная энергия, отдача (R-)",
}


def filter_by_period(
    df: pd.DataFrame,
    period: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """
    period: all | last_month | last_week | first_month | first_week | custom
    Для 'custom' использует date_from/date_to (YYYY-MM-DD).
    """
    if df.empty:
        return df

    data_min = df["timestamp"].min()
    data_max = df["timestamp"].max()

    if period == "last_month":
        df = df[df["timestamp"] >= data_max - pd.Timedelta(days=30)]
    elif period == "last_week":
        df = df[df["timestamp"] >= data_max - pd.Timedelta(days=7)]
    elif period == "first_month":
        df = df[df["timestamp"] <= data_min + pd.Timedelta(days=30)]
    elif period == "first_week":
        df = df[df["timestamp"] <= data_min + pd.Timedelta(days=7)]
    elif period == "custom":
        if date_from:
            df = df[df["timestamp"] >= pd.Timestamp(date_from)]
        if date_to:
            df = df[df["timestamp"] < pd.Timestamp(date_to) + pd.Timedelta(days=1)]
    # 'all' — без фильтра

    return df


def build_timeseries_chart(
    meter_id: str,
    energy: str = "A+",
    period: str = "all",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Optional[str]:
    """
    График потребления во времени. Агрегация выбирается автоматически
    по длине периода: один день -> сырые измерения, до месяца -> по
    суткам, больше -> по неделям. Это не даёт графику выглядеть как
    плоская линия при выборе всего периода и одновременно не рисует
    тысячи точек при выборе одного дня.
    """
    raw_df = load_meter_raw_timeseries(meter_id)
    if raw_df is None or raw_df.empty or energy not in raw_df.columns:
        return None

    df = filter_by_period(raw_df, period, date_from, date_to)
    if df.empty:
        return None

    span_days = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400

    df = df.set_index("timestamp")
    if period == "custom" and date_from and date_to and date_from == date_to:
        agg = df[[energy]].reset_index()
        mode_label = "raw"
    elif span_days <= 2:
        agg = df[[energy]].reset_index()
        mode_label = "raw"
    elif span_days <= 31:
        agg = df[[energy]].resample("D").mean().reset_index()
        mode_label = "day"
    elif span_days <= 120:
        agg = df[[energy]].resample("W").mean().reset_index()
        mode_label = "week"
    else:
        agg = df[[energy]].resample("ME").mean().reset_index()
        mode_label = "month"

    agg = agg.dropna(subset=[energy])
    if agg.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        y=agg[energy].astype(float).tolist(),
        mode="lines" if mode_label == "raw" else "lines+markers",
        line=dict(color="#4F8EF7", width=1.5 if mode_label == "raw" else 2),
        marker=dict(size=4),
        name=energy
    ))
    fig.update_layout(
        title=f"{ENERGY_LABELS.get(energy, energy)} — {PERIOD_LABELS.get(period, period)}",
        xaxis_title="Дата",
        yaxis_title=energy,
        margin=dict(l=40, r=20, t=50, b=40),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig_to_json(fig)


def build_weekday_chart_from_raw(
    meter_id: str,
    energy: str = "A+",
    period: str = "all",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Optional[str]:
    """Среднее потребление по дням недели за выбранный период."""
    raw_df = load_meter_raw_timeseries(meter_id)
    if raw_df is None or raw_df.empty or energy not in raw_df.columns:
        return None

    df = filter_by_period(raw_df, period, date_from, date_to)
    if df.empty:
        return None

    df = df.copy()
    df["weekday"] = df["timestamp"].dt.weekday
    weekday_means = df.groupby("weekday")[energy].mean().reindex(range(7))
    labels = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=[None if pd.isna(v) else float(v) for v in weekday_means.values],
        marker_color="#4F8EF7"
    ))
    fig.update_layout(
        title=f"Среднее по дням недели ({energy}), {PERIOD_LABELS.get(period, period)}",
        yaxis_title=energy,
        margin=dict(l=40, r=20, t=50, b=40),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig_to_json(fig)



def build_risk_scatter() -> Optional[str]:
    df = RISK_DF.copy()
    if df.empty or "anomaly_score" not in df.columns or "risk_score" not in df.columns:
        return None

    df = df.dropna(subset=["anomaly_score", "risk_score"])
    if df.empty:
        return None

    df["color_group"] = df["risk_score"].apply(
        lambda s: "Высокий (≥70)" if s >= 70
        else ("Средний (40-70)" if s >= 40 else "Низкий (<40)")
    )
    color_map = {
        "Высокий (≥70)": "#dc3545",
        "Средний (40-70)": "#fd7e14",
        "Низкий (<40)": "#28a745",
    }
    fig = px.scatter(
        df,
        x="anomaly_score",
        y="risk_score",
        color="color_group",
        color_discrete_map=color_map,
        hover_data=["meter_id", "consumer_class"],
        labels={
            "anomaly_score": "Anomaly Score (Isolation Forest)",
            "risk_score": "Risk Score",
            "color_group": "Уровень риска",
        },
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(
        height=480,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
        ),
    )
    return fig_to_json(fig)


def build_transition_hist() -> Optional[str]:
    if TRANSITION_DF is None or TRANSITION_DF.empty:
        return None

    # показываем распределение только по счётчикам, которые сейчас в норме —
    # для уже аномальных счётчиков transition_risk_proba не интерпретируема
    normal_meter_ids = set(RISK_DF[RISK_DF["anomaly"] == 1]["meter_id"])
    df = TRANSITION_DF[TRANSITION_DF["meter_id"].isin(normal_meter_ids)]
    df = df.dropna(subset=["transition_risk_proba"])
    if df.empty:
        return None

    fig = px.histogram(
        df,
        x="transition_risk_proba",
        nbins=30,
        color_discrete_sequence=["#4F8EF7"],
        labels={"transition_risk_proba": "Вероятность (transition_risk_proba)"},
    )
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig_to_json(fig)


# ------------------------------------------------------------------
# Маршруты
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def overview(
    request: Request,
    consumer_class: str = Query(default=""),
    risk_min: float = Query(default=0.0),
    risk_max: float = Query(default=100.0),
    only_anomalies: bool = Query(default=False),
    page: int = Query(default=1),
    sort_by: str = Query(default="risk_score"),
    sort_dir: str = Query(default="desc"),
):
    df = RISK_DF.copy()

    if consumer_class:
        df = df[df["consumer_class"] == consumer_class]
    df = df[(df["risk_score"] >= risk_min) & (df["risk_score"] <= risk_max)]
    if only_anomalies:
        df = df[df["anomaly"] == -1]

    sort_columns = {"risk_score", "anomaly_score"}
    if sort_by not in sort_columns:
        sort_by = "risk_score"
    ascending = sort_dir == "asc"
    df = df.sort_values(sort_by, ascending=ascending)

    page_size = 50
    total = len(df)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    df_page = df.iloc[(page - 1) * page_size : page * page_size]

    rows = []
    for _, r in df_page.iterrows():
        is_anomaly = r.get("anomaly", 1) == -1
        rows.append({
            "meter_id":       r["meter_id"],
            "consumer_class": r.get("consumer_class", "—"),
            "risk_score":     round(r["risk_score"], 1),
            "anomaly_score":  round(float(r.get("anomaly_score", 0)), 4),
            "anomaly":        r.get("anomaly", 1),
            "anomaly_type":   r.get("anomaly_type", "—") if is_anomaly else "Норма",
            "primary_reason": r.get("primary_reason", "—") if is_anomaly else "Нет значимых отклонений",
            "color":          risk_color(r["risk_score"]),
        })

    # сводная статистика
    total_meters  = len(RISK_DF)
    n_anomalies   = int((RISK_DF["anomaly"] == -1).sum())
    avg_risk      = round(RISK_DF["risk_score"].mean(), 1)
    high_risk     = int((RISK_DF["risk_score"] >= 70).sum())

    classes = sorted(RISK_DF["consumer_class"].dropna().unique().tolist())

    return templates.TemplateResponse("overview.html", {
        "request":          request,
        "rows":             rows,
        "total":            total,
        "page":             page,
        "total_pages":      total_pages,
        "consumer_class":   consumer_class,
        "risk_min":         risk_min,
        "risk_max":         risk_max,
        "only_anomalies":   only_anomalies,
        "classes":          classes,
        "total_meters":     total_meters,
        "n_anomalies":      n_anomalies,
        "avg_risk":         avg_risk,
        "high_risk":        high_risk,
        "sort_by":          sort_by,
        "sort_dir":         sort_dir,
    })


@app.get("/meter/{meter_id}", response_class=HTMLResponse)
async def meter_detail(request: Request, meter_id: str, source: str = Query(default="overview")):
    meter_id = meter_id.strip()

    risk_row = RISK_DF[RISK_DF["meter_id"] == meter_id]
    if risk_row.empty:
        return HTMLResponse("<h2>Счётчик не найден</h2>", status_code=404)

    r = risk_row.iloc[0]
    is_anomaly = int(r.get("anomaly", 1)) == -1

    # Вероятность перехода в аномалию имеет смысл ТОЛЬКО для счётчиков,
    # которые сейчас в норме: модель предсказывает её на основе первой
    # половины наблюдений, которая для уже-аномальных счётчиков не
    # репрезентативна (аномалия там уже наступила).
    transition_row = None
    if not is_anomaly and TRANSITION_DF is not None:
        t = TRANSITION_DF[TRANSITION_DF["meter_id"] == meter_id]
        if not t.empty:
            tr = t.iloc[0]
            transition_row = {
                "proba":        round(tr["transition_risk_proba"] * 100, 1),
                "period_start": tr["period_start"].strftime("%Y-%m-%d %H:%M"),
                "period_end":   tr["period_end"].strftime("%Y-%m-%d %H:%M"),
            }

    info = {
        "meter_id":       meter_id,
        "consumer_class": r.get("consumer_class", "—"),
        "risk_score":     round(r["risk_score"], 1),
        "anomaly":        int(r.get("anomaly", 1)),
        "anomaly_type":   r.get("anomaly_type", "—") if is_anomaly else "Норма",
        "primary_reason": r.get("primary_reason", "—") if is_anomaly else "Нет значимых отклонений",
        "anomaly_score":  round(float(r.get("anomaly_score", 0)), 4),
        "color":          risk_color(r["risk_score"]),
    }

    date_bounds = get_meter_date_bounds(meter_id) if HAS_RAW_DATA else None

    timeseries_chart = None
    weekday_chart = None

    if HAS_RAW_DATA:
        timeseries_chart = build_timeseries_chart(meter_id, energy="A+", period="all")
        weekday_chart     = build_weekday_chart_from_raw(meter_id, energy="A+", period="all")

    # "Назад" возвращает туда, откуда пришли — на /risk или на / (обзор)
    if source == "risk":
        back_url, back_label = "/risk", "к карте рисков"
    else:
        back_url, back_label = "/", "к списку"

    return templates.TemplateResponse("meter.html", {
        "request":          request,
        "info":             info,
        "transition":       transition_row,
        "timeseries_chart": timeseries_chart,
        "weekday_chart":    weekday_chart,
        "has_raw_data":     HAS_RAW_DATA,
        "date_bounds":      date_bounds,
        "energy_options":   ENERGY_COLUMNS,
        "back_url":         back_url,
        "back_label":       back_label,
    })


@app.get("/api/meter/{meter_id}/timeseries", response_class=JSONResponse)
async def meter_timeseries_api(
    meter_id: str,
    energy: str = Query(default="A+"),
    period: str = Query(default="all"),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
):
    """
    Блок: потребление во времени.

    energy: A+ | A- | R+ | R-
    period: all | last_month | last_week | first_month | first_week | custom
    date_from / date_to: YYYY-MM-DD — используются только при period=custom.
    """
    meter_id = meter_id.strip()

    valid_periods = {"all", "last_month", "last_week", "first_month", "first_week", "custom"}
    if energy not in ENERGY_COLUMNS or period not in valid_periods:
        return JSONResponse({"error": "invalid params"}, status_code=400)

    fig_json = build_timeseries_chart(meter_id, energy=energy, period=period,
                                       date_from=date_from, date_to=date_to)
    if fig_json is None:
        return JSONResponse({"error": "no data"}, status_code=404)
    return JSONResponse(json.loads(fig_json))


@app.get("/api/meter/{meter_id}/weekday-stats", response_class=JSONResponse)
async def meter_weekday_stats_api(
    meter_id: str,
    energy: str = Query(default="A+"),
    period: str = Query(default="all"),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
):
    """Блок: среднее по дням недели за выбранный период."""
    meter_id = meter_id.strip()

    valid_periods = {"all", "last_month", "last_week", "first_month", "first_week", "custom"}
    if energy not in ENERGY_COLUMNS or period not in valid_periods:
        return JSONResponse({"error": "invalid params"}, status_code=400)

    fig_json = build_weekday_chart_from_raw(meter_id, energy=energy, period=period,
                                             date_from=date_from, date_to=date_to)
    if fig_json is None:
        return JSONResponse({"error": "no data"}, status_code=404)
    return JSONResponse(json.loads(fig_json))


@app.get("/risk", response_class=HTMLResponse)
async def risk_map(request: Request):
    top_risk = (
        RISK_DF[RISK_DF["anomaly"] == -1]
        .sort_values("risk_score", ascending=False)
        .head(30)
    )

    top_rows = []
    for _, r in top_risk.iterrows():
        top_rows.append({
            "meter_id":       r["meter_id"],
            "consumer_class": r.get("consumer_class", "—"),
            "risk_score":     round(r["risk_score"], 1),
            "anomaly_type":   r.get("anomaly_type", "—"),
            "primary_reason": r.get("primary_reason", "—"),
            "color":          risk_color(r["risk_score"]),
        })

    scatter_chart     = build_risk_scatter()
    transition_chart  = build_transition_hist()

    by_class = (
        RISK_DF.groupby("consumer_class")["risk_score"]
        .mean()
        .round(1)
        .sort_values(ascending=False)
        .reset_index()
    )
    class_chart_fig = px.bar(
        by_class,
        x="consumer_class",
        y="risk_score",
        color="risk_score",
        color_continuous_scale=["#28a745", "#fd7e14", "#dc3545"],
        title="Средний Risk Score по классам потребителей",
        labels={"consumer_class": "Класс", "risk_score": "Risk Score"},
    )
    class_chart_fig.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=50, b=120),
        xaxis_tickangle=-45,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        coloraxis_showscale=False,
    )
    class_chart = fig_to_json(class_chart_fig)

    return templates.TemplateResponse("risk_map.html", {
        "request":          request,
        "top_rows":         top_rows,
        "scatter_chart":    scatter_chart,
        "transition_chart": transition_chart,
        "class_chart":      class_chart,
    })