"""Financial charts using saved observations, with explicit units and horizons."""

import altair as alt
import pandas as pd
import streamlit as st
from dashboard_ui.data import read_csv
from dashboard_ui.disclosures import PORTFOLIOS

DEFAULT_CHART_HEIGHT = 240
DEFAULT_LINE_HEIGHT = 230
FORECAST_CHART_HEIGHT = 290
COMPARISON_CHART_HEIGHT = 270
ANNUAL_RETURN_CHART_HEIGHT = 210
EVT_DIAGNOSTIC_CHART_HEIGHT = 200
MIN_HORIZONTAL_BAR_HEIGHT = 350
HORIZONTAL_CATEGORY_THRESHOLD = 8

SINGLE_BAR_SIZE = 18
GROUPED_BAR_SIZE = 13
HORIZONTAL_BAR_SIZE = 10
ANNUAL_BAR_SIZE = 35
EXCEPTION_POINT_SIZE = 35

LINE_STROKE_WIDTH = 1.7
COMPARISON_STROKE_WIDTH = 1.6
LOSS_LINE_STROKE_WIDTH = 1.0
DRAWDOWN_AREA_OPACITY = 0.55

COLOR_LOSS_RED = "#ef717a"
COLOR_GAIN_GREEN = "#4cc49a"
COLOR_VAR_AMBER = "#e9b44c"
COLOR_REALIZED_GREY = "#8097b0"
COLOR_EVT_BLUE = "#56b4e9"

RISK_MEASURE_DOMAIN = ["VaR", "Expected Shortfall"]
RISK_MEASURE_RANGE = [COLOR_VAR_AMBER, COLOR_LOSS_RED]

LABELS = {
    "nav_usd": "Nominal net portfolio value",
    "drawdown": "Drawdown from prior peak",
    "var_usd": "VaR",
    "es_usd": "Expected Shortfall",
    "weight": "Target allocation",
    "var": "VaR threshold",
    "realized_hpl_loss": "Realized frozen-exposure loss",
    "terminal_loss_fraction": "Terminal loss",
    "maximum_drawdown_loss": "Maximum drawdown",
    "nominal_no_spending_usd": "Nominal wealth / no spending",
    "real_no_spending_base_usd": "Real wealth / Dec 2019 USD",
    "terminal_loss_usd": "Terminal loss contribution",
    "es_component": "ES contribution",
    "loss_usd": "Scenario loss",
}


def label(name: str) -> str:
    if name.startswith("end_weight_"):
        return name.removeprefix("end_weight_")
    return LABELS.get(name, name.replace("_", " ").capitalize())


def add_nan_segments(frame: pd.DataFrame, series_col: str = "Series") -> pd.Series:
    """Assign segment IDs across NaN gaps so Altair breaks lines cleanly."""
    return (
        frame.groupby(series_col, sort=False)
        .Value.transform(lambda s: s.isna().cumsum())
        .astype(str)
    )


def show(chart: alt.Chart, height: int = DEFAULT_CHART_HEIGHT) -> None:
    with alt.data_transformers.enable("default", max_rows=None):
        st.altair_chart(chart.properties(height=height), width="stretch", theme=None)


def _line_format_and_title(
    columns: list[str], percent: bool, usd: bool
) -> tuple[str, str]:
    is_usd = usd or any("usd" in col for col in columns)
    if percent:
        return ".1%", "Return / allocation"
    if is_usd:
        return "$,.0f", "USD"
    return ",.3f", "Value"


def lines(
    frame: pd.DataFrame,
    x: str,
    columns: list[str],
    percent: bool = False,
    usd: bool = False,
    height: int = DEFAULT_LINE_HEIGHT,
) -> None:
    melted = frame[[x, *columns]].melt(x, var_name="Series", value_name="Value")
    melted[x] = pd.to_datetime(melted[x])
    melted["segment"] = add_nan_segments(melted)
    melted["Series"] = melted.Series.map(label)
    fmt, y_title = _line_format_and_title(columns, percent, usd)
    chart = (
        alt.Chart(melted)
        .mark_line(strokeWidth=LINE_STROKE_WIDTH)
        .encode(
            x=alt.X(f"{x}:T", title="Valuation date"),
            y=alt.Y(
                "Value:Q",
                title=y_title,
                axis=alt.Axis(format=fmt),
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("Series:N", title=None),
            detail=["Series:N", "segment:N"],
            tooltip=[
                alt.Tooltip(f"{x}:T", title="Date"),
                "Series:N",
                alt.Tooltip("Value:Q", format=fmt),
            ],
        )
    )
    if columns == ["drawdown"]:
        chart = chart.mark_area(
            color=COLOR_LOSS_RED, opacity=DRAWDOWN_AREA_OPACITY
        ).encode(color=alt.value(COLOR_LOSS_RED))
    show(chart, height)


def _bar_format_and_title(percent: bool, usd: bool) -> tuple[str, str]:
    if percent:
        return ".1%", "Fraction / %"
    if usd:
        return "$,.0f", "USD loss"
    return ",.2f", "Value"


def _apply_vertical_bar_color(
    chart: alt.Chart, category: str, columns: list[str]
) -> alt.Chart:
    if len(columns) == 1:
        col_name = columns[0]
        if col_name == "weight":
            return chart.encode(color=alt.Color(f"{category}:N", legend=None))
        if "loss" in col_name or "component" in col_name:
            return chart.encode(
                color=alt.condition(
                    alt.datum.Value >= 0,
                    alt.value(COLOR_LOSS_RED),
                    alt.value(COLOR_GAIN_GREEN),
                )
            )
    elif "var_usd" in columns:
        return chart.encode(
            color=alt.Color(
                "Measure:N",
                scale=alt.Scale(domain=RISK_MEASURE_DOMAIN, range=RISK_MEASURE_RANGE),
            )
        )
    return chart


def _build_vertical_bars(
    melted: pd.DataFrame,
    category: str,
    columns: list[str],
    fmt: str,
    y_title: str,
) -> alt.Chart:
    bar_size = SINGLE_BAR_SIZE if len(columns) == 1 else GROUPED_BAR_SIZE
    chart = (
        alt.Chart(melted)
        .mark_bar(size=bar_size)
        .encode(
            x=alt.X(
                f"{category}:N",
                title=None,
                sort=None,
                axis=alt.Axis(labelAngle=0, labelLimit=180),
            ),
            y=alt.Y("Value:Q", title=y_title, axis=alt.Axis(format=fmt)),
            xOffset="Measure:N",
            color=alt.Color("Measure:N", title=None),
            tooltip=[
                alt.Tooltip(f"{category}:N"),
                "Measure:N",
                alt.Tooltip("Value:Q", format=fmt),
            ],
        )
    )
    return _apply_vertical_bar_color(chart, category, columns)


def _build_horizontal_bars(
    melted: pd.DataFrame, category: str, fmt: str, usd: bool
) -> alt.Chart:
    return (
        alt.Chart(melted)
        .mark_bar(size=HORIZONTAL_BAR_SIZE)
        .encode(
            y=alt.Y(
                f"{category}:N",
                title=None,
                sort=None,
                axis=alt.Axis(labelLimit=260),
            ),
            x=alt.X(
                "Value:Q",
                title="USD loss on USD 10m" if usd else "Value",
                axis=alt.Axis(format=fmt),
            ),
            yOffset="Measure:N",
            color=alt.Color(
                "Measure:N",
                scale=alt.Scale(domain=RISK_MEASURE_DOMAIN, range=RISK_MEASURE_RANGE),
            ),
            tooltip=[
                alt.Tooltip(f"{category}:N"),
                "Measure:N",
                alt.Tooltip("Value:Q", format=fmt),
            ],
        )
    )


def bars(
    frame: pd.DataFrame,
    category: str,
    columns: list[str],
    title: str | None = None,
    percent: bool = False,
    usd: bool = False,
    height: int = DEFAULT_CHART_HEIGHT,
) -> None:
    melted = frame[[category, *columns]].melt(
        category, var_name="Measure", value_name="Value"
    )
    melted["Measure"] = melted.Measure.map(label)
    fmt, y_title = _bar_format_and_title(percent, usd)
    if frame[category].nunique() > HORIZONTAL_CATEGORY_THRESHOLD:
        chart = _build_horizontal_bars(melted, category, fmt, usd)
        height = max(height, MIN_HORIZONTAL_BAR_HEIGHT)
    else:
        chart = _build_vertical_bars(melted, category, columns, fmt, y_title)
    if title:
        chart = chart.properties(title=title)
    show(chart, height)


def forecast_chart(frame: pd.DataFrame) -> None:
    base = alt.Chart(frame)
    loss = base.mark_line(
        color=COLOR_REALIZED_GREY, strokeWidth=LOSS_LINE_STROKE_WIDTH
    ).encode(
        x=alt.X("realization_date:T", title="Realization date"),
        y=alt.Y(
            "realized_hpl_loss:Q",
            title="One-session frozen-exposure loss",
            axis=alt.Axis(format=".1%"),
        ),
    )
    var = base.mark_line(
        color=COLOR_VAR_AMBER, strokeWidth=COMPARISON_STROKE_WIDTH
    ).encode(x="realization_date:T", y="var:Q")
    exceptions = (
        base.transform_filter(alt.datum.realized_hpl_loss > alt.datum.var)
        .mark_point(color=COLOR_LOSS_RED, filled=True, size=EXCEPTION_POINT_SIZE)
        .encode(
            x="realization_date:T",
            y="realized_hpl_loss:Q",
            tooltip=[
                alt.Tooltip("realization_date:T", title="Exception date"),
                alt.Tooltip("var:Q", format=".2%", title="VaR"),
                alt.Tooltip("realized_hpl_loss:Q", format=".2%", title="Realized loss"),
            ],
        )
    )
    show(
        (loss + var + exceptions).properties(
            title="FORECAST EXCEPTIONS / AMBER = VaR · GREY = REALIZED LOSS · RED = EXCEPTION"
        ),
        FORECAST_CHART_HEIGHT,
    )


def comparison_paths(ctx) -> None:
    parts = []
    for strategy, display_name in PORTFOLIOS.items():
        frame = read_csv(
            str(ctx.path),
            ctx.data["summary"]["result_digest"],
            f"{strategy}_cost_{ctx.bps:g}bps_daily.csv",
        )
        part = frame[["Date", "nav_usd"]].copy()
        part["Portfolio"] = display_name
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined.Date)
    chart = (
        alt.Chart(combined)
        .mark_line(strokeWidth=COMPARISON_STROKE_WIDTH)
        .encode(
            x=alt.X("Date:T", title="Valuation date"),
            y=alt.Y(
                "nav_usd:Q",
                title="Net wealth / USD 10m initial capital",
                axis=alt.Axis(format="$,.0f"),
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("Portfolio:N", title=None),
            tooltip=[
                alt.Tooltip("Date:T"),
                "Portfolio:N",
                alt.Tooltip("nav_usd:Q", title="Wealth", format="$,.0f"),
            ],
        )
    )
    show(chart, COMPARISON_CHART_HEIGHT)


def annual_returns(year: pd.Series) -> None:
    frame = year.rename("Return").rename_axis("Year").reset_index()
    show(
        alt.Chart(frame)
        .mark_bar(size=ANNUAL_BAR_SIZE)
        .encode(
            x=alt.X("Year:O", title="Calendar year", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "Return:Q", title="Net simple return", axis=alt.Axis(format=".1%")
            ),
            color=alt.condition(
                alt.datum.Return >= 0,
                alt.value(COLOR_GAIN_GREEN),
                alt.value(COLOR_LOSS_RED),
            ),
            tooltip=["Year:O", alt.Tooltip("Return:Q", format=".2%")],
        ),
        ANNUAL_RETURN_CHART_HEIGHT,
    )


def frontier_chart(frontier: pd.DataFrame) -> None:
    st.altair_chart(
        alt.Chart(frontier)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "estimated_volatility_annual:Q",
                axis=alt.Axis(format=".0%"),
                title="Estimated annual volatility",
            ),
            y=alt.Y(
                "estimated_return_annual:Q",
                axis=alt.Axis(format=".0%"),
                title="Estimated annual arithmetic return",
            ),
            color="method:N",
            order="point:Q",
        ),
        width="stretch",
        theme=None,
    )


def evt_threshold_charts(grid: pd.DataFrame) -> None:
    if grid.empty:
        return
    shape_col, mrl_col = st.columns(2)
    with shape_col:
        show(
            alt.Chart(grid)
            .mark_line(point=True, color=COLOR_EVT_BLUE)
            .encode(
                x=alt.X(
                    "threshold_percentile:Q",
                    title="Threshold percentile",
                    axis=alt.Axis(format=".0%"),
                ),
                y=alt.Y("shape:Q", title="GPD shape"),
            ),
            EVT_DIAGNOSTIC_CHART_HEIGHT,
        )
    with mrl_col:
        show(
            alt.Chart(grid)
            .mark_line(point=True, color=COLOR_VAR_AMBER)
            .encode(
                x=alt.X("threshold:Q", title="Loss threshold"),
                y=alt.Y("mean_residual_life:Q", title="Mean residual life"),
            ),
            EVT_DIAGNOSTIC_CHART_HEIGHT,
        )
