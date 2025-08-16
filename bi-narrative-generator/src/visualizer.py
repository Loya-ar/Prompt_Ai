# src/visualizer.py
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------- Global theme (high-contrast & accessible) ---------
ACCESSIBLE_COLORWAY = [
    "#1b6ca8",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

def _style_fig(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=height,
        margin=dict(l=10, r=10, t=60 if title else 30, b=10),
        colorway=ACCESSIBLE_COLORWAY,
        font=dict(family="Inter, Segoe UI, system-ui, -apple-system, Arial",
                  size=14, color="#111827"),   # dark ink
        title=dict(text=title or "", x=0.01, xanchor="left",
                   font=dict(size=20, color="#111827", family="Inter, Segoe UI")),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#e5e7eb", borderwidth=1,
                    font=dict(size=12, color="#111827"))
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False,
                     linecolor="#9ca3af", ticks="outside", tickcolor="#9ca3af")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False,
                     linecolor="#9ca3af", ticks="outside", tickcolor="#9ca3af")
    fig.update_traces(textfont_color="#111827")
    return fig


# --------- Chart helpers you can call from main.py ---------

def line_trend(df: pd.DataFrame, y: str, x: str | None = None, color: str | None = None,
               title: str | None = None) -> go.Figure:
    data = df.copy()
    if x is None:
        data = data.reset_index().rename(columns={"index": "index"})
        x = "index"
    fig = px.line(data, x=x, y=y, color=color, markers=True)
    return _style_fig(fig, title or f"Trend — {y} over {x}")

def bar_top_categories(df: pd.DataFrame, category: str, value: str, top_n: int = 10,
                       title: str | None = None) -> go.Figure:
    agg = df.groupby(category, dropna=False)[value].sum().nlargest(top_n).reset_index()
    fig = px.bar(agg, x=category, y=value, text_auto=True)
    fig.update_traces(texttemplate="%{text:.2s}", textposition="outside", cliponaxis=False)
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")
    return _style_fig(fig, title or f"Top {top_n} {category} by {value}")

def scatter_xy(df: pd.DataFrame, x: str, y: str, color: str | None = None,
               trendline: bool = True, title: str | None = None) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, color=color,
                     trendline="ols" if trendline else None,
                     trendline_color_override="#111827")
    return _style_fig(fig, title or f"Scatter — {y} vs {x}")

def histogram(df: pd.DataFrame, column: str, bins: int = 30, color: str | None = None,
              title: str | None = None) -> go.Figure:
    fig = px.histogram(df, x=column, color=color, nbins=bins, barmode="overlay", opacity=0.85)
    return _style_fig(fig, title or f"Distribution — {column}")

def boxplot(df: pd.DataFrame, y: str, group: str | None = None,
            title: str | None = None) -> go.Figure:
    fig = px.box(df, x=group, y=y, points="suspectedoutliers")
    return _style_fig(fig, title or (f"Box Plot — {y}" if not group else f"Box Plot — {y} by {group}"))

def heatmap_corr(df: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr(numeric_only=True) if len(num_cols) >= 2 else pd.DataFrame()
    if corr.empty:
        # graceful empty fig
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough numeric columns for correlation heatmap",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#1f2937")
        )
        return _style_fig(fig, title)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Blues")
    # Force annotation text to be dark
    for ann in fig.layout.annotations or []:
        ann.font = dict(color="#111827", size=12)
    return _style_fig(fig, title, height=520)
