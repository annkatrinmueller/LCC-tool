import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from pathlib import Path

from plotly.subplots import make_subplots

TECHNOLOGY_COLORS = {
    "AWE": {
        "S1": "#4C78A8",
        "ST": "#9ECAE1",
    },
    "PEMWE": {
        "S1": "#E67E22",
        "ST": "#F5B97D",
    },
}


def plot_timeseries(
    df,
    y,
    *,
    x="DATE",
    title=None,
    x_title="Date",
    y_title=None,
):
    fig = px.line(
        df,
        x=x,
        y=y,
        markers=True,
        title=title,
    )

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
    )

    # fig.show()

    return fig


def plot_cost_breakdown(
    df,
    *,
    category="Material",
    value="Cost",
    title="Cost breakdown",
):
    plot_df = df.sort_values(
        value,
        ascending=True,
    )

    fig = px.bar(
        plot_df,
        x=value,
        y=category,
        orientation="h",
        title=title,
    )

    fig.update_layout(
        xaxis_title="EUR",
        yaxis_title=None,
    )

    return fig


def plot_distribution(
    values,
    *,
    title,
    x_title,
):
    fig = px.histogram(
        x=values,
        nbins=60,
        title=title,
        marginal="box",
    )

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Count",
    )

    # fig.show()

    return fig


def plot_sobol(
    results,
    *,
    category="Material",
    title="Sensitivity analysis",
):
    plot_df = results.sort_values("ST", ascending=True)

    fig = px.bar(
        plot_df,
        y=category,
        x=["S1", "ST"],
        orientation="h",
        barmode="group",
        title=title,
    )

    fig.update_layout(
        xaxis_title="Sobol index",
        yaxis_title=None,
        legend_title=None,
    )

    # fig.show()

    return fig


import plotly.graph_objects as go


def plot_forecast(
    history,
    forecast,
    *,
    history_col,
    mean_col,
    lower_col,
    upper_col,
    title,
    y_title,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["DATE"],
            y=history[history_col],
            mode="lines",
            name="History",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["DATE"],
            y=forecast[mean_col],
            mode="lines",
            name="Mean",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["DATE"],
            y=forecast[upper_col],
            mode="lines",
            name="Upper",
            line={"dash": "dot"},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["DATE"],
            y=forecast[lower_col],
            mode="lines",
            name="Lower",
            line={"dash": "dot"},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_title,
        hovermode="x unified",
    )

    return fig


def plot_sobol_results(
    problem,
    Si,
    *,
    title=None,
    technology="AWE",
    dominant_threshold=0.05,
    minor_axis_floor=0.0055,
    save_path=None,
):
    """
    Create Sobol sensitivity visualizations.

    Figures
    -------
    1. S1 and ST indices

    2. First-order versus total-order indices

    3. S1 and ST indices expressed as percentages

    """

    if technology not in TECHNOLOGY_COLORS:
        raise ValueError(
            f"Unknown technology '{technology}'. "
            f"Expected one of {list(TECHNOLOGY_COLORS)}."
        )

    colors = TECHNOLOGY_COLORS[technology]

    # =====================================================
    # Build results table
    # =====================================================

    results = pd.DataFrame({
        "Parameter": problem["names"],
        "S1": Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST": Si["ST"],
        "ST_conf": Si["ST_conf"],
    })

    results["Interaction_effect"] = (
        results["ST"] - results["S1"]
    )

    results["S1_percent"] = (
        results["S1"] * 100.0
    )

    results["ST_percent"] = (
        results["ST"] * 100.0
    )

    # =====================================================
    # Split dominant and minor sensitivities
    # =====================================================

    dominant_data = (
        results[
            results["ST"] >= dominant_threshold
        ]
        .sort_values("ST", ascending=True)
        .reset_index(drop=True)
    )

    minor_data = (
        results[
            results["ST"] < dominant_threshold
        ]
        .sort_values("ST", ascending=True)
        .reset_index(drop=True)
    )

    # Determine figure height from the number of parameters.
    dominant_height = max(
        200,
        38 * len(dominant_data),
    )

    minor_height = max(
        300,
        32 * len(minor_data),
    )

    total_height = (
        dominant_height
        + minor_height
        + 180
    )

    # Give the two panels approximately proportional space.
    total_rows = max(
        len(dominant_data)
        + len(minor_data),
        1,
    )

    dominant_ratio = max(
        len(dominant_data) / total_rows,
        0.25,
    )

    minor_ratio = max(
        len(minor_data) / total_rows,
        0.25,
    )

    ratio_sum = (
        dominant_ratio
        + minor_ratio
    )

    row_heights = [
        dominant_ratio / ratio_sum,
        minor_ratio / ratio_sum,
    ]

    # =====================================================
    # Shared layout
    # =====================================================

    shared_layout = {
        "title": title,
        "font": {
            "size": 16,
        },
        "margin": {
            "t": 25,
            "r": 20,
            "b": 70,
            "l": 35,
        },
        "legend_title": "Legend",
        "legend": {
            "orientation": "v",
            "x": 0,
            "xanchor": "center",
            "y": -0.08,
            "yanchor": "top",
            "font": {
                "size": 14,
            },
        },
    }

    # =====================================================
    # Figure 1: S1 and ST indices
    # =====================================================

    fig_indices = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.09,
        row_heights=row_heights,
    )

    def add_index_bars(
        fig,
        data,
        row,
        *,
        showlegend,
        percentage=False,
    ):
        """
        Add S1 and ST horizontal bars to one subplot row.
        """

        if percentage:
            s1_values = data["S1_percent"]
            st_values = data["ST_percent"]
            multiplier = 100.0
        else:
            s1_values = data["S1"]
            st_values = data["ST"]
            multiplier = 1.0

        fig.add_trace(
            go.Bar(
                y=data["Parameter"],
                x=s1_values,
                name="S1",
                orientation="h",
                marker_color=colors["S1"],
                error_x={
                    "type": "data",
                    "array":
                        data["S1_conf"]
                        * multiplier,
                    "visible": True,
                },
                customdata=data["S1_conf"],
                hovertemplate=(
                    "%{y}<br>"
                    "S1: %{customdata:.4f}"
                    "<extra></extra>"
                ),
                showlegend=showlegend,
            ),
            row=row,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                y=data["Parameter"],
                x=st_values,
                name="ST",
                orientation="h",
                marker_color=colors["ST"],
                error_x={
                    "type": "data",
                    "array":
                        data["ST_conf"]
                        * multiplier,
                    "visible": True,
                },
                customdata=data["ST_conf"],
                hovertemplate=(
                    "%{y}<br>"
                    "ST: %{customdata:.4f}"
                    "<extra></extra>"
                ),
                showlegend=showlegend,
            ),
            row=row,
            col=1,
        )

    add_index_bars(
        fig_indices,
        dominant_data,
        row=1,
        showlegend=True,
    )

    add_index_bars(
        fig_indices,
        minor_data,
        row=2,
        showlegend=False,
    )

    # Independent scales make both magnitude groups readable.
    dominant_max = (
        dominant_data["ST"]
        + dominant_data["ST_conf"]
    ).max()

    minor_max = (
        minor_data["ST"]
        + minor_data["ST_conf"]
    ).max()

    fig_indices.update_xaxes(
        range=[
            0.0,
            max(
                float(dominant_max) * 1.05,
                0.01,
            ),
        ],
        title_text="",
        row=1,
        col=1,
    )

    fig_indices.update_xaxes(
        range=[
            0.0,
            max(
                float(minor_max) * 1.45,
                minor_axis_floor,
            ),
        ],
        title_text="Sobol sensitivity index",
        row=2,
        col=1,
    )

    fig_indices.update_layout(
        **shared_layout,
        barmode="group",
        height=total_height,
    )

    # =====================================================
    # Figure 2: S1 versus ST
    # =====================================================

    max_value = float(
        np.nanmax([
            results["S1"].max(),
            results["ST"].max(),
        ])
    )

    max_value = max(
        max_value * 1.05,
        0.01,
    )

    fig_comparison = go.Figure()

    fig_comparison.add_trace(
        go.Scatter(
            x=results["S1"],
            y=results["ST"],
            mode="markers+text",
            text=results["Parameter"],
            textposition="top center",
            marker={
                "color": colors["ST"],
                "size": 10,
            },
            customdata=np.column_stack([
                results["S1_conf"],
                results["ST_conf"],
            ]),
            hovertemplate=(
                "%{text}<br>"
                "S1: %{x:.4f}<br>"
                "ST: %{y:.4f}<br>"
                "S1 confidence: ±%{customdata[0]:.4f}<br>"
                "ST confidence: ±%{customdata[1]:.4f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # S1 = ST means no contribution from interactions.
    fig_comparison.add_trace(
        go.Scatter(
            x=[0.0, max_value],
            y=[0.0, max_value],
            mode="lines",
            line={
                "dash": "dash",
                "color": colors["S1"],
            },
            name="S1 = ST",
        )
    )

    fig_comparison.update_layout(
        **shared_layout,
        height=650,
        xaxis_title="First-order index (S1)",
        yaxis_title="Total-order index (ST)",
    )

    fig_comparison.update_xaxes(
        range=[0.0, max_value],
    )

    fig_comparison.update_yaxes(
        range=[0.0, max_value],
        scaleanchor="x",
        scaleratio=1,
    )

    # =====================================================
    # Figure 3: Sobol indices in percent
    #
    # Use exactly the same dominant/minor structure as
    # Figure 1 so both views remain visually consistent.
    # =====================================================

    fig_percentage = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.09,
        row_heights=row_heights,
    )

    add_index_bars(
        fig_percentage,
        dominant_data,
        row=1,
        showlegend=True,
        percentage=True,
    )

    add_index_bars(
        fig_percentage,
        minor_data,
        row=2,
        showlegend=False,
        percentage=True,
    )

    fig_percentage.update_xaxes(
        range=[
            0.0,
            max(
                float(dominant_max)
                * 100.0
                * 1.05,
                1.0,
            ),
        ],
        title_text="",
        row=1,
        col=1,
    )

    fig_percentage.update_xaxes(
        range=[
            0.0,
            max(
                float(minor_max)
                * 100.0
                * 1.45,
                minor_axis_floor * 100.0,
            ),
        ],
        title_text="Sobol sensitivity index [%]",
        row=2,
        col=1,
    )

    fig_percentage.update_layout(
        **shared_layout,
        barmode="group",
        height=total_height,
    )

    # =====================================================
    # Collect figures
    # =====================================================

    figures = {
        "indices": fig_indices,
        "comparison": fig_comparison,
        "percentage": fig_percentage,
    }

    # =====================================================
    # Optional SVG export
    # =====================================================

    if save_path is not None:
        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        for name, fig in figures.items():
            output_path = (
                save_path.parent
                / f"{save_path.name}_{name}.svg"
            )

            fig.write_image(
                output_path,
                format="svg",
            )

    return results, figures


def plot_sobol_interaction_matrix(
    problem,
    Si,
    *,
    title="Second-order Sobol interactions",
    decimals=4,
):
    """
    Plot pairwise second-order Sobol indices as a symmetric
    interaction matrix.

    S2 confidence values are included in the hover information.
    """

    names = problem["names"]

    s2 = np.asarray(
        Si["S2"],
        dtype=float,
    ).copy()

    s2_conf = np.asarray(
        Si["S2_conf"],
        dtype=float,
    ).copy()

    n = len(names)

    # SALib normally populates the upper triangle only.
    # Mirror S2 and its confidence values for visualization.
    for i in range(n):
        for j in range(i + 1, n):
            s2[j, i] = s2[i, j]
            s2_conf[j, i] = s2_conf[i, j]

    # No parameter interacts with itself.
    np.fill_diagonal(s2, np.nan)
    np.fill_diagonal(s2_conf, np.nan)

    # Use a symmetric scale around zero.
    # Small negative Sobol estimates can occur because of
    # numerical sampling uncertainty.
    max_abs = np.nanmax(np.abs(s2))

    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    text = np.empty(
        s2.shape,
        dtype=object,
    )

    for i in range(n):
        for j in range(n):
            if np.isfinite(s2[i, j]):
                text[i, j] = f"{s2[i, j]:.{decimals}f}"
            else:
                text[i, j] = ""

    fig = go.Figure(
        data=go.Heatmap(
            z=s2,
            x=names,
            y=names,
            zmin=-max_abs,
            zmax=max_abs,
            zmid=0,
            customdata=s2_conf,
            text=text,
            texttemplate="%{text}",
            hovertemplate=(
                "%{y} × %{x}<br>"
                "S2: %{z:.6f}<br>"
                "Confidence: ±%{customdata:.6f}"
                "<extra></extra>"
            ),
            colorbar={
                "title": "S2",
            },
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=None,
        height=750,
        width=850,
    )

    fig.update_xaxes(
        tickangle=-45,
    )

    return fig


def plot_sobol_interactions(
    problem,
    Si,
    *,
    top_n=12,
    title="Largest second-order Sobol interactions",
):
    """
    Plot the largest pairwise second-order Sobol indices
    with confidence intervals.
    """

    names = problem["names"]

    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            s2 = Si["S2"][i, j]
            s2_conf = Si["S2_conf"][i, j]

            if not np.isfinite(s2):
                continue

            rows.append(
                {
                    "Pair": (f"{names[i]} × {names[j]}"),
                    "S2": float(s2),
                    "S2_conf": float(s2_conf),
                }
            )

    plot_df = (
        pd.DataFrame(rows)
        .sort_values(
            "S2",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            "S2",
            ascending=True,
        )
    )

    fig = go.Figure(
        go.Bar(
            y=plot_df["Pair"],
            x=plot_df["S2"],
            orientation="h",
            error_x={
                "type": "data",
                "array": plot_df["S2_conf"],
                "visible": True,
            },
            customdata=plot_df["S2_conf"],
            hovertemplate=(
                "%{y}<br>"
                "S2: %{x:.6f}<br>"
                "Confidence: ±%{customdata:.6f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Second-order Sobol index (S2)",
        yaxis_title=None,
    )

    fig.add_vline(
        x=0,
        line_width=1,
        line_dash="dot",
    )

    return fig
