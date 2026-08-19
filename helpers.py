"""
This module contains some helper functions to render stuff that we don't really need in the notebook.
"""

import networkx as nx
from IPython.display import HTML, display
from causal_testing.testing.causal_test_case import CausalTestCase
import numpy as np
import pandas as pd
import holoviews as hv
from bokeh.models import Arrow, NormalHead, Ellipse, HoverTool
from scipy.interpolate import splev, make_splprep
from causal_testing.visualisation.causal_test_result_visualiser import results_dag
from causal_testing.specification.causal_dag import CausalDAG
import re

hv.extension("bokeh")


def render_dag(dag: nx.DiGraph, size: str = None):
    """
    Show a causal DAG as part of the output of a cell.

    :param dag: The dag to display.
    :param size: The size it should be in inches, e.g. "10,7" for a 10 by 7 inch plot.
    """
    rendered = nx.nx_agraph.to_agraph(dag)
    if size is not None:
        rendered.graph_attr.update(size=size)

    rendered.graph_attr["rankdir"] = "LR"
    rendered.layout(prog="dot")

    # Get raw SVG string
    svg_data = rendered.draw(format="svg").decode("utf-8")

    # Inject CSS to make edge lines easier to hover over
    hover_css = """
   <style>
       .edge path {
           stroke-width: 3px !important;  /* Make line slightly thicker */
           cursor: pointer;
       }
       .edge:hover path {
           stroke-width: 5px !important;
       }
   </style>
   """

    display(HTML(hover_css + svg_data))


def _get_split_category_order(df: pd.DataFrame, col: str, value_col: str) -> list:
    """
    Partitions categories into two groups relative to overall_median:
    - Group median < overall_median: sorted by category min (ascending).
    - Group median >= overall_median: sorted by category max (ascending).
    """
    stats = df.groupby(col)[value_col].agg(["median", "min", "max"]).reset_index()

    # Sort lower half by min value, upper half by max value
    lower_order = stats[stats["median"] < stats["median"].median()].sort_values(by="min", ascending=True)[col].tolist()
    upper_order = stats[stats["median"] >= stats["median"].median()].sort_values(by="max", ascending=True)[col].tolist()

    return lower_order + upper_order


def sort_df_by_median_split(
    df: pd.DataFrame,
    treatment_col: str = "estimator.treatment_variable",
    outcome_col: str = "estimator.outcome_variable",
    value_col: str = "result.adequacy.kurtosis",
) -> pd.DataFrame:
    """
    Sorts treatment and outcome variables relative to the overall median kurtosis.
    """

    # Fill missing (treatment, outcome) combinations with empty rows
    # We need this to ensure that it's possible to obtain the correct ordering in the heatmap
    df = (
        df.set_index([treatment_col, outcome_col])
        .reindex(
            pd.MultiIndex.from_product(
                [
                    df[treatment_col].dropna().unique(),
                    df[outcome_col].dropna().unique(),
                ],
                names=[treatment_col, outcome_col],
            )
        )
        .reset_index()
    )

    # Apply ordered categoricals so HoloViews maps the axes to these index positions
    df_sorted = df[[treatment_col, outcome_col, value_col]].copy()
    df_sorted[treatment_col] = pd.Categorical(
        df[treatment_col], categories=_get_split_category_order(df, treatment_col, value_col), ordered=True
    )
    df_sorted[outcome_col] = pd.Categorical(
        df[outcome_col], categories=_get_split_category_order(df, outcome_col, value_col), ordered=True
    )

    df_sorted = df_sorted.sort_values(by=[treatment_col, outcome_col]).dropna()

    # Need to convert the values back to strings, otherwise holoviz thinks they're not unique
    df_sorted[treatment_col] = df_sorted[treatment_col].astype(str)
    df_sorted[outcome_col] = df_sorted[outcome_col].astype(str)
    return df_sorted


def data_adequacy_heatmap(test_cases: list[CausalTestCase]) -> hv.HeatMap:
    """
    Visualise data adequacy as an adjacency matrix heatmap of the kurtosis.

    :param test_cases: List of executed test cases to process.
    """
    adequacy = pd.json_normalize(map(lambda t: t.to_dict(), test_cases))

    for col in ["effect_estimate", "ci_low", "ci_high", "adequacy.kurtosis"]:
        columns = [c for c in adequacy.columns if c.startswith(f"result.{col}.")]
        adequacy[f"result.{col}"] = adequacy[columns].bfill(axis=1).iloc[:, 0]
        adequacy = adequacy.drop(columns=columns)
    adequacy = sort_df_by_median_split(adequacy)

    # Get data bounds
    vmin = adequacy["result.adequacy.kurtosis"].min()
    vmax = adequacy["result.adequacy.kurtosis"].max()

    # Calculate zero position (0.0 to 1.0)
    zero_ratio = (0 - vmin) / (vmax - vmin)

    # Generate the colour samples from the negative and positive colourmaps
    num_samples = 1000
    n_neg = int(num_samples * zero_ratio)
    n_pos = num_samples - n_neg

    neg_colors = hv.plotting.util.process_cmap("blues_r", provider="bokeh", ncolors=n_neg)
    pos_colors = hv.plotting.util.process_cmap("YlOrRd", provider="bokeh", ncolors=n_pos)
    asymmetric_cmap = neg_colors + pos_colors

    # Render
    return hv.HeatMap(
        adequacy,
        kdims=[
            ("estimator.treatment_variable", "Treatment variable"),
            ("estimator.outcome_variable", "Outcome variable"),
        ],
        vdims=[("result.adequacy.kurtosis", "Kurtosis")],
    ).opts(
        cmap=asymmetric_cmap,
        clim=(vmin, vmax),
        clipping_colors={"NaN": "grey"},  # Grey out invalid tests
        colorbar=True,
        xrotation=90,
        width=600,
        height=500,
        tools=["hover"],
        xlabel="Treatment variable",
        ylabel="Outcome variable",
        clabel="Causal test adequacy",
    )


def dag_adequacy_heatmap(test_cases: list[CausalTestCase]) -> hv.HeatMap:
    """
    Visualise dag adequacy as an adjacency matrix heatmap of the percentage of passing test cases.

    :param test_cases: List of executed test cases to process.
    """
    adequacy = pd.json_normalize(map(lambda t: t.to_dict(), test_cases))

    # Turn passing test cases into a percentage
    adequacy["result.adequacy.passing"] = (
        adequacy["result.adequacy.passing"] / adequacy["result.adequacy.bootstrap_size"]
    ) * 100

    return hv.HeatMap(
        sort_df_by_median_split(adequacy, value_col="result.adequacy.passing"),
        kdims=[
            ("estimator.treatment_variable", "Treatment variable"),
            ("estimator.outcome_variable", "Outcome variable"),
        ],
        vdims=[("result.adequacy.passing", "Passing (%)")],
    ).opts(
        cmap="RdYlGn",
        clim=(0, 100),
        clipping_colors={"NaN": "grey"},  # Grey out invalid tests
        colorbar=True,
        xrotation=90,
        width=600,
        height=500,
        tools=["hover"],
        xlabel="Treatment variable",
        ylabel="Outcome variable",
        clabel="Percentage passing test cases",
    )


def parse_dot_spline(pos_str: str) -> list[tuple[float, float]]:
    """
    Parse Graphviz 'pos' string into control points.
    See https://graphviz.org/docs/attr-types/splineType for syntax details.
    NOTE: This will ignore segments separated by ";", but this shouldn't be a problem in our limited context.

    :param pos_str: The graphviz position string representing the list of control points.
    """
    end_point = None
    points = []

    for point_type, x, y in re.findall(r"(?:(s|e),)?(\d+(?:.\d+)?),(\d+(?:\.\d+)?)", pos_str):
        x, y = float(x), float(y)
        if point_type == "s":
            points = [(x, y)] + points
        elif point_type == "e":
            end_point = x, y
        else:
            points.append((x, y))

    if end_point:
        points.append(end_point)

    # Remove consecutive duplicate points
    points = np.array(points)
    mask = np.ones(len(points), dtype=bool)
    mask[1:] = np.any(np.diff(points, axis=0) != 0, axis=1)
    return points[mask]


def edge_spline(
    control_points, target_node_centre, target_node_width, target_node_height=16, num_points=100, shorten=0.05
):
    """
    Generate an edge spline from the given control points, trimmed at source & target ellipse boundaries.

    :param control_points: The control points of the spline.
    :param target_node_centre: The coordinates of the centre of the target node.
    :param target_node_width: The width of the target_node.
    :param target_node_height: The height of the target_node (defaults to 16).
    :param num_points: The number of spline points to generage (defaults to 100).
    :param shorten: The percentage of the line to shorten by to allow for the arrow head (defaults to 5%).
    """
    b_spline, _ = make_splprep(control_points.T, k=3, s=0)

    # Evaluate the BSpline object at uniform parametric points
    smooth_points = b_spline(np.linspace(0, 1, num_points)).T

    # Calculate angle and boundary radius for end node
    dx_e = target_node_centre[0] - smooth_points[-round(num_points * shorten)][0]
    dy_e = target_node_centre[1] - smooth_points[-round(num_points * shorten)][1]
    angle_end = np.arctan2(dy_e, dx_e)
    r_end = (target_node_width * target_node_height) / np.sqrt(
        (target_node_width * np.sin(angle_end)) ** 2 + (target_node_height * np.cos(angle_end)) ** 2
    )

    dists_end = np.hypot(smooth_points[:, 0] - target_node_centre[0], smooth_points[:, 1] - target_node_centre[1])
    end_idx = len(smooth_points) - np.searchsorted(dists_end[::-1], r_end)

    trimmed_path = smooth_points[:end_idx]
    return trimmed_path


def node_width(label: str, text_font_size: int = 9, padding: int = 24) -> float:
    """
    Calculate the width that a node should be to accomadate the label.

    :param label: The node label.
    :param text_font_size: The font size in pt.
    :param padding: Node inner padding in pt.
    """
    return len(label) * text_font_size + padding


def style_graph_hook(plot: hv.plotting.bokeh.graphs.GraphPlot, element: hv.Graph):
    """
    Hook to properly style nodes to be an ellipse of the correct size.

    :param plot: The current plot figure.
    :param element: The Graph element.
    """
    fig = plot.handles["plot"]
    graph_renderer = plot.handles["glyph_renderer"]

    # Supply widths and heights to the node source
    node_source = graph_renderer.node_renderer.data_source
    node_source.data["width"] = element.nodes.data["index"].apply(node_width)
    node_source.data["height"] = [32] * len(element.nodes.data)

    # Define primary Ellipse glyph
    graph_renderer.node_renderer.glyph = Ellipse(
        x="x",
        y="y",
        width="width",
        height="height",
        fill_color="white",
        line_color="gray",
    )

    # Define hover / inspection Ellipse glyph (prevents reverting to green circles)
    graph_renderer.node_renderer.hover_glyph = Ellipse(
        x="x",
        y="y",
        width="width",
        height="height",
        fill_color="skyblue",
        line_color="gray",
    )

    # Add Arrowheads with matching edge colors
    for _, row in element.data.iterrows():
        color = row["color"]
        arrow = Arrow(
            end=NormalHead(fill_color=color, line_color=color, size=8),
            x_start=row["arrow_starts_x"],
            y_start=row["arrow_starts_y"],
            x_end=row["arrow_ends_x"],
            y_end=row["arrow_ends_y"],
            line_alpha=0,
        )
        fig.add_layout(arrow)


def interactive_results_dag(dag: CausalDAG, test_cases: list[CausalTestCase]) -> hv.Overlay:
    """
    Generate an interactive holoview graph of the causal DAG showing failing tests.

    :param dag: The original causal dag.
    :param test_cases: Executed causal test cases with a defined `result` object.
    :returns: Inveractive holoviews graph.
    """
    results = results_dag(dag=dag, test_cases=test_cases)
    for test in test_cases:
        effect_estimate = pd.concat(
            [
                test.result.effect_estimate.ci_low,
                test.result.effect_estimate.value,
                test.result.effect_estimate.ci_high,
            ],
            axis=1,
        )
        effect_estimate.columns = ["ci_low", "estimate", "ci_high"]
        try:
            results[test.treatment_variable][test.outcome_variable]["title"] = effect_estimate.to_html()
        except KeyError:
            continue

    # Use DOT to do the layout
    agraph = nx.nx_agraph.to_agraph(results)
    agraph.layout(prog="dot")

    node_positions = {}
    for node in agraph.nodes():
        x, y = map(float, node.attr["pos"].split(","))
        node_positions[node.name] = (x, y)

    # Build the edges
    edges_df = pd.DataFrame([{"source": u, "target": v} | data for u, v, data in results.edges(data=True)])

    edges_df["trimmed_path"] = edges_df[["source", "target"]].apply(
        lambda row: edge_spline(
            control_points=parse_dot_spline(agraph.get_edge(row["source"], row["target"]).attr["pos"]),
            target_node_centre=node_positions[row["target"]],
            target_node_width=node_width(row["target"]) / 2,
        ),
        axis=1,
    )
    edges_df[["arrow_starts_x", "arrow_starts_y"]] = pd.DataFrame(
        [trimmed_path[-2] for trimmed_path in edges_df["trimmed_path"]], index=edges_df.index
    )
    edges_df[["arrow_ends_x", "arrow_ends_y"]] = pd.DataFrame(
        [trimmed_path[-1] for trimmed_path in edges_df["trimmed_path"]], index=edges_df.index
    )

    # Build the graph from the nodes and edges
    graph = hv.Graph(
        (
            edges_df,
            hv.Nodes(
                [(x, y, node_id) for node_id, (x, y) in node_positions.items()],
                kdims=["x", "y", "index"],
            ),
            hv.EdgePaths(edges_df["trimmed_path"].tolist()),
        ),
        kdims=["source", "target"],
        vdims=[c for c in edges_df.columns if c not in ["source", "target", "trimmed_path"]],
    ).opts(
        edge_line_dash="style",
        edge_line_width=1.5,
        edge_color="color",
        edge_hover_line_color="color",
        width=900,
        height=450,
        hooks=[style_graph_hook],
        xaxis=None,
        yaxis=None,
        tools=[
            HoverTool(
                tooltips="""
            <div style="padding: 6px; border: 1px solid #ccc; font-family: sans-serif;">
                <strong>Treatment:</strong> @source<br>
                <strong>Outcome:</strong> @target<br>
                <strong>Causal Effect:</strong> <br/> @title{safe}<br>
            </div>
        """
            )
        ],
        inspection_policy="edges",
    )

    # Label layers
    node_labels = hv.Labels(
        [(x, y, node_id) for node_id, (x, y) in node_positions.items()], kdims=["x", "y"], vdims=["label"]
    ).opts(
        text_font_size="9pt",
        text_color="black",
        text_align="center",
        text_baseline="middle",
        yoffset=0,
    )

    edge_labels = hv.Labels(
        pd.concat(
            [
                pd.DataFrame(
                    # Stack the x and y elements of the middle index of each trimmed path
                    np.vstack(edges_df["trimmed_path"].apply(lambda path: path[len(path) // 2]).values),
                    columns=["x", "y"],
                ),
                edges_df["label"],
            ],
            axis=1,
        ),
        kdims=["x", "y"],
        vdims=["label"],
    ).opts(
        text_font_size="9pt",
        text_color="darkblue",
        text_align="center",
        text_baseline="middle",
    )

    return graph * node_labels * edge_labels
