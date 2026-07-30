"""
This module contains some helper functions to render stuff that we don't really need in the notebook.
"""

import networkx as nx
from IPython.display import HTML, display
from causal_testing.testing.causal_test_case import CausalTestCase
import pandas as pd
import holoviews as hv

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
        kdims=["estimator.treatment_variable", "estimator.outcome_variable"],
        vdims=["result.adequacy.kurtosis"],
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
        adequacy,
        kdims=["estimator.treatment_variable", "estimator.outcome_variable"],
        vdims=["result.adequacy.passing"],
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
