# This file contains functions to build and visualize a networkx graph of
# Hamming-1 connections between TCRs
from typing import Optional, Tuple

import networkx as nx  # type: ignore
import pandas as pd
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap  # type: ignore
from matplotlib.patches import Patch  # type: ignore
from networkx.classes.graph import Graph as NxGraph  # type: ignore


def build_and_plot_bioid_ham1_graph(
    df_tcrs: DataFrame,
    node_color_attribute: Optional[str] = None,  # type: ignore
    width: int = 10,
    height: int = 10,
    edge_width: float = 0.6,
    edge_alpha: float = 0.5,
    cmap: Colormap = plt.cm.jet,  # type: ignore
    node_size: int = 25,
    show_legend: bool = False,
    legend_bbox_to_anchor: Tuple[float, float] = (1.03, 1.0),
    legend_loc: str = "upper left",
    bioid_column: str = "bioIdentity",
    attribute_columns: Optional[str] = None,
    title: str = "Hamming-1 connection graph",
) -> Tuple[NxGraph, plt.Figure, plt.Axes]:
    """Builds and plots a networkx graph of hamming-1 connections between bioIdentities.

    Args:
        df_tcrs (DataFrame): DataFrame of TCRs
        node_color_attribute (str, optional): attribute for node color. Defaults to None.
        width (int, optional): plot width. Defaults to 10.
        height (int, optional): plot height. Defaults to 10.
        edge_width (float, optional): edge width. Defaults to 0.6.
        edge_alpha (float, optional): edge alpha. Defaults to 0.5.
        cmap (Colormap, optional): colormap. Defaults to plt.cm.jet.
        node_size (int, optional): node size. Defaults to 25.
        show_legend (bool, optional): Show legend? Defaults to False.
        legend_bbox_to_anchor (Tuple[float, float], optional): legend location. Defaults to (1.03, 1.0).
        legend_loc (str, optional): legend anchor. Defaults to 'upper left'.
        bioid_column (str, optional): bioIdentity column. Defaults to "bioIdentity".
        attribute_columns (List[str], optional): columns to use as node attributes. Defaults to None.
        title (str, optional): title. Defaults to "Hamming-1 connection graph".

    Returns:
        Tuple[NxGraph, plt.Figure, plt.Axes]: Graph, Figure and Axes
    """
    G = build_bioid_ham1_graph(
        df_tcrs, bioid_column=bioid_column, attribute_columns=attribute_columns
    )
    f, ax = plot_bioid_ham1_graph(
        G,
        node_color_attribute=node_color_attribute,
        width=width,
        height=height,
        edge_width=edge_width,
        edge_alpha=edge_alpha,
        cmap=cmap,
        node_size=node_size,
        show_legend=show_legend,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
        legend_loc=legend_loc,
        title=title,
    )
    return G, f, ax


def build_bioid_ham1_graph_and_extract_ccs(
    spark: SparkSession,
    pdf_tcrs: pd.DataFrame,
    min_bioids: int = 1,
    bioid_column: str = "bioIdentity",
) -> pd.DataFrame:
    """Builds a networkx graph of hamming-1 connections between bioIdentities and extracts connected
    components with at least `min_bioids` bioIdentities.

    Takes a Pandas DataFrame of TCRs and returns the same dataframe with connected component as a new
    column. Raises ValueError if pdf_tcrs has bioIdentity duplicates.

    Args:
        df_tcrs (DataFrame): DataFrame of TCRs
        min_bioids (int, optional): minimum bioids per component. Defaults to 1.
        bioid_column (str, optional): bioIdentity column. Defaults to "bioIdentity". If you want to use something
            else, like a preprocessed bioIdentity column, you can specify it here.

    Returns:
        pd.DataFrame: pdf_tcrs with connected_component column
    """
    if len(set(pdf_tcrs[bioid_column])) != len(pdf_tcrs):
        raise ValueError("pdf_tcrs has bioIdentity duplicates")
    df_tcrs = spark.createDataFrame(pdf_tcrs)
    G = build_bioid_ham1_graph(df_tcrs, bioid_column=bioid_column)
    pdf_ccs = extract_connected_components(G, min_bioids, bioid_column=bioid_column)
    pdf_tcrs_with_ccs = pdf_tcrs.merge(pdf_ccs, on=bioid_column)
    return pdf_tcrs_with_ccs


def build_bioid_ham1_graph_and_extract_ccs_spark(
    df_tcrs: DataFrame,
    min_bioids: int = 1,
    bioid_column: str = "bioIdentity",
    suffix_from_column: Optional[str] = None,
    column_suffix_separator: str = "@",
    plus_standin_char: str = "#",
) -> pd.DataFrame:
    """Builds a networkx graph of hamming-1 connections between bioIdentities and extracts connected
    components with at least `min_bioids` bioIdentities.

    Takes a Pandas DataFrame of TCRs and returns the same dataframe with connected component as a new
    column. Raises ValueError if pdf_tcrs has bioIdentity duplicates.

    Args:
        df_tcrs (DataFrame): DataFrame of TCRs
        min_bioids (int, optional): minimum bioids per component. Defaults to 1.
        bioid_column (str, optional): bioIdentity column. Defaults to "bioIdentity". If you want to use something
            else, like a preprocessed bioIdentity column, you can specify it here.
        suffix_from_column (str, optional): If specified, adds the specified column's value as a
            suffix to the bioIdentity (effectively breaking up components between those values).
        column_suffix_separator (str, optional): Separator to use when adding suffix. Defaults to "@".
        plus_standin_char (str, optional): Character to use as a stand-in for '+' in bioIdentities,
            since + is meaningful in bioidentities. Defaults to "#".

    Returns:
        pd.DataFrame: pdf_tcrs with connected_component column
    """
    if suffix_from_column is not None:
        # swap out plus signs in bioIdentity to avoid confusion when adding suffixes
        df_tcrs = df_tcrs.withColumnRenamed(
            suffix_from_column, "suffix_from_column_orig"
        ).withColumn(
            suffix_from_column,
            F.regexp_replace(F.col("suffix_from_column_orig"), r"\+", plus_standin_char),
        )
        # add suffix to bioIdentity, separated by column_suffix_separator
        df_tcrs = df_tcrs.withColumnRenamed(bioid_column, "bioIdentity_orig").withColumn(
            bioid_column,
            F.concat_ws(
                column_suffix_separator, F.col("bioIdentity_orig"), F.col(suffix_from_column)
            ),
        )
    G = build_bioid_ham1_graph(df_tcrs, bioid_column=bioid_column)
    pdf_ccs = extract_connected_components(G, min_bioids, bioid_column=bioid_column)

    if suffix_from_column is not None:
        # need to strip off suffixes
        pdf_ccs[suffix_from_column] = (
            pdf_ccs[bioid_column].str.split(column_suffix_separator).str[1]
        )
        # convert plus_standin_char back to +
        pdf_ccs[suffix_from_column] = pdf_ccs[suffix_from_column].str.replace(
            plus_standin_char, "+", regex=False
        )
        pdf_ccs[bioid_column] = pdf_ccs[bioid_column].str.split(column_suffix_separator).str[0]
        # put suffix in cc name
        pdf_ccs["connected_component"] = (
            pdf_ccs[suffix_from_column].astype(str)
            + "_"
            + pdf_ccs["connected_component"].astype(str)
        )
    return pdf_ccs


def build_bioid_ham1_graph(
    df_tcrs: DataFrame,  # type: ignore
    bioid_column: str = "bioIdentity",
    attribute_columns: Optional[str] = None,
) -> NxGraph:
    """Builds a networkx graph from a DataFrame of TCRs, where nodes are bioIdentities and edges are
    hamming-1 connections between bioIdentities.

    Args:
        df_tcrs (DataFrame): DataFrame of TCRs
        bioid_column (str, optional): bioIdentity column. Defaults to "bioIdentity
        attribute_columns (List[str], optional): columns to use as node attributes. Defaults to None.
            Regardless of what's specified here, vfamily and jfamily will always be added as node
            attributes, derived from the bioIdentity column. That's because they're so often useful,
            but it's probably a bad idea.
        suffix_from_column (str, optional): If specified, adds the specified column's value as a
            suffix to the bioIdentity (effectively breaking up components between those values).

    Returns:
        NxGraph: Networkx graph of hamming-1 connections between bioIdentities
    """
    df_tcrs_forham1 = df_tcrs
    if bioid_column != "bioIdentity":
        df_tcrs_forham1 = df_tcrs.drop("bioIdentity").withColumnRenamed(
            bioid_column, "bioIdentity"
        )

    ham1_comparison = InternalHamming1Comparison(df_tcrs_forham1)
    pdf_ham1pairs = ham1_comparison.df_comparison.toPandas()

    all_bioIdentities = set(pdf_ham1pairs["bioIdentity_i"]).union(
        set(pdf_ham1pairs["bioIdentity_j"])
    )

    G = nx.Graph()

    for bioIdentity in all_bioIdentities:
        _, vgene, jgene = bioIdentity.split("+")
        vfamily = vgene.split("-")[0]
        jfamily = jgene.split("-")[0]
        G.add_node(bioIdentity, vfamily=vfamily, jfamily=jfamily)

    # if there are other attributes to add, we have to get a little clever, because we haven't explicitly
    # verified that bioIdentity is unique. We'll just take the first value for each bioIdentity
    if attribute_columns is not None:
        # this had better be small enough to toPandas()
        pdf_tcrs = df_tcrs.toPandas()
        for bioIdentity in all_bioIdentities:
            for attr in attribute_columns:
                if attr in ["vfamily", "jfamily"]:
                    continue
                G.nodes[bioIdentity][attr] = pdf_tcrs.loc[
                    pdf_tcrs[bioid_column] == bioIdentity, attr
                ].iloc[0]

    for _, row in pdf_ham1pairs.iterrows():
        G.add_edge(row["bioIdentity_i"], row["bioIdentity_j"], weight=1)

    return G


def plot_bioid_ham1_graph(
    G: NxGraph,
    node_color_attribute: Optional[str] = None,  # type: ignore
    width: int = 10,
    height: int = 10,
    edge_width: float = 0.6,
    edge_alpha: float = 0.5,
    cmap: Colormap = plt.cm.jet,
    node_size: int = 25,
    show_legend: bool = False,  # type: ignore
    legend_bbox_to_anchor: Tuple[float, float] = (1.03, 1.0),
    legend_loc: str = "upper left",
    title: str = "Hamming-1 connection graph",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots a networkx graph of hamming-1 connections between bioIdentities.

    Args:
        G (NxGraph): graph
        node_color_attribute (str, optional): attribute for node color. Defaults to None.
        width (int, optional): plot width. Defaults to 10.
        height (int, optional): plot height. Defaults to 10.
        edge_width (float, optional): edge width. Defaults to 0.6.
        edge_alpha (float, optional): edge alpha. Defaults to 0.5.
        cmap (Colormap, optional): colormap. Defaults to plt.cm.jet.
        node_size (int, optional): node size. Defaults to 25.
        show_legend (bool, optional): Show legend? Defaults to False.
        legend_bbox_to_anchor (Tuple[float, float], optional): legend location. Defaults to (1.03, 1.0).
        legend_loc (str, optional): legend anchor. Defaults to 'upper left'.
        title (str, optional): title. Defaults to "Hamming-1 connection graph".
    """
    pos = nx.spring_layout(G)

    node_colors = "b"
    if node_color_attribute is not None:
        # Define the node colors based on the attribute
        node_color_vals = nx.get_node_attributes(G, node_color_attribute).values()
        ordered_unique_color_vals = sorted(list(set(node_color_vals)))
        colors_inorder = sns.color_palette(n_colors=len(ordered_unique_color_vals))
        val_color_map = dict(zip(*[ordered_unique_color_vals, colors_inorder]))
        node_colors = [val_color_map[x] for x in node_color_vals]  # type: ignore

    # Initialize the plot
    f = plt.figure(1, figsize=(width, height))
    ax = f.add_subplot(1, 1, 1)

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, cmap=cmap)
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha, width=edge_width)
    ax.set_title(title)

    # Draw the legend
    if show_legend:
        if node_color_attribute is None:
            legend_elements = [Patch(facecolor="b", label="Nodes")]
        else:
            legend_elements = [
                Patch(facecolor=colors_inorder[i], label=ordered_unique_color_vals[i])
                for i in range(len(colors_inorder))
            ]
        ax.legend(handles=legend_elements, bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc)  # type: ignore

    return f, ax


def extract_connected_components(
    G: NxGraph, min_bioids: int = 1, bioid_column: str = "bioIdentity"  # type: ignore
) -> pd.DataFrame:
    """Extracts connected components from a networkx graph into a Pandas dataframe.
    Component identifiers aren't meaningful, but are unique integers.

    Will fail if there are no components with at least `min_bioids` bioIdentities

    Args:
        G (NxGraph): graph
        n_bioids (int, optional): minimum bioids per component. Defaults to 1.
        bioIdentity_column (str, optional): bioIdentity column. Defaults to "bioIdentity"

    Returns:
        pd.DataFrame: Pandas dataframe with columns `bioIdentity` and `connected_component`
    """
    pdfs = []
    for idx, bioids in enumerate(nx.connected_components(G)):
        if len(bioids) >= min_bioids:
            pdf_one_cc = pd.DataFrame({bioid_column: list(bioids)})
            pdf_one_cc["connected_component"] = idx
            pdfs.append(pdf_one_cc)
    pdf_ccs = pd.concat(pdfs)
    return pdf_ccs


def prepend_other_column_to_bioid(
    pdf_tcrs: pd.DataFrame, other_column: str, sep: str = "|"
) -> pd.DataFrame:
    """Utility function to prepend another column to bioIdentity, in a new column, with
    a separator between. This is useful for e.g. forcing Hamming-1
    connections between bioIdentities to only be between members of the same cluster.

    Args:
        pdf_tcrs (pd.DataFrame): DataFrame of TCRs
        other_column (str): column to prepend to bioIdentity
        sep (str, optional): separator. Defaults to "|".
    """
    return pdf_tcrs[other_column] + sep + pdf_tcrs.bioIdentity  # type: ignore

