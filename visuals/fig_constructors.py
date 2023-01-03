# -*- coding: utf-8 -*-
"""
functions used to help the construction of figures,
using plotly and matplotlib 

"""
__author__ = ["Samuel Bazaz"]
__credits__ = ["Samuel Bazaz"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Samuel Bazaz"]

##############
#  Packages  #
##############

import os
import sys

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from plotly.subplots import make_subplots

#################
#  Importation  #
#################

path = os.path.dirname(os.getcwd())
sys.path.append(path)

###########################
#  Mosaique Constructors  #
###########################


def add_trace_mos(subplot: go.Figure, go_fig, i: int, j: int):
    """add a figure to a subplot at the right position
    
    Parameters
    ----------
    subplot : plotly figure,
        subplot
    go_fig : plotly graph object,
        to add
    i : int,
         row-1
    j : int,
        col-1
    no Return
    ---------
    """
    subplot.add_trace(
        go_fig, row=i + 1, col=j + 1,
    )


def add_traces_mos(subplot: go.Figure, lst_go_fig: list, i: int, j: int):
    """add figures to a subplot at the right position
    
    Parameters
    ----------
    subplot : plotly figure,
        subplot
    lst_go_fig : plotly graph object,
        to add
    i : int,
         row-1
    j : int,
        col-1
    no Return
    ---------
    """
    for go_fig in lst_go_fig:
        add_trace_mos(subplot, go_fig, i, j)


def init_subplot(
    nvar: int, ncol: int, varnames: list, vspace: float = 0.03, hspace: float = 0.03
) -> ():
    """Create a plotly subplot in the right format for a simple mosaique

    Parameters
    ----------
    nvar : int,
        number of variables
    ncol : int,
        number of columns
    varnames : str list, 
        names of the variables
    vspace : float,
        verticale space between plots
    hspace : float,
        horizontal space between plots
    Returns
    -------
    ... : plotly figure
    """
    return make_subplots(
        rows=nvar // ncol + 1,
        cols=ncol,
        subplot_titles=varnames,
        x_title="time",
        y_title="value",
        vertical_spacing=vspace,
        horizontal_spacing=hspace,
    )


def fill_subplot(nvar: int, ncol: int, subplot: go.Figure, add_plot, nsim: int = 1):
    """Fill a plotly subplot based on a certain function add_plot

    Parameters
    ----------
    nvar : int,
        number of variables
    ncol : int,
        number of columns
    subplot : plotly figure,
        subplot
    add_plot : function,
        add a certain plot to a subplot
    nsim : int,
        number of curves per plot in the mosaique
        
    no Return
    ---------
    """

    for i in range(nvar // ncol + 1):
        for j in range(ncol):
            if i * ncol + j < nvar:
                for isim in range(nsim):
                    add_plot(subplot, i, j, isim)


def construct_mosaique_1sim(
    output, ncol: int, title: str, add_plot, width: int = 1600, height: int = 2000
) -> go.Figure:
    """Create a mosaique based on a certain function add_plot from 1 simulation

    Parameters
    ----------
    output : pd.Dataframe,
        output a the simulation
    ncol : int,
        number of columns
    title : str,
        title of the mosaique
    add_plot : function,
        add a certain plot to a subplot
    width:int,
        width of the figure
    height:int,
        height of the figure
        
    Returns
    -------
    subplot : plotly figure
    """
    nvar = len(output.columns)
    subplot = init_subplot(nvar, ncol, output.columns)
    fill_subplot(nvar, ncol, subplot, add_plot)

    subplot.update_layout(
        title_text=title, width=width, height=height, template="plotly_white"
    )
    subplot.update_annotations(font_size=12)
    # subplot.show()
    return subplot


def construct_mosaique_nsim(
    nvar: int,
    ncol: int,
    title: str,
    varnames: list[str],
    add_plot,
    nsim: int,
    width: int = 1600,
    height: int = 2000,
    hspace: float = 0.04,
    vspace: float = 0.03,
) -> go.Figure:
    """Create a mosaique based on a certain function add_plot from nsim simulations

    Parameters
    ----------
    nvar : int,
        number of variables
    ncol : int,
        number of columns
    title : str,
        title of the mosaique
    varnames : str list,
        names of the variables
    add_plot : function,
        add a certain plot to a subplot
    nsim :int,
        number of simulations
    width:int,
        width of the figure
    height:int,
        height of the figure
    hspace : float,
        horizontal space between plots
    vspace : float,
        verticale space between plots

    
    Returns
    -------
    subplot : plotly figure
    """
    # nvar = len(outputs[0])
    subplot = init_subplot(nvar, ncol, varnames, vspace, hspace)
    fill_subplot(nvar, ncol, subplot, add_plot, nsim)

    subplot.update_layout(
        title_text=title, width=width, height=height, template="plotly_white"
    )  # showlegend=False
    subplot.update_annotations(font_size=12)
    # subplot.show()
    subplot.update_annotations(font_size=14)
    subplot.update_yaxes(
        tickangle=0,
        tickfont={"size": 8, "color": "#808080"},
        nticks=7,
        ## to have log format:
        # tickformat=".1e",
        # tickformat ="3f.",
        # title_standoff = 25,
    )
    subplot.update_xaxes(
        tickangle=0,
        tickfont={"size": 8, "color": "#808080"},
        title_standoff=25,
        nticks=7,
    )
    return subplot


##################################
#  Adapted mosaique constructors #
##################################


def add_trace_adaptmos(
    subplot: go.Figure,
    varname: str,
    go_fig,
    i: int,
    j: int,
    yaxis_to_right: str = "unemployment_rate",
):
    """Add a trace to a subplot 
    at the specified position with the correct y axis.

    Parameters
    ----------
    subplot : plotly.graph_objs.Figure
        The subplot to add the trace to.
    varname : str
        The name of the variable.
    go_fig : plotly.graph_objs.graph_objs
        The graph object to add.
    i : int
        The row index (0-indexed) of the position to add the trace to.
    j : int
        The column index (0-indexed) of the position to add the trace to.
    yaxis_to_right : str
        The name of the variable that should be plotted on the right y axis.
    
    Returns
    -------
    None
    """
    subplot.add_trace(
        go_fig, row=i + 1, col=j + 1, secondary_y=(varname in yaxis_to_right),
    )


def add_traces_adaptmos(
    subplot: go.Figure,
    varname: str,
    lst_go_fig: list,
    i: int,
    j: int,
    yaxis_to_right: str = "unemployment_rate",
):
    """Add a traces to a subplot 
    at the specified position with the correct y axis.

    Parameters
    ----------
    subplot : plotly.graph_objs.Figure
        The subplot to add the trace to.
    varname : str
        The name of the variable.
    go_fig : plotly.graph_objs.graph_objs
        The graph object to add.
    i : int
        The row index (0-indexed) of the position to add the trace to.
    j : int
        The column index (0-indexed) of the position to add the trace to.
    yaxis_to_right : str
        The name of the variable that should be plotted on the right y axis.
    
    Returns
    -------
    None
    """
    for go_fig in lst_go_fig:
        add_trace_adaptmos(
            subplot, varname, go_fig, i, j, yaxis_to_right=yaxis_to_right
        )


def init_subplot_adapted(
    ng: int, groupnames: list, ncol: int, vspace: float = 0.09, hspace: float = 0.03
) -> go.Figure:
    """Create a plotly subplot in the right format for an adapted mosaique

    Parameters
    ----------
    ng : int,
        number of groups
    groupnames : str list,
        names of the groups
    ncol : int,
        number of columns
    vspace : float,
        verticale space between plots
    hspace : float,
        horizontal space between plots
        
    Returns
    -------
    ... : plotly figure
    """
    nrow = ng // ncol + 1
    return make_subplots(
        rows=nrow,
        cols=ncol,
        specs=[[{"secondary_y": True} for j in range(ncol)] for i in range(nrow)],
        x_title="time",
        y_title="value",
        subplot_titles=groupnames,
        vertical_spacing=vspace,
        horizontal_spacing=hspace,
    )


def construct_adaptmos(
    ncol: int,
    title: str,
    groupnames: list[str],
    groups: list[list[str]],
    add_plot,
    width: int = 1700,
    height: int = 2000,
    hspace: float = 0.09,
    vspace: float = 0.03,
) -> go.Figure:
    """
    Create an adapted mosaique based on a certain function add_plot.

    Parameters
    ----------
    ncol : int
        Number of columns.
    title : str
        Title of the mosaique.
    groupnames : list[str]
        Names of the groups.
    groups: list[list[str]]
        List of the groups of variables.
    add_plot : function
        Function to add a certain plot to a subplot.
    width: int, optional
        Width of the figure, default is 1700.
    height: int, optional
        Height of the figure, default is 2000.
    vspace : float, optional
        Vertical space between plots, default is 0.03.
    hspace : float, optional
        Horizontal space between plots, default is 0.09.

    Returns
    -------
    go.Figure
        Subplot.
    """
    ng = len(groups)
    subplot = init_subplot_adapted(ng, groupnames, ncol, vspace, hspace)

    fill_subplot(ng, ncol, subplot, add_plot)

    subplot.update_layout(
        title_text=title, width=width, height=height, template="plotly_white"
    )  # showlegend=False
    subplot.update_annotations(font_size=14)
    subplot.update_yaxes(
        tickangle=0,
        tickfont={"size": 8, "color": "#808080"},
        nticks=7,
        ## to have log format:
        # tickformat=".1e",
        # tickformat ="3f.",
        # title_standoff = 25,
    )
    subplot.update_xaxes(
        tickangle=0,
        tickfont={"size": 8, "color": "#808080"},
        title_standoff=25,
        nticks=7,
    )
    return subplot


#############################
#  Network fig constructors #
#############################


def sub_net(net: nx.Graph, threshold: float, nodes: list) -> nx.Graph:
    """
    Create a subgraph from a network by keeping edges with weight
    above a certain threshold and filtering out self-loops.
    
    Parameters
    ----------
    net : nx.Graph
        The network to create the subgraph from.
    threshold : float
        The minimum weight an edge must have to be included in the subgraph.
    nodes : list
        A list of nodes to keep in the subgraph. If empty, all nodes in the subnet are kept.
    
    Returns
    -------
    subnet : nx.Graph
    The subgraph of the input network.
    """
    eligible_edges = [
        (from_node, to_node, edge_attributes)
        for from_node, to_node, edge_attributes in net.edges(data=True)
        if edge_attributes["weight"] > threshold and from_node != to_node
    ]
    subnet = nx.Graph()
    subnet.add_edges_from(eligible_edges)
    if nodes == []:
        return subnet
    return subnet.subgraph(nodes)


def draw_net(
    net: nx.Graph,
    df_labels2,
    pos,
    figsize: int = 15,
    title: str = "Jaccard connexity",
    edge_alpha: float = 0.5,
    node_alpha: float = 0.6,
    node_size: int = 3000,
) -> plt.Figure:
    """Draw a networkx graph using matplotlib.
       
       Parameters
       ----------
       net : nx.Graph
           The networkx graph to be drawn
       df_labels2 : pandas DataFrame
           DataFrame containing labels for the nodes in the graph
       pos : dict
           Dictionary with node positions for the graph
       figsize : int, optional
           The size of the figure (default is 15)
       title : str, optional
           The title of the figure (default is "Jaccard connexity")
       edge_alpha : float, optional
           The transparency of the edges (default is 0.5)
       node_alpha : float, optional
           The transparency of the nodes (default is 0.6)
       node_size : int, optional
           The size of the nodes (default is 3000)
           
       Returns
       -------
       fig : plt.Figure
           The resulting matplotlib figure
    """
    sub_labels = df_labels2.loc[np.array(net.nodes), :]
    fig = plt.figure(figsize=(figsize, figsize))
    plt.title(title, fontdict={"fontsize": figsize})

    edgewidth = [
        edge_attributes["weight"]
        for from_node, to_node, edge_attributes in net.edges(data=True)
    ]

    nx.draw_networkx_edges(
        net, pos, width=edgewidth, alpha=edge_alpha, edge_color="tab:grey"
    )

    for shape in set(sub_labels["node_shape"]):
        sub_temp = sub_labels.loc[sub_labels["node_shape"] == shape, :]
        nx.draw_networkx_nodes(
            net,
            pos,
            nodelist=sub_temp.index,
            node_color=np.array(sub_temp["node_color"]),
            node_size=np.array(node_size),
            alpha=node_alpha,
            edgecolors=np.array(sub_temp["edgecolors"]),
            node_shape=shape,
            linewidths=5,
        )
    nx.draw_networkx_labels(net, pos, font_size=10, font_color="black")
    plt.tight_layout()
    plt.axis("off")
    plt.show()
    return fig


def cluster_netfig(
    net: nx.Graph,
    df_labels2,
    fpos,
    nclust: int = 3,
    threshold: float = 0.8,
    posname: str = "jacc_kamada",
    scale1: int = 400,
    scale2: int = 150,
    size1: int = 70,
    size2: int = 30,
) -> tuple[plt.Figure, str]:
    """Generate a figure of a subnetwork of a given graph, based on a certain cluster and threshold.

    Parameters
    ----------
    net : nx.Graph
        The original graph.
    df_labels2 : pd.DataFrame
        Dataframe containing labels for the nodes in the graph.
    fpos : function
        Function to generate the position of the nodes in the subnetwork.
    nclust : int, optional
        The number of clusters of the approaches to display (default is 3).
        If nclust is -1, display all nodes.
    threshold : float, optional
        Threshold for the edges to include in the subnetwork (default is 0.8).
    posname : str, optional
        Name of the position function used (default is "jacc_kamada").
    scale1 : float, optional
        Scale factor for the position function when nclust is -1 (default is 400).
    scale2 : float, optional
        Scale factor for the position function when when nclust is an positiv (default is 150).
        
    Returns
    -------
    fig : plotly figure,
        figure of the subnetwork
    fig_name : str,
        name of the figure
    """
    if nclust == -1:
        nodes = list(df_labels2.index)
        fig_name = f"{posname}_ths{threshold}"
        scale = scale1
        figsize = size1
    else:
        nodes = list(df_labels2.loc[df_labels2["k"] == str(nclust)].index)
        fig_name = f"{posname}_k{nclust}_ths{threshold}"
        scale = scale2
        figsize = size2
    subnet = sub_net(net, threshold, nodes)
    pos = fpos(subnet, scale=scale)
    fig = draw_net(
        subnet,
        df_labels2,
        pos,
        figsize=figsize,
        title=fig_name.replace("_", " "),
        edge_alpha=0.5,
        node_alpha=0.6,
        node_size=3000,
    )
    return fig, fig_name
