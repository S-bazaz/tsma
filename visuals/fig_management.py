# -*- coding: utf-8 -*-
"""
functions used to manage the creation and saving of multiple figures
"""
__author__ = ["Samuel Bazaz"]
__credits__ = ["Samuel Bazaz"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Samuel Bazaz"]

##############
#  Packages  #
##############
import json
import re

import os
import sys

import plotly.io as pio
import numpy as np

#################
#  Importation  #
#################

path = os.path.dirname(os.getcwd())
sys.path.append(path)

from visuals.figures import (
    f_line_gross,
    adapted_mosaique,
    philips_okun_curves,
    adapted_mosaique_transient,
    mosaique_transient,
    mosaique_nsim,
    mosaique_ntransients,
    my_dendrogram,
    net_kamada_kawai,
    net_circular,
    net_spring,
    net_spectral,
    net_shell,
)

from basics.text_management import (
    dict_to_html,
    list_to_lines,
    get_clust_families,
    family_to_figname,
    get_nclust,
)
from analyses.clustering import get_faux_clust_ts

#####################
#  saves and query  #
#####################


def fig_to_image(fig, newpath_fig: str, figname: str, image_format: str = "png"):
    """Save an image of a figure if the image_format is right
    
    Parameters
    ----------

    fig : figure ( plotly object ),
        the figure you want to save
    newpath_fig: string,
        path of the file where you save
    figname : string,
        name of the image
    image_format : string,
    
    no Return
    -------
    """
    if image_format in ["pdf", "png", "jpeg"]:
        fig.write_image(
            newpath_fig + "/" + figname + "." + image_format,
            format=image_format,
            engine="kaleido",
            # width = 1980,
            # height = 1080,
        )


def save_fig(fig, newpath_fig: str, figname: str, save_format: str = "html") -> None:
    """Save an image if the  image_format is right and a html file of a figure
    Parameters
    ----------

    fig : figure ( plotly object ),
        the figure you want to save
    newpath_fig: string,
        path of the file where you save
    figname : string,
        name of the image
    image_format : string,
    no Return
    -------

    """
    if save_format == "html":
        fig.write_html(newpath_fig + "/" + figname + ".html")
    else:
        fig_to_image(fig, newpath_fig, figname, save_format)


def read_from_html(filepath: str):
    """Read an html file at filepath, for instance a figure

    Parameters
    ----------
    filepath : string,
        the path of the file
    Returns
    -------
    ... : html object,
        ex: plotly figure

    """
    # read the html file
    with open(filepath) as f:
        html = f.read()
    # convert in json object
    call_arg_str = re.findall(r"Plotly\.newPlot\((.*)\)", html[-(2 ** 16) :])[0]
    call_args = json.loads(f"[{call_arg_str}]")
    # convert the json object into plotly figure
    plotly_json = {"data": call_args[1], "layout": call_args[2]}
    return pio.from_json(json.dumps(plotly_json))


######################
#  default plotting  #
######################


def adapted_mos_default(output, title: str, grouping: list, var_compared: str) -> ():
    """Create a default adapted mosaique for Gross 2022 model
    
    Parameters
    ----------
    output : pd.DataFrame,
        results of the simulation
    title : str,
        title of the figure
    grouping : list,
        list of lists, each containing the list of variables to group together in the same column
    var_compared : str,
        variable that will be compared to the variables in the grouping
        
    Returns
    -------
    fig : go.Figure
    """
    title2 = (
        "<a href = 'https://doi.org/10.1016/j.ecolecon.2010.03.021' ><b>Gross 2022</b></a><br><br>"
        + title
    )

    return adapted_mosaique(
        output,
        title2,
        grouping,
        ncol=3,
        nskip=1,
        width=1600,
        vspace=0.025,
        yaxis_to_right=[var_compared],
    )


def transient_default(
    outputs,
    title: str,
    varnames: list,
    dct_groups: dict,
    sign: float,
    ncol: int,
    nskip: int,
) -> ():
    """Create an adapted mosaique for the transient function,
    with the auto grouping from auto_dct_groups
    
    Parameters
    ----------
    outputs : ndarray,
        results of the simulation
    title : str,
        name of the figure (title)
    varnames : list,
        names of the variables
    dct_groups : dict,
        dictionary with the groups of variables
    sign : float
        The significance of the test.
        For example, 0.1 corresponds to a 90% confidence interval.
    ncol : int,
        number of columns of the mosaic
    nskip : int, optional
        Number of empty columns to skip at the beginning of the figure
        
    Returns
    -------
    ... : go figure
    """
    return adapted_mosaique_transient(
        outputs,
        title,
        varnames,
        dct_groups,
        sign=sign,
        ncol=ncol,
        width=1700,
        height=2000,
        hspace=0.09,
        vspace=0.03,
        nskip=nskip,
        f_line=f_line_gross,
        yaxis_to_right=[],
    )


def okun_phillips_f_img(m, save_path: str, img_format: str) -> None:
    """Save a figure of the wage and price Phillips curves, 
    based on firms' micro data, and Okun's Law with unemployment rate and change
    in unemployment rate, based on macro data to a file.

    Parameters
    ----------
    m : Model
        Model object.
    save_path : str
        Path to save the figure.
    img_format : str
        Format of the saved figure (e.g. 'png', 'jpeg', 'pdf').
    """
    fig = philips_okun_curves(m)
    fig.savefig(save_path, dpi=200, format=img_format)


#####################
#  default savings  #
#####################


def save_transient(
    parameters: dict[str, any],
    hyper_parameters: dict[str, any],
    initial_values: dict[str, any],
    outputs: any,
    varnames: list[str],
    dct_groups: dict[str, list[str]],
    sim_id: str,
    path_figures: str,
    fig_format: str,
    sign: float = 0.1,
    ncol: int = 3,
    nskip: int = 1,
    ncol_para: int = 5,
    f_fig=transient_default,
) -> None:
    """Save a transient analysis figure.
    
    Parameters
    ----------
    parameters : dict
        simulation parameters
    hyper_parameters : dict
        simulation hyper-parameters
    initial_values : dict
        simulation initial values
    outputs : np.array
        simulation outputs
    varnames : list of str
        list of variable names
    dct_groups : dict
        dictionary of groups of variables
    sim_id : str
        simulation id
    path_figures : str
        path to the directory where to save the figure
    fig_format : str
        figure format
    sign : float, optional
        sign to be used in the adapted mosaique function, by default 0.1
    ncol : int, optional
        number of columns in the adapted mosaique function, by default 3
    nskip : int, optional
        number of rows to skip in the adapted mosaique function, by default 1
    ncol_para : int, optional
        number of columns to display the simulation parameters in, by default 5
    f_fig : function, optional
        function to create the figure, by default transient_default
    """

    name = (
        dict_to_html({"Transient_analysis": sim_id}, k_dec="b", v_dec="b")[0]
        + "<br><br>"
    )
    lst_para = (
        dict_to_html(parameters)
        + dict_to_html(hyper_parameters)
        + dict_to_html(initial_values)
    )
    title = name + list_to_lines(lst_para, ncol_para, "   ")
    fig = f_fig(outputs, title, varnames, dct_groups, sign, ncol, nskip)
    save_fig(fig, path_figures, "transient_analysis" + sim_id, fig_format)


########################
#  clustering savings  #
########################


def save_cluster_ts_analyses(
    outputs0: any,
    varnames0: list[str],
    path_figures: str,
    df_clusters: any,
    df_params: any,
    nsim_clust: int = 10,
    onesim: bool = True,
    nsim: bool = True,
    ci: bool = True,
    ci_clust: bool = True,
) -> None:
    """
    Save time series analyses for clusters.

    Parameters
    ----------
    outputs0 : np.array
        The outputs of the model.
    varnames0 : list[str]
        The list of variables to plot.
    path_figures : str
        The path to the directory where the figures will be saved.
    df_clusters : pd.DataFrame,
        The dataframe containing the clusters.
    df_params : pd.DataFrame,
        The dataframe containing the parameters.
    nsim_clust : int, optional
        The number of simulations to use in the "nsim" plot for each cluster, by default 10.
    onesim : bool, optional
        Whether to produce a "onesim" plot for each cluster, by default True.
    nsim : bool, optional
        Whether to produce an "nsim" plot for each cluster, by default True.
    ci : bool, optional
        Whether to produce a "CI" plot for all simulations of each cluster, by default True.
    ci_clust : bool, optional
        Whether to produce a "CI" plot for each cluster, by default True.
    """
    # Get some auxiliary functions
    fout, ftitle = get_faux_clust_ts(df_params, outputs0)

    # Define a function to save figures
    fsave = lambda fig, figname: save_fig(
        fig, path_figures, figname=figname, save_format="png"
    )
    df_clusters2 = df_clusters.copy()
    for code in df_clusters2["clustering_code"].unique():
        lst_outputs = []
        lst_sim_ids = []
        sim_ids_1sim = []

        for cluster_id in df_clusters2.loc[
            df_clusters2["clustering_code"] == code, "cluster_id"
        ].values:

            print(cluster_id)

            sim_ids = df_clusters2.loc[cluster_id, "sim_ids"]
            if len(sim_ids) > 0:
                outputs = fout(sim_ids)

                lst_sim_ids.append(sim_ids)
                lst_outputs.append(outputs)
                sim_ids_1sim.append(sim_ids[0])

                # CI
                if ci_clust:
                    figname = f"CI_{cluster_id}"
                    title = ftitle([sim_ids], figname)
                    fsave(
                        mosaique_transient(outputs, 0.1, 5, title, varnames0, nskip=3),
                        figname,
                    )
                if nsim:
                    sim_ids_nsim = sim_ids[:nsim_clust]
                    outputs_nsim = fout(sim_ids_nsim)
                    figname = f"nSim_{cluster_id}"
                    title = ftitle([sim_ids_nsim], figname)
                    fsave(
                        mosaique_nsim(
                            outputs_nsim, 5, title, varnames0, pal="viridis", nskip=3
                        ),
                        figname,
                    )
        nclust = get_nclust(code)
        lst_outputs = lst_outputs[:nclust]
        lst_sim_ids = lst_sim_ids[:nclust]
        sim_ids_1sim = sim_ids_1sim[:nclust]

        if onesim:
            outputs_1sim = fout(sim_ids_1sim)
            figname = f"1Sim_{code}"
            title = ftitle(sim_ids_1sim, figname)
            fsave(
                mosaique_nsim(
                    outputs_1sim, 5, title, varnames0, pal="viridis", nskip=3
                ),
                figname,
            )
        if ci:
            figname = f"CI_{code}"
            title = ftitle(lst_sim_ids, figname)
            fsave(
                mosaique_ntransients(
                    lst_outputs, 0.1, 5, title, varnames0, pal="viridis", nskip=3
                ),
                figname,
            )


def save_cluster_dendrogram(
    df_clusters2: any, df_jaccdist: any, path_figures: str, threshold: float = 0.8
) -> None:
    """Save dendrograms for clusters.
    
    Copy code
    Parameters
    ----------
    df_clusters2 : any
        DataFrame with cluster information.
    df_jaccdist : any
        DataFrame with Jaccard distance between clusters.
    path_figures : str
        Path where to save the figures.
    threshold : float, optional
        Minimum Jaccard distance to be considered a cluster, by default 0.8.
    
    Returns
    -------
    None
    """

    fsave = lambda fig, figname: save_fig(
        fig, path_figures, figname=figname, save_format="png"
    )
    # Get the families of clusters
    clust_families = get_clust_families(df_clusters2)
    # Save dendrograms for each family
    for family, labels in clust_families.items():
        fig_code = family_to_figname(family)
        print(fig_code)
        figname = f"Dendrogram_{fig_code}"
        df = df_jaccdist.loc[labels, labels]
        fig = my_dendrogram(df, figname.replace("_", " "), threshold=threshold)
        fsave(fig, figname)
    # Save total dendrogram
    figname = "Dendrogram_total"
    fig = my_dendrogram(df_jaccdist, figname.replace("_", " "), threshold=threshold)
    fsave(fig, figname)


#################################
#  clustering networks savings  #
#################################

def save_cluster_netfigs(
    net: any,
    df_labels2: any,
    path_figure: str,
    f_fig,
    thresholds: list[float] = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
    scale1: int = 400,
    scale2: int = 150,
    size1: int = 70,
    size2: int = 30,
) -> None:
    """
    Save network figures for different clustering configurations.

    Parameters
    ----------
    net : nx.Graph,
        Network.
    df_labels2 : pd.DataFrame,
        Dataframe with labels.
    path_figure : str
        Path to save figures.
    f_fig :
        Function to create the figure.
    thresholds : List[float], optional
        List of thresholds, 
        by default [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    scale1 : int, optional
        Scale for large clusters, by default 400
    scale2 : int, optional
        Scale for small clusters, by default 150
    size1 : int, optional
        Size for large clusters, by default 70
    size2 : int, optional
        Size for small clusters, by default 30
    """

    def aux_save(nclust, threshold):
        fig, figname = f_fig(
            net,
            df_labels2,
            nclust=nclust,
            threshold=threshold,
            scale1=scale1,
            scale2=scale2,
            size1=size1,
            size2=size2,
        )
        fig.savefig(os.sep.join([path_figure, f"{figname}.png"]))

    kmin = int(df_labels2["k"].min())
    kmax = int(df_labels2["k"].max())
    lst_k = np.arange(kmin, kmax + 1)
    for thres in thresholds:
        for k in lst_k:
            print(f"threshold {thres} nclust {k}")
            aux_save(k, thres)
        aux_save(-1, thres)


def save_kamada_kawai(
    net: any,
    df_labels2: any,
    path_figure: str,
    thresholds: list[float] = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
) -> None:
    """
    Save figures of a network using the Kamada-Kawai algorithm for layout.

    Parameters
    ----------
    net : nx.Graph,
        The network to visualize.
    df_labels2 : pd.DataFrame,
        DataFrame with labels for the nodes in the network.
    path_figure : str
        Path to save the figures.
    thresholds : list[float], optional
        List of thresholds to use for the visualization. 
        Defaults to [0.1, 0.2, 0.4, 0.6, 0.8, 0.9].
    """
    save_cluster_netfigs(
        net,
        df_labels2,
        path_figure,
        net_kamada_kawai,
        thresholds,
        scale1=400,
        scale2=150,
        size1=70,
        size2=30,
    )


def save_circular(
    net: any,
    df_labels2: any,
    path_figure: str,
    thresholds: list[float] = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
) -> None:
    """
    Save figures of a network using the circular layout.

    Parameters
    ----------
    net : nx.Graph,
        The network to visualize.
    df_labels2 : pd.DataFrame,
        DataFrame with labels for the nodes in the network.
    path_figure : str
        Path to save the figures.
    thresholds : list[float], optional
        List of thresholds to use for the visualization. Defaults to [0.1, 0.2, 0.4, 0.6, 0.8, 0.9].
    """
    save_cluster_netfigs(
        net,
        df_labels2,
        path_figure,
        net_circular,
        thresholds,
        scale1=400,
        scale2=150,
        size1=70,
        size2=30,
    )


def save_spring(
    net: any,
    df_labels2: any,
    path_figure: str,
    thresholds: list[float] = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
) -> None:
    """
    Save figures of a network using the Spring algorithm for layout.

    Parameters
    ----------
    net : nx.Graph,
        The network to visualize.
    df_labels2 : pd.DataFrame,
        DataFrame with labels for the nodes in the network.
    path_figure : str
        Path to save the figures.
    thresholds : list[float], optional
        List of thresholds to use for the visualization. Defaults to [0.1, 0.2, 0.4, 0.6, 0.8, 0.9].
    """
    save_cluster_netfigs(
        net,
        df_labels2,
        path_figure,
        net_spring,
        thresholds,
        scale1=400,
        scale2=150,
        size1=70,
        size2=30,
    )


def save_spectral(
    net: any,
    df_labels2: any,
    path_figure: str,
    thresholds: list[float] = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
) -> None:
    """
    Save figures of a network using the Spectral algorithm for layout.

    Parameters
    ----------
    net : nx.Graph,
        The network to visualize.
    df_labels2 : pd.DataFrame,
        DataFrame with labels for the nodes in the network.
    path_figure : str
        Path to save the figures.
    thresholds : list[float], optional
        List of thresholds to use for the visualization. Defaults to [0.1, 0.2, 0.4, 0.6, 0.8, 0.9].
    """
    save_cluster_netfigs(
        net,
        df_labels2,
        path_figure,
        net_spectral,
        thresholds,
        scale1=400,
        scale2=150,
        size1=70,
        size2=30,
    )


def save_shell(
    net: any,
    df_labels2: any,
    path_figure: str,
    thresholds: list[float] = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
) -> None:
    """
    Save figures of a network using the Shell layout.

    Parameters
    ----------
    net : nx.Graph,
        The network to visualize.
    df_labels2 : pd.DataFrame,
        DataFrame with labels for the nodes in the network.
    path_figure : str
        Path to save the figures.
    thresholds : list[float], optional
        List of thresholds to use for the visualization. Defaults to [0.1, 0.2, 0.4, 0.6, 0.8, 0.9].
    """
    save_cluster_netfigs(
        net,
        df_labels2,
        path_figure,
        net_shell,
        thresholds,
        scale1=400,
        scale2=150,
        size1=70,
        size2=30,
    )
