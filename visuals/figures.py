# -*- coding: utf-8 -*-
"""
functions used to create figures,
using plotly, seabborn, and networkx

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

import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx

from plotly.figure_factory import create_dendrogram
from sklearn.metrics import r2_score

#################
#  Importation  #
#################

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from tsma.basics.transfers import auto_dct_groups

from tsma.basics.text_management import name_to_rgb

from tsma.visuals.fig_constructors import (
    add_trace_mos,
    add_traces_mos,
    construct_mosaique_1sim,
    construct_mosaique_nsim,
    add_trace_adaptmos,
    add_traces_adaptmos,
    construct_adaptmos,
    cluster_netfig,
)
from tsma.analyses.vandin import transient_analysis

# pip install -U kaleido
pio.renderers.default = "browser"

####################
#  Mosaique Plots  #
####################


def mosaique(
    output,
    ncol: int,
    title: str,
    width: int = 1600,
    height: int = 2000,
    pal: str = "Spectral",
) -> go.Figure:
    """Create a subplot of the outcomes with different colors.

    Parameters
    ----------
    output : pd.DataFrame
        The output of a simulation
    ncol : int
        Number of columns in the mosaique
    title : str
        Title of the figure
    width : int, optional
        Width of the figure, by default 1600
    height : int, optional
        Height of the figure, by default 2000
    pal : str, optional
        Color palette to use, by default "Spectral"

    Returns
    -------
    subplot : plotly.graph_objs.figure.Figure
        A plotly figure containing the subplots of the simulation outputs
    """

    def add_plot(subplot, i, j, isim):
        go_fig = go.Scatter(
            x=output.index,
            y=output.iloc[:, i * ncol + j],
            line=dict(
                width=1.5,
                color=sns.color_palette(pal, np.shape(output)[1]).as_hex()[
                    i * ncol + j
                ],
            ),
        )
        add_trace_mos(subplot, go_fig, i, j)

    return construct_mosaique_1sim(output, ncol, title, add_plot, width, height)


# Hist
def mosaique_hist(
    output,
    ncol: int,
    title: str,
    width: int = 1600,
    height: int = 2000,
    pal: str = "Spectral",
) -> go.Figure:
    """Create a subplot of histogrames of the outcomes with different colors

    Parameters
    ----------
    output : pd.DataFrame,
        the output of a simulation
    ncol : int,
        number of columns in the mosaique
    title : str,
        give the title
    width : int,
        width of the figure
    height : int,
        height of the figure
    pal : str,
        name of the color palette to use
        
    Returns
    -------
    subplot : plotly figure
    """

    def add_plot(subplot, i, j, isim):
        go_fig = go.Histogram(
            x=output.iloc[:, i * ncol + j],
            marker=dict(
                color=sns.color_palette(pal, np.shape(output)[1]).as_hex()[i * ncol + j]
            ),
            nbinsx=100,
        )
        add_trace_mos(subplot, go_fig, i, j)

    return construct_mosaique_1sim(output, ncol, title, add_plot, width, height)


########################
#  Mosaique Plots nsim #
########################


def mosaique_nsim(
    outputs: np.array,
    ncol: int,
    title: str,
    varnames: list[str],
    width: int = 1600,
    height: int = 2000,
    pal: str = "Spectral",
    nskip: int = 0,
) -> go.Figure:
    """
    Create a subplot of the outcomes with different colors for multiple simulations
    
    Parameters
    ----------
    outputs : 3D array
        the output of the simulations
    ncol : int,
        number of columns in the mosaique
    title : str,
        give the title
    varnames : list[str],
        list of the names of the variables
    width : int, optional
        width of the figure (default is 1600)
    height : int, optional
        height of the figure (default is 2000)
    pal : str, optional
        color palette (default is "Spectral")
    nskip : int, optional
        number of empty plots to skip in the beginning (default is 0)
    
    Returns
    -------
    subplot : plotly figure
    """
    time = np.arange(len(outputs))
    nvar = len(outputs[0])
    varnames2 = [""] * nskip + list(varnames)
    nvar2 = nvar + nskip
    if len(np.shape(outputs)) == 3:
        nsim = len(outputs[0][0])
    else:
        nsim = 1

    def add_plot(subplot, i, j, isim):
        ivar = i * ncol + j - nskip
        if ivar >= 0:
            go_fig = go.Scatter(
                x=time,
                y=outputs[:, ivar, isim] if nsim > 1 else outputs[:, ivar],
                line=dict(width=1.5, color=sns.color_palette(pal, nsim).as_hex()[isim]),
            )
            add_trace_mos(subplot, go_fig, i, j)

    return construct_mosaique_nsim(
        nvar2, ncol, title, varnames2, add_plot, nsim, width, height
    )


# Hist
def mosaique_hist_nsim(
    outputs: np.array,
    ncol: int,
    title: str,
    varnames: list[str],
    width: int = 1600,
    height: int = 2000,
    pal: str = "Spectral",
    nskip: int = 0,
) -> go.Figure:
    """
    Create a subplot of histograms of the outcomes with different colors for multiple simulations.

    Parameters
    ----------
    outputs : np.array
        3D array of the output of the simulations.
    ncol : int
        Number of columns in the mosaique.
    title : str
        Title of the figure.
    varnames : list[str]
        List of variable names for each column of `outputs`.
    width : int, optional
        Width of the figure, by default 1600.
    height : int, optional
        Height of the figure, by default 2000.
    pal : str, optional
        Color palette for the histograms, by default "Spectral".
    nskip : int, optional
        Number of empty columns to skip at the beginning of the figure, by default 0.

    Returns
    -------
    go.Figure
        Subplot of histograms.
    """
    nvar = len(outputs[0])
    varnames2 = [""] * nskip + list(varnames)
    nvar2 = nvar + nskip

    if len(np.shape(outputs)) == 3:
        nsim = len(outputs[0][0])
    else:
        nsim = 1

    def add_plot(subplot, i, j, isim):
        ivar = i * ncol + j - nskip
        if ivar >= 0:
            go_fig = go.Histogram(
                x=outputs[:, ivar, isim] if nsim > 1 else outputs[:, ivar],
                nbinsx=100,
                marker=dict(color=sns.color_palette(pal, nsim).as_hex()[isim]),
            )
            add_trace_mos(subplot, go_fig, i, j)

    return construct_mosaique_nsim(
        nvar2, ncol, title, varnames2, add_plot, nsim, width, height
    )


########################
#  Mosaique transient  #
########################


def mosaique_transient(
    outputs: np.array,
    sign: float,
    ncol: int,
    title: str,
    varnames: list[str],
    width: int = 1600,
    height: int = 2000,
    pal: str = "Spectral",
    nskip: int = 0,
) -> go.Figure:
    """Create a subplot of transient analyses based on multiple simulations.

    Parameters
    ----------
    outputs : 3D array
        The output of the simulations.
    sign : float
        The significance of the test. For example, 0.1 corresponds to a 90% confidence interval.
    ncol : int
        The number of columns in the mosaique.
    title : str
        The title of the figure.
    varnames : list of str
        The names of the variables to be plotted.
    width : int, optional
        The width of the figure, by default 1600.
    height : int, optional
        The height of the figure, by default 2000.
    pal : str, optional
        The color palette to use for the plots, by default "Spectral".
    nskip : int, optional
        Number of empty columns to skip at the beginning of the figure, by default 0.

    Returns
    -------
    go.Figure
        The plotly figure object for the subplot.
    """

    mean, d = transient_analysis(outputs, sign)
    time = np.arange(len(outputs))
    nvar = len(outputs[0])

    varnames2 = [""] * nskip + list(varnames)
    nvar2 = nvar + nskip

    def add_plot(subplot, i, j, isim):
        ivar = i * ncol + j - nskip
        if ivar >= 0:
            color = sns.color_palette(pal, nvar).as_hex()[ivar]

            lst_go_fig = [
                go.Scatter(x=time, y=mean[:, ivar], line=dict(width=1.5, color=color)),
                go.Scatter(
                    x=time,
                    y=mean[:, ivar] + 0.5 * d[:, ivar],
                    line=dict(width=1.5, color=color),
                ),
                go.Scatter(
                    x=time,
                    y=mean[:, ivar] - 0.5 * d[:, ivar],
                    fill="tonexty",
                    line=dict(width=1.5, color=color),
                ),
            ]
            add_traces_mos(subplot, lst_go_fig, i, j)

    return construct_mosaique_nsim(
        nvar2, ncol, title, varnames2, add_plot, 1, width, height
    )


def mosaique_ntransients(
    lst_outputs: list[np.array],
    sign: float,
    ncol: int,
    title: str,
    varnames: list[str],
    width: int = 1600,
    height: int = 2000,
    pal: str = "Spectral",
    nskip: int = 0,
) -> go.Figure:
    """Create a subplot of transient analyses based on multiple simulations

    Parameters
    ----------
    lst_outputs : list[np.array]
        a list of the outputs of the simulations
    sign : float,
        the significance of the test. For example, 0.1 corresponds to a 90% confidence interval.
    ncol : int,
        number of columns in the mosaic plot
    title : string,
        the title of the plot
    varnames : list[str],
        a list of names for the variables
    width : int, optional
        the width of the plot (default is 1600)
    height : int, optional
        the height of the plot (default is 2000)
    pal : str, optional
        the name of the color palette to use (default is "Spectral")
    nskip : int, optional
        Number of empty columns to skip at the beginning of the figure, by default 0.
    
    Returns
    -------
    subplot : plotly.graph_objects.Figure
        a plotly figure object containing the mosaic plot of transient analyses
    """

    time = np.arange(len(lst_outputs[0]))
    nvar = len(lst_outputs[0][0])
    varnames2 = [""] * nskip + list(varnames)
    nvar2 = nvar + nskip
    lst_mean = []
    lst_d = []

    for outputs in lst_outputs:
        mean, d = transient_analysis(outputs, sign)
        lst_mean.append(mean)
        lst_d.append(d)
    ntrace = len(lst_outputs)

    def add_plot(subplot, i, j, itrace):
        ivar = i * ncol + j - nskip
        if ivar >= 0:
            # color = "green"
            color = sns.color_palette(pal, ntrace).as_hex()[itrace]
            mean = lst_mean[itrace]
            d = lst_d[itrace]

            lst_go_fig = [
                go.Scatter(x=time, y=mean[:, ivar], line=dict(width=1.5, color=color)),
                go.Scatter(
                    x=time,
                    y=mean[:, ivar] + 0.5 * d[:, ivar],
                    line=dict(width=1.5, color=color),
                ),
                go.Scatter(
                    x=time,
                    y=mean[:, ivar] - 0.5 * d[:, ivar],
                    fill="tonexty",
                    line=dict(width=1.5, color=color),
                ),
            ]
            add_traces_mos(subplot, lst_go_fig, i, j)

    return construct_mosaique_nsim(
        nvar2, ncol, title, varnames2, add_plot, ntrace, width, height
    )


####################
#  style functions #
####################


def color_set_gross(
    varname: str, k: int, c: int, lst: list[str] = ["green", "goldenrod", "purple"]
) -> str:
    """Associate a color to the variable for adapted_mosaique and the Gross model

    Parameters
    ----------
    varname : str
        Name of the variable.
    k : int
        The number of the curve in the plot.
    c : int
        Adjustment parameter.
    lst : list[str], optional
        List of colors to use, by default ["green", "goldenrod", "purple"]

    Returns
    -------
    str
        The color to use for the given variable.
    """
    if varname == "unemployment_rate":
        return "darkorchid"
    elif k < len(lst):
        return lst[k]
    return name_to_rgb(varname, c)


def type_line_gross(varname: list[str]) -> str:
    """Determine the type of line to use for a given variable in adapted_mosaique,
    for the Gross model
    
    Parameters
    ----------
    varname : list[str]
        A list of strings representing the name of the variable.
    
    Returns
    -------
    str
        The type of line to use for the given variable. This will be either "dashdot" or "solid".
    """
    lst = varname.split("_")
    if "tgt" in lst:
        return "dashdot"
    return "solid"


def f_line_gross(varname: str, k: int) -> tuple[str, str, float, float]:
    """Returns the line style, color, and width for a given variable name and index.

    Parameters
    ----------
    varname : str
        The name of the variable.
    k : int
        The index of the curve in the plot.

    Returns
    -------
    tuple[str, str, float, float]
        A tuple containing the line style, color, width, and dash length.
    """
    return type_line_gross(varname), color_set_gross(varname, k, 10), 0.5, 1.5


def color_sector(
    varname: str,
    c: int,
    dct: dict[str, str] = {
        "energy": "goldenrod",
        "resources": "green",
        "goods": "purple",
    },
) -> str:
    """Associate a color to a sector.

    Parameters
    ----------
    varname: str
        Name of the variable.
    c: int
        Adjustment parameter.
    dct: dict[str, str]
        Dictionary mapping sector names to colors.

    Returns
    -------
    str
        The color to be associated with the sector.
    """
    suffix = varname.split("_")[-1]
    if suffix in dct:
        return dct[suffix]
    else:
        return name_to_rgb(varname, c)


def type_line_em(varname: str) -> str:
    """Assign a type of line to the variable for use in the 'adapted_mosaique' function
    for the threesector model
    
    Parameters
    ----------
    varname : str
        Name of the variable

    Returns
    -------
    str
        Type of line to use for the variable's curve
    """

    lst = varname.split("_")
    if "dem" in lst:
        return "dot"
    elif "tgt" in lst:
        return "dash"
    elif "cons" in lst:
        return "dashdot"
    else:
        return "solid"


def f_line_em(
    varname: str,
    k: int,
    dct: dict = {"en": "goldenrod", "re": "green", "go": "purple"},
    c=10,
):
    """Assign a line type and color to a variable for use in adapted_mosaique
    for the threesector model
    
    Parameters
    ----------
    varname : str
        Name of the variable
    k : int
        The number of the curve in the plot
    dct : dict, optional
        Dictionary mapping sector suffixes to colors,
        by default {"en": "goldenrod", "re": "green", "go": "purple"}
    c : int, optional
        Adjustment parameter, by default 10

    Returns
    -------
    tuple
        Tuple containing the line type, color, and widths for the variable
    """
    return type_line_gross(varname), color_sector(varname, c, dct), 0.5, 1.5


##################################
#  Adapted mosaique constructors #
##################################


def my_scatter_adaptmos(
    x: np.array,
    y: np.array,
    k: int,
    varname: str,
    b_lower: bool = False,
    f_line=f_line_gross,
) -> go.Scatter:
    """Create a customized go.Scatter for use in adapted mosaiques.
    
    Parameters
    ----------
    x : np.array
        x axis data
    y : np.array
        y axis data
    k : int
        Number of the variable in the group
    varname : str
        Name of the variable
    b_lower : bool, optional
        Indicates whether the curve is the lowest in a 
        confidence interval plot (default is False)
    f_line : function, optional
        Function that returns the line type, color, opacity, 
        and width for the scatter plot (default is f_line_gross)
        
    Returns
    -------
    go.Scatter
        Customized scatter plot
    """
    type_line, color, opa, width = f_line(varname, k)
    return go.Scatter(
        x=x,
        y=y,
        name=varname,
        opacity=opa,
        line=dict(shape="linear", width=width, dash=type_line, color=color,),
        fill="tonexty" if b_lower else None,
    )


######################
#  Adapted mosaique  #
######################


def adapted_mosaique(
    output,
    title: str,
    dct_groups: dict[str, list[str]],
    ncol: int = 3,
    width: int = 1700,
    height: int = 2000,
    hspace: float = 0.07,
    vspace: float = 0.03,
    f_line=f_line_gross,
    yaxis_to_right: list[str] = ["unemployment_rate"],
    nskip: int = 0,
) -> go.Figure:

    """
    Create a subplot of the outcomes with different colors, by groups

    Parameters
    ----------
    output : pd.DataFrame
        Dataframe of the output of a simulation
    title : str
        Title of the plot
    dct_groups : Dict[str, List[str]]
        Dictionary with keys as the title of the subplots
        and values as lists of strings representing the names 
        of the variables in each subplot
    ncol : int, optional
        Number of columns in the mosaique
    width : int, optional
        Width of the figure
    height : int, optional
        Height of the figure
    vspace : float, optional
        Vertical space between plots
    hspace : float, optional
        Horizontal space between plots
    f_line : function, optional
        Function to customize the line style of each trace in the plot
    yaxis_to_right : List[str], optional
        List of variable names whose y-axis should be to the right of the plot
   nskip : int, optional
       Number of empty columns to skip at the beginning of the figure, by default 0.

    Returns
    -------
    subplot : plotly.graph_objs.Figure
        Plotly figure object
    """
    groupnames = [""] * nskip + list(dct_groups.keys())
    groups = [[""]] * nskip + list(dct_groups.values())

    def add_plot(subplot, i, j, isim):
        igr = i * ncol + j
        if igr >= nskip:
            for k in range(len(groups[igr])):
                varname = groups[igr][k]
                if varname != "":

                    go_fig = my_scatter_adaptmos(
                        output.index, output[varname], k, varname, f_line=f_line
                    )
                    add_trace_adaptmos(
                        subplot, varname, go_fig, i, j, yaxis_to_right=yaxis_to_right
                    )

    return construct_adaptmos(
        ncol, title, groupnames, groups, add_plot, width, height, hspace, vspace,
    )


def adapted_mosaique_transient(
    outputs,
    title: str,
    varnames: list[str],
    dct_groups: dict[str, list[str]],
    sign=0.1,
    ncol: int = 3,
    width: int = 1700,
    height: int = 2000,
    hspace: float = 0.07,
    vspace: float = 0.03,
    f_line=f_line_gross,
    yaxis_to_right: list[str] = ["unemployment_rate"],
    nskip: int = 0,
) -> go.Figure:

    """Create a subplot of the outcomes with different colors, by groups

    Parameters
    ----------
    outputs: np.array,
        of the output of multiple simulations
    title: str,
        give the title
    varnames: list[str],
        list of names of the variables to plot
    dct_groups: dict[str, list[str]],
        dictionary specifying how to group the plots, 
        with keys being group names and values being lists of variable names
    sign: float,
        level of confidence used in the transient analysis
    ncol: int,
        number of columns in the mosaique
    width: int,
        width of the figure
    height: int,
        height of the figure
    vspace: float,
        vertical space between plots
    hspace: float,
        horizontal space between plots
    f_line: function,
        function used to customize line properties in the plots
    yaxis_to_right: list[str],
        list of variable names to plot with the y-axis on the right
    nskip: int,
        number of plots to skip at the beginning of the subplot

    Returns
    -------
    subplot: plotly figure
    """
    groupnames = [""] * nskip + list(dct_groups.keys())
    groups = [[""]] * nskip + list(dct_groups.values())
    mean, d = transient_analysis(outputs, sign)
    time = np.arange(len(mean))

    def add_plot(subplot, i, j, isim):

        for k in range(len(groups[i * ncol + j])):
            varname = groups[i * ncol + j][k]
            if varname != "":
                index = varnames.index(varname)
                f_go = lambda x, b_lower: my_scatter_adaptmos(
                    time, x, k, varname, f_line=f_line, b_lower=b_lower
                )
                lst_go_fig = [
                    f_go(mean[:, index], False),
                    f_go(mean[:, index] + 0.5 * d[:, index], False),
                    f_go(mean[:, index] - 0.5 * d[:, index], True),
                ]

                add_traces_adaptmos(
                    subplot, varname, lst_go_fig, i, j, yaxis_to_right=yaxis_to_right
                )

    return construct_adaptmos(
        ncol, title, groupnames, groups, add_plot, width, height, hspace, vspace,
    )


###########################
#  Okun & Philips curves  #
###########################


def r2_for_polyreg(x: np.array, y: np.array) -> float:
    """Give the R2 of a polynomial regression of order 2, based on x and y

    Parameters
    ----------
    x : n-array,
        abscissa
    y : 2D-array of shape (n,1),
        ordinate
    
    Returns
    -------
    ... : float,
    """
    z = (np.polyfit(x, y, 2).T)[0]
    pred = np.poly1d(z)
    return round(r2_score(y.T[0], pred(x)), 4)


def philips_okun_curves(m, r2: bool = True) -> plt.Subplot:
    """Create a subplot with :
        - Wage and Price Phillips curves, based on firms' micro data. 
        - Okun's Law with unemployment Rate and Change in unemployment Rate, based on macro data
    !!!! 
        The micro-data is not saved on databases,
        So it has to be calculated before using this function 
        To have r2 option, you would need to have enough points 
        for instance: f_n, hh_n, t_end == ( 10, 10, 30)
    !!!!

    Parameters
    ----------
    m : Gross2020 model,
        Gross2020 model object containing micro and macro data
    r2 : bool, optional
        Indicates whether to include R2 values in the plot titles, by default True

    Returns
    -------
    fig : matplotlib figure
        Matplotlib figure object containing the Phillips and Okun's Law curves
    """
    f_proc = lambda x, lst: x[lst].groupby(lst[0]).agg(["mean"])
    f_trace = lambda df, y, ax: sns.regplot(
        x=df.index, y=y, data=df, order=2, ax=ax, scatter_kws={"alpha": 0.3},
    )
    f_set = lambda ax, title, x, y: ax.set(title=title, xlabel=x, ylabel=y)
    ## data processing--------------------------------------------------------
    df = f_proc(m.output_micro, ["unemployment_rate", "inflation", "wage_inflation"])
    df1 = f_proc(m.output, ["unemployment_rate", "real_GDP_growth"])
    df2 = f_proc(m.output, ["unemployment_change", "real_GDP_growth"])

    ## plotting----------------------------------------------------------------
    plt.style.use("seaborn")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    f_trace(df, "inflation", axs[0, 0])
    f_trace(df, "wage_inflation", axs[0, 1])
    f_trace(df1, "real_GDP_growth", axs[1, 0])
    f_trace(df2, "real_GDP_growth", axs[1, 1])

    title1 = "Wage Philips Curve"
    title2 = "Price Philips Curve"
    title3 = "Okun's Law"
    title4 = "Okun's Law 2"
    if r2:
        title1 += f' (R2 = {r2_for_polyreg(df.index, df[["inflation"]].values)})'
        title2 += f' (R2 = {r2_for_polyreg(df.index, df[["wage_inflation"]].values)})'
        title3 += f" (R2 = {r2_for_polyreg(df1.index, df1.values)})"
        title4 += f" (R2 = {r2_for_polyreg(df2.index, df2.values)})"
    f_set(axs[0, 0], title1, "Unemployment rate", "Wage Inflation")
    f_set(axs[0, 1], title2, "Unemployment rate", "Price Inflation")
    f_set(axs[1, 0], title3, "Unemployment rate", "Real GDP Growth")
    f_set(axs[1, 1], title4, "Change in unemployment rate", "Real GDP Growth")

    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    return fig


#####################################
#  auto adapted mosaique functions  #
#####################################


def auto_mosaique(
    output, lst_suffix: list[str], name: str = "test", colpara: int = 1
) -> go.Figure:
    """Create an adapted mosaique with the auto grouping from auto_dct_groups
    
    Parameters
    ----------
    output : dataframe,
        results of the simulation
    lst_suffix : list,
        ex: ["energy", "resources", "goods"]
    name : string,
        name of the figure (title)
    colpara : int,
        number of groups of columns by rows
    the total number of columns is given by colpara*n_sector
        
    Returns
    -------
    ... : go figure
    """

    groupnames, groups = auto_dct_groups(list(output), lst_suffix)
    return adapted_mosaique(output, colpara * len(lst_suffix), name, groupnames, groups)


######################
#  clutering  plots  #
######################


def my_heatmap(df_sign, name: str, size: int = 15, pal: str = "viridis") -> plt.Figure:
    """Create a heatmap from a pandas DataFrame.
    
    Parameters
    ----------
    df_sign : pd.DataFrame,
        the data to be plotted as a heatmap
    name : str,
        the title of the plot
    size : int,
        the size of the figure (default is 15)
    pal : str,
        the color palette to use for the heatmap (default is "viridis")
        
    Returns
    -------
    fig : plt.Figure
        the figure object containing the heatmap plot
    """
    plt.style.use("seaborn")
    fig = plt.figure(figsize=(size, size))
    plt.title(name)
    sns.heatmap(df_sign, cmap=pal)
    return fig


def distance_hist(df, dist_name: str) -> go.Figure:
    """
    Create histograms for each column of df faceted by the other columns,
    with the params_id column used to color the bars.
    
    Parameters:
    ----------
    df (pd.DataFrame): DataFrame containing the data to plot.
    dist_name (str): Title for the histogram plot.
    
    Returns:
    -------
    go.Figure: Plotly figure object for the histogram plot.
    """
    fig = px.histogram(
        df,
        x=list(set(df.columns) - set(["params_id", "sim_id"])),
        facet_col=list(set(df.columns) - set(["params_id", "sim_id"])),
        facet_col_wrap=5,
        title=dist_name,
        color=df["params_id"],
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    return fig


def phase_diag_pairplot(
    df_params2,
    clust_code: str,
    size: int = 5,
    height: int = 2,
    aspect: int = 1,
    thr_low: float = 0.2,
    thr_upp: float = 0.5,
    alpha_low: float = 0.2,
    alpha_upp: float = 0.4,
    alpha_diag: float = 0.07,
    linewidth_diag: float = 0.6,
    palette: str = "Spectral",
) -> plt.Figure:
    """
    Create a pairplot with the seaborn library with a density estimation of a given variable 
    on the diagonal for each cluster. And a bivariate distributions 
    by kernel density estimation (KDE) on the other plots.
    The upper KDE has a higher density threshold and thus shows the main clusters only.
    
    Parameters
    ----------
    df_params2: pd.DataFrame
        Dataframe of parameters with a "cluster" column
    clust_code: str
        Title of the figure
    height: int
        height of each subplot
    aspect: int
        aspect ratio of each subplot
    thr_low: float
        threshold of the kde plots on the lower plots
    thr_upp: float
        threshold of the kde plots on the upper plots
    alpha_low: float
        alpha parameter of the lower plots
    alpha_upp: float
        alpha parameter of the upper plots
    alpha_diag: float
        alpha parameter of the diagonal plots
    linewidth_diag: float
        linewidth parameter of the diagonal plots
    palette: str
        name of the seaborn palette to use for the hue parameter
        
    Returns
    -------
    fig: plt.Figure
        Pairplot figure object
    """
    pp = sns.pairplot(
        df_params2.iloc[:, 1:],
        hue="cluster",
        height=height,
        aspect=aspect,
        diag_kind="kde",
        diag_kws={"linewidth": linewidth_diag, "alpha": alpha_diag},
        palette=palette,
    )

    pp.map_lower(
        sns.kdeplot, thresh=thr_low, levels=2, color=".2", fill=True, alpha=alpha_low
    )
    pp.map_upper(
        sns.kdeplot, thresh=thr_upp, levels=15, color=".2", fill=True, alpha=alpha_upp
    )
    fig = pp.fig
    # fig.update_layout(figsize=(size, size))
    fig.set_size_inches(size, size)
    fig.set_dpi(200)
    fig.subplots_adjust(top=0.93, wspace=0.3)
    fig.suptitle("Phase Diagram Paire Plots " + clust_code, fontsize=16)
    return fig


#############################
#  clutering network plots  #
#############################


def my_dendrogram(
    df, title: str, threshold: float = 0.8, height_para: int = 33, width_para: int = 300
) -> go.Figure:
    """
    Create a dendrogram visualization of the input dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data to be visualized.
    title : str
        Title of the dendrogram.
    threshold : float, optional
        Threshold value for coloring the dendrogram branches, by default 0.8.
    height_para : int, optional
        Height parameter for the dendrogram, by default 33.
    width_para : int, optional
        Width parameter for the dendrogram, by default 300.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Dendrogram visualization.
    """
    n = len(df)
    height = height_para * n
    width = int(np.log(n)) * width_para
    fig = create_dendrogram(
        df, labels=df.index, orientation="left", color_threshold=threshold
    )
    fig.update_layout(width=width, height=height, title=f"<b>{title}</b>")
    return fig


def net_kamada_kawai(
    net: nx.Graph,
    df_labels2,
    nclust: int = 3,
    threshold: float = 0.8,
    scale1: int = 400,
    scale2: int = 150,
    size1: int = 70,
    size2: int = 30,
) -> tuple[plt.Figure, str]:
    """Generate a Kamada-Kawai layout of a network and clusters of nodes.

    Parameters
    ----------
    net : nx.Graph
        Network to be visualized.
    df_labels2 : pd.DataFrame
        DataFrame containing the labels for each node in the network.
    nclust : int
        Number of clusters to be visualized.
    threshold : float
        Threshold for determining which edges to draw.
    scale1 : int
        Scale for the Kamada-Kawai layout of the full network.
    scale2 : int
        Scale for the Kamada-Kawai layout of each cluster.
    size1 : int
        Size of nodes in the full network.
    size2 : int
        Size of nodes in the clusters.

    Returns
    -------
    fig : plt.Figure
        The resulting plot.
    title : str
        The title of the plot.
    """

    def fpos(subnet: nx.Graph, scale: int = 150):
        """Generate Kamada-Kawai layout for a given subnetwork.

        Parameters
        ----------
        subnet : nx.Graph
            Subnetwork to be visualized.
        scale : int
            Scale for the Kamada-Kawai layout.

        Returns
        -------
        pos : dict
            Dictionary of positions for each node in the subnetwork.
        """
        return nx.kamada_kawai_layout(subnet, scale=scale)

    return cluster_netfig(
        net,
        df_labels2,
        fpos,
        nclust,
        threshold,
        "jacc_kamada",
        scale1,
        scale2,
        size1,
        size2,
    )


def net_circular(
    net: nx.Graph,
    df_labels2,
    nclust: int = 3,
    threshold: float = 0.8,
    scale1: int = 400,
    scale2: int = 150,
    size1: int = 70,
    size2: int = 30,
) -> tuple[plt.Figure, str]:
    """
    Plots a circular layout of a network, with clusters highlighted.
    
    Parameters
    ----------
    net: nx.Graph
        The network to be plotted.
    df_labels2 : pd.DataFrame
        DataFrame with cluster labels for each node.
    nclust : int
        Number of clusters to highlight.
    threshold : float
        Threshold for deciding which edges to draw.
    scale1 : int
        Scale for the layout of the whole network.
    scale2 : int
        Scale for the layout of each cluster.
    size1 : int
        Size of nodes in the whole network.
    size2 : int
        Size of nodes in each cluster.
        
    Returns
    -------
    fig : plt.Figure
        The resulting plot.
    title : str
        The title of the plot.
    """

    def fpos(subnet, scale: int = 150):
        return nx.circular_layout(subnet, scale=scale)

    return cluster_netfig(
        net,
        df_labels2,
        fpos,
        nclust,
        threshold,
        "jacc_circular",
        scale1,
        scale2,
        size1,
        size2,
    )


def net_spring(
    net: nx.Graph,
    df_labels2,
    nclust: int = 3,
    threshold: float = 0.8,
    scale1: int = 400,
    scale2: int = 150,
    size1: int = 70,
    size2: int = 30,
) -> tuple[plt.Figure, str]:
    """
    Plots a spring layout of a network, with clusters highlighted.
    
    Parameters
    ----------
    net: nx.Graph
        The network to be plotted.
    df_labels2 : pd.DataFrame
        DataFrame with cluster labels for each node.
    nclust : int
        Number of clusters to highlight.
    threshold : float
        Threshold for deciding which edges to draw.
    scale1 : int
        Scale for the layout of the whole network.
    scale2 : int
        Scale for the layout of each cluster.
    size1 : int
        Size of nodes in the whole network.
    size2 : int
        Size of nodes in each cluster.
        
    Returns
    -------
    fig : plt.Figure
        The resulting plot.
    title : str
        The title of the plot.
    """

    def fpos(subnet, scale=150):
        return nx.spring_layout(subnet, scale=scale)

    return cluster_netfig(
        net,
        df_labels2,
        fpos,
        nclust,
        threshold,
        "jacc_spring",
        scale1,
        scale2,
        size1,
        size2,
    )


def net_spectral(
    net: nx.Graph,
    df_labels2,
    nclust: int = 3,
    threshold: float = 0.8,
    scale1: int = 400,
    scale2: int = 150,
    size1: int = 70,
    size2: int = 30,
) -> tuple[plt.Figure, str]:
    """
    Plots a spectral layout of a network, with clusters highlighted.
    
    Parameters
    ----------
    net: nx.Graph
        The network to be plotted.
    df_labels2 : pd.DataFrame
        DataFrame with cluster labels for each node.
    nclust : int
        Number of clusters to highlight.
    threshold : float
        Threshold for deciding which edges to draw.
    scale1 : int
        Scale for the layout of the whole network.
    scale2 : int
        Scale for the layout of each cluster.
    size1 : int
        Size of nodes in the whole network.
    size2 : int
        Size of nodes in each cluster.
        
    Returns
    -------
    fig : plt.Figure
        The resulting plot.
    title : str
        The title of the plot.
    """

    def fpos(subnet, scale=150):
        return nx.spectral_layout(subnet, scale=scale)

    return cluster_netfig(
        net,
        df_labels2,
        fpos,
        nclust,
        threshold,
        "jacc_spectral",
        scale1,
        scale2,
        size1,
        size2,
    )


def net_shell(
    net: nx.Graph,
    df_labels2,
    nclust: int = 3,
    threshold: float = 0.8,
    scale1: int = 400,
    scale2: int = 150,
    size1: int = 70,
    size2: int = 30,
) -> tuple[plt.Figure, str]:
    """
    Plots a shell layout of a network, with clusters highlighted.
    
    Parameters
    ----------
    net: nx.Graph
        The network to be plotted.
    df_labels2 : pd.DataFrame
        DataFrame with cluster labels for each node.
    nclust : int
        Number of clusters to highlight.
    threshold : float
        Threshold for deciding which edges to draw.
    scale1 : int
        Scale for the layout of the whole network.
    scale2 : int
        Scale for the layout of each cluster.
    size1 : int
        Size of nodes in the whole network.
    size2 : int
        Size of nodes in each cluster.
        
    Returns
    -------
    fig : plt.Figure
        The resulting plot.
    title : str
        The title of the plot.
    """

    def fpos(subnet, scale=150):
        return nx.shell_layout(subnet, scale=scale)

    return cluster_netfig(
        net, df_labels2, fpos, nclust, threshold, "shell", scale1, scale2, size1, size2,
    )
