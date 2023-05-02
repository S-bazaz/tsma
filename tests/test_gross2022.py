# -*- coding: utf-8 -*-
"""
Unit test for gross2022 softwware

"""

__author__ = ["Samuel Bazaz"]
__credits__ = ["Samuel Bazaz"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Samuel Bazaz"]

##############
#  packages  #
##############

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from copy import deepcopy

#################
#  importations #
#################
from gross2022 import Gross2022 as model
from output_management_gross import (
    sub_dict,
    replace_for_path,
    encode_dct,
    encode,
    decode,
    get_save_path,
    list_to_tab_columns,
    dict_to_tab_columns,
    dict_to_values_command,
    matrix_to_values_command,
    dict_to_where_command,
    initialize_outputs,
    get_id_parameters,
    get_max_id,
    save_parameters,
    save_simulation,
    query_parameters,
    query_simulation,
)

from plotting_functions_gross import (
    fig_to_image,
    save_fig,
    read_from_html,
    add_trace_mos,
    add_traces_mos,
    init_subplot,
    fill_subplot,
    construct_mosaique_1sim,
    construct_mosaique_nsim,
    mosaique,
    mosaique_hist,
    mosaique_nsim,
    mosaique_hist_nsim,
    mosaique_transient,
    color_set,
    type_line,
    my_scatter_adaptmos,
    add_trace_adaptmos,
    add_traces_adaptmos,
    init_subplot_adapted,
    construct_adaptmos,
    adapted_mosaique,
    adapted_mosaique_transient,
    r2_for_polyreg,
    get_philips_okun_curves,
    intersection,
    transfert_x_lst_to_lst,
    transfert_List_to_List,
    separate_suffix,
    auto_groups,
    auto_mosaique,
)

from dash_functions_gross import (
    inverse_and_concate,
    position_sliders,
    add_index_and_exp,
    decompose_mainname,
    varname_to_latex,
    myslider2,
    myinput,
    mybutton,
    myradio,
    columns_of_sliders,
    columns_of_inputs,
    top_buttons,
    sidebar_container,
    html_struct_mixte_dbc,
    app_automosaique,
)

from iterators_gross import (
    n_simulate,
    add_simulate,
    aux_simulate,
    aux_zip_para,
    aux_pool_sim,
    n_sim_multiproc,
    auto_transient,
    get_relaxation_time,
    get_relaxation_time_agreg,
)

from statistical_functions_gross import (
    get_mean_and_var,
    get_batch_means,
    get_batch_means_agreg,
    iid_ci_width,
    rech_dycho_closest_inf,
    batch_test,
    batch_test_agreg,
    batch_test_agreg_pearson,
    kolmogorov_smirnov_test,
    transient_analysis,
    asymptotical_analysis,
)

from data_collect_gross import (
    get_nset_and_nvar,
    save_adapted_mos,
    sim_session,
    multi_para_session,
)

#######################
#  output_management  #
#######################


def test_sub_dict_1():
    """Create a sub dictionary with the given keys

    Parameters
    ----------
    original : dictionary,
        the initial dictionary
    keys : list,
        the list of keys which filters the dictionary
    inside : bool (default True),
        True to keep keys, False to retain all others

    Returns
    -------
    ... : dictionary,

    """
    assert sub_dict({"a": 1, "b": 2}, ["a"], inside=True) == {"a": 1}


def test_sub_dict_2():
    """Create a sub dictionary with the given keys

    Parameters
    ----------
    original : dictionary,
        the initial dictionary
    keys : list,
        the list of keys which filters the dictionary
    inside : bool (default True),
        True to keep keys, False to retain all others

    Returns
    -------
    ... : dictionary,

    """
    assert sub_dict({"a": 1, "b": 2}, ["a"], inside=False) == {"b": 2}


def test_replace_for_path():
    """Convert a string of instances the values of a dictionary 
    into a good format for creating paths.
    
    Parameters
    ----------
    string : string,
    Returns
    -------
    string : string,
    """
    assert replace_for_path("dict_values[array[dict_keys A, B],\n:C]") == "[A_B]_C"


def test_encode_dct():
    """ encode dictionary content with appropriate prefixes 
    to recreate the original: p {origin_dictionary}
    
    Parameters
    ----------
    origin : dict,
        dictionary to encode
    id_dict : string, 
        id of the originating dictionary (e.g. parameters, hyper_parameters,...)
    
    Returns
    -------
    target : dict, 
    """
    assert encode_dct({"a": 1, "b": 2}, "1") == {"1__a": 1.0, "1__b": 2.0}


def test_decode_encode_1():
    """ encode
    Convert parameters, hyper_parameters, initial_values into a
    single dictionary,the original dictionary in encoded in the prefix of the name.

    Parameters
    ----------
    parameters: dict,
    hyper_parameters: dict,
    initial_values: dict,
    
    Returns
    -------
    res : dict,
    """

    """ decode
    Convert an encodeed dictionary to the decoded initial dictionaries
    of parameters, hyper_parameters, initial_values
    
    Parameters
    ----------
    origin: dict,
        encoded dictionary
    sectors : list[str]
        sector indexes, ex: ["energy", "resources", "goods"]
        
    Return
    ------
    parameters: dict
    hyper_parameters: dict
    initial_values: dict
    """
    res = decode(encode({"a": 1, "b": 2}, {"c": 1, "d": 2}, {"e": 1, "f": 2}))

    assert res == ({"a": 1, "b": 2}, {"c": 1, "d": 2}, {"e": 1, "f": 2})


def test_decode_encode_2():
    """ encode
    Convert parameters, hyper_parameters, initial_values into a
    single dictionary,the original dictionary in encoded in the prefix of the name.

    Parameters
    ----------
    parameters: dict,
    hyper_parameters: dict,
    initial_values: dict,
    
    Returns
    -------
    res : dict,
    """

    """ decode
    Convert an encodeed dictionary to the decoded initial dictionaries
    of parameters, hyper_parameters, initial_values
    
    Parameters
    ----------
    origin: dict,
        encoded dictionary
    sectors : list[str]
        sector indexes, ex: ["energy", "resources", "goods"]
        
    Return
    ------
    parameters: dict
    hyper_parameters: dict
    initial_values: dict
    """
    res = decode(encode({"a": 1, "b": 2}, {"c": 1, "d": 2}, {"e": 1, "f": 2}))
    assert isinstance(res[0]["a"], float) and isinstance(res[1]["c"], int)


def test_get_save_path():
    """ Construct the path to the saved outputs
    
    Parameters
    ----------
    model: Model
    folder: str (default = 'outputs')
        
    Return
    ------
    save_path : string
    """
    m = model({}, {}, {})
    assert get_save_path(m).endswith("Gross2022\Outputs\gross2022")


def test_list_to_tab_columns():
    """ Gives the list of columns in the right format for table creation from a list
    
    Parameters
    ----------
    L : list string,
      name of the columns 
        
    id_name : string,
      name of the primary key
    
    Return
    ------
    res: string,
    
    """
    assert (
        list_to_tab_columns(["a", "b"], "prim_key")
        == "prim_key integer PRIMARY KEY , a float, b float"
    )


def test_dict_to_tab_columns():
    """ Lists columns in the proper format to create a table from a dictionary.
    
    Parameters
    ----------
    dct : dict,
      where keys are the names of the columns 
        
    primary_key : string,
      name of the primary key
    
    Return
    ------
    ...: string,
    
    """
    assert (
        dict_to_tab_columns({"a": 1, "b": 2}, "prim_key")
        == "prim_key integer PRIMARY KEY , a float, b float"
    )


def test_dict_to_values_command():
    """
    It's used to save one row of a database
    Create a VALUES block with the adapted syntax to store values from a dictionary.
    However this function can be used with a list as input.
    
    Parameters
    ----------
    dct : dict, or str list
        give the names of the variables stored
    
    Return
    ------
    res: string,
    
    """
    assert dict_to_values_command({"a": 1, "b": 2}) == "VALUES (  :a ,:b )"


def test_matrix_to_values_command():
    """ 
    It's used to save a table.
    Create a VALUES block with the adapted syntax to store values from a matrix.
    
    Parameters
    ----------
    ncol: dict,
        number of columns in the table 
    
    Return
    ------
    res: string
    """
    assert matrix_to_values_command(2) == " VALUES ( ?, ? )"


def test_dict_to_where_command():
    """ Create a WHERE block, in order to select the rows with conditions on values.
    those conditions are equality conditions where the values are given by dct.
    
    Parameters
    ----------
    dct : dict,
        dictionary for the values
    
    Return
    ------
    res: string,
    
    """
    assert dict_to_where_command({"a": 1, "b": 2}) == "WHERE ( ( a = :a) AND ( b = :b))"


def test_initialize_outputs():
    """ Create the files and the databases for the outputs of a new model.
    
    Parameters
    ----------
    parameters : dict,
        encoded dictionary use to initialize the columns of parameters.db
    outputcols : list[str],
        give the columns to initialize simulations.db
    model : Model,
        model object
        
    No Return
    ------
    """
    parameters = {
        "b_margin": 0.05,
        "hh_cons_propensity": 1,
        "norisk_interest": 0,
        "tgt_capital_ratio": 0.1,
        "smooth_interest": 15,
        "beta": 1,  # add to the model
    }
    hyper_parameters = {"f_n": 2, "hh_n": 3, "t_end": 2, "seed": 0}
    initial_values = {
        "wages_0": 1,
        "wages_1": 1,
        "prices_0": 1.01,
        "prices_1": 1,
    }

    m = model(parameters, hyper_parameters, initial_values)
    output = m.simulate(False, False)
    save_path = get_save_path(m)
    para_path = os.sep.join([save_path, "parameters.db"])
    sim_path = os.sep.join([save_path, "simulations.db"])
    initialize_outputs(parameters, list(output), m)
    return os.path.exists(para_path) and os.path.exists(sim_path)


def test_get_max_id():
    """ Give the highest id number given to a stored simulation
    
    Parameters
    ----------
    model : Model
        model object
        
    Return
    ------
    ...: int
        maximum id number of the parameter set
    """
    parameters = {
        "b_margin": 0.05,
        "hh_cons_propensity": 1,
        "norisk_interest": 0,
        "tgt_capital_ratio": 0.1,
        "smooth_interest": 15,
        "beta": 1,  # add to the model
    }
    hyper_parameters = {"f_n": 2, "hh_n": 3, "t_end": 2, "seed": 0}
    initial_values = {
        "wages_0": 1,
        "wages_1": 1,
        "prices_0": 1.01,
        "prices_1": 1,
    }

    m = model(parameters, hyper_parameters, initial_values)
    index = get_max_id(m)
    assert isinstance(index, int)


def test_save_and_query():
    """ get_id_parameters
    Give the id number the first simulation with certain parameters
    
    Parameters
    ----------
    parameters : dict
        encoded dictionary use to initialize the columns of parameters.db
    model : Model
        model object
        
    Return
    ------
    sim_id: string
        id of the parameter set
    """

    """ save_parameters
    Save a set of parameters in parameters.db and return the id of this new set
    
    Parameters
    ----------
    parameters : dict,
        encoded dictionary use to initialize the columns of parameters.db
    model : Model,
        model object
        
    Return
    ------
    sim_id: string,
        id of the simulation for the set of parameters
    """

    """ query_parameters
    Give the set of parameters in parameters.db for a given simulation id and model
    
    Parameters
    ----------
    sim_id : string,
        the id of the simulation
    model : model,
        to get the name 
    
    Return
    ------
    df: pd.Dataframe,
    
    """

    """ save_simulation
    Save a simulation in Simulations.db for a given model and id 
    
    Parameters
    ----------
    output : pd.DataFrame,
        result of the simulation 
    sim_id : string,
        the id of the simulation
    model : model,
        to get the name 
    """

    """query_simulation
    Give the output of a simulation for a given simulation id and model
    
    Parameters
    ----------
    sim_id : string,
        the id of the simulation
    model : Model,
        to get the name 
        
    Return
    ------
    df: pd.Dataframe,
    """
    parameters = {
        "b_margin": 0.05,
        "hh_cons_propensity": 1,
        "norisk_interest": 0,
        "tgt_capital_ratio": 0.1,
        "smooth_interest": 15,
        "beta": 1,  # add to the model
    }
    hyper_parameters = {"f_n": 2, "hh_n": 3, "t_end": 2, "seed": 0}
    initial_values = {
        "wages_0": 1,
        "wages_1": 1,
        "prices_0": 1.01,
        "prices_1": 1,
    }
    m = model(parameters, hyper_parameters, initial_values)
    params = encode(parameters, hyper_parameters, initial_values)
    output = m.simulate(False, False)
    index = get_id_parameters(params, m)
    if index is None:
        sim_id_para = str(save_parameters(params, m))
        sim_id_sim = "S" + sim_id_para
        save_simulation(output, sim_id_sim, m)
    else:
        sim_id_para = str(index[0])
        sim_id_sim = "S" + sim_id_para
    assert all(query_parameters(sim_id_para, m) == params) and all(
        query_simulation(sim_id_sim, m) == output
    )


########################
#  plotting functions  #
########################


def test_fig_to_image():
    """Save an image of a figure if the  image_format is right
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
    path = os.getcwd()
    output = pd.DataFrame({"a": [0, 1]})
    fig = mosaique(output, 3, "test_fig_to_image")
    fig_to_image(fig, path, "test_fig_to_image", "png")
    fig_to_image(fig, path, "test_fig_to_image", "jpeg")
    fig_to_image(fig, path, "test_fig_to_image", "pdf")
    path1 = os.sep.join([path, "test_fig_to_image.png"])
    path2 = os.sep.join([path, "test_fig_to_image.jpeg"])
    path3 = os.sep.join([path, "test_fig_to_image.pdf"])

    res = all([os.path.exists(path1), os.path.exists(path2), os.path.exists(path3)])
    os.remove(path1)
    os.remove(path2)
    os.remove(path3)
    assert res


def test_save_read_html():
    """save_fig
    Save an image if the  image_format is right and a html file of a figure
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
    """read_from_html
    Read an html file at filepath, for instance a figure

    Parameters
    ----------
    filepath : string,
        the path of the file
    Returns
    -------
    ... : html object,
        ex: plotly figure
    """
    path = os.getcwd()
    path_fig = os.sep.join([path, "test_save_read_html.html"])
    output = pd.DataFrame({"a": [0, 1]})
    fig = mosaique(output, 3, "test_save_read_html")
    save_fig(fig, path, "test_save_read_html", save_format="html")
    fig2 = read_from_html(path_fig)
    os.remove(path_fig)
    assert fig.data == fig2.data and fig.layout == fig2.layout


def test_add_trace_mos():
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

    go1 = go.Scatter(x=[0, 1], y=[1, 0], xaxis="x", yaxis="y")
    subplot = make_subplots(rows=1, cols=1)
    add_trace_mos(subplot, go1, 0, 0)
    assert subplot.data[0] == go1


def test_add_traces_mos():
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

    go1 = go.Scatter(x=[0, 1], y=[1, 0], xaxis="x", yaxis="y")
    go2 = go.Scatter(x=[0, 1], y=[1, 1], xaxis="x", yaxis="y")
    subplot = make_subplots(rows=1, cols=1)
    add_traces_mos(subplot, [go1, go2], 0, 0)
    assert subplot.data == (go1, go2)


def test_init_subplot():
    """Create a plotly subplot in the right format for a simple mosaique

    Parameters
    ----------
    nvar : int,
        number of variables
    ncol : int,
        number of columns
    varnames : str list, 
        names of the variables
        
    Returns
    -------
    ... : plotly figure
    """
    assert init_subplot(1, 1, ["test"]).layout.xaxis.domain == (0.0, 1.0)


def test_fill_subplot():
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
    go1 = go.Scatter(x=[0, 1], y=[1, 0], xaxis="x", yaxis="y")

    def add_plot(subplot, i, j, isim):
        add_trace_mos(subplot, go1, i, j)

    subplot = make_subplots(rows=1, cols=1)
    fill_subplot(1, 1, subplot, add_plot, 2)
    assert subplot.data == (go1, go1)


def test_construct_mosaique_1sim():
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
    output = pd.DataFrame({"a": [0, 1], "b": [1, 1]})
    go1 = go.Scatter(x=output.index, y=output.iloc[:, 0], xaxis="x", yaxis="y")
    go2 = go.Scatter(x=output.index, y=output.iloc[:, 1], xaxis="x2", yaxis="y2")
    lst_go_fig = [go1, go2]

    def add_plot(subplot, i, j, isim):
        add_trace_mos(subplot, lst_go_fig[i * 2 + j], i, j)

    subplot = construct_mosaique_1sim(output, 2, "test", add_plot)
    assert subplot.data == (go1, go2)


def test_construct_mosaique_nsim():
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
        
    Returns
    -------
    subplot : plotly figure
    """
    varnames = ["a", "b"]
    f_aux = lambda k: "" if k < 2 else "2"
    lst_go_fig = [
        go.Scatter(x=[0, 1], y=[0, k], xaxis="x" + f_aux(k), yaxis="y" + f_aux(k))
        for k in range(4)
    ]

    def add_plot(subplot, i, j, isim):
        add_trace_mos(subplot, lst_go_fig[4 * i + 2 * j + isim], i, j)

    subplot = construct_mosaique_nsim(2, 2, "test", varnames, add_plot, 2)
    assert subplot.data == tuple(lst_go_fig)


def test_mosaique():
    """Create a subplot of the outcomes with different colors.

    Parameters
    ----------
    output : pd.Dataframe,
        the output of a simulation
    ncol : int,
        number of columns in the mosaique
    title : string,
        give the title

    Returns
    -------
    subplot : plotly figure
    """
    path = os.getcwd()
    path_test = os.sep.join([path, "fig_unit_test"])
    path_fig = os.sep.join([path_test, "test_mosaique.html"])

    output = pd.DataFrame({"a": [0, 1], "b": [1, 1]})
    fig = mosaique(output, 3, "test_mosaique")

    # save_fig(fig, path_test, "test_mosaique", save_format="html")
    fig2 = read_from_html(path_fig)
    assert fig.data == fig2.data and fig.layout == fig2.layout


def test_mosaique_hist():
    """Create a subplot of histogrames of the outcomes with different colors

    Parameters
    ----------
    output : pd.Dataframe,
        the output of a simulation
    ncol : int,
        number of columns in the mosaique
    title : string,
        give the title

    Returns
    -------
    subplot : plotly figure
    """
    path = os.getcwd()
    path_test = os.sep.join([path, "fig_unit_test"])
    path_fig = os.sep.join([path_test, "test_mosaique_hist.html"])

    output = pd.DataFrame({"a": [0, 1], "b": [1, 1]})
    fig = mosaique_hist(output, 3, "test_mosaique_hist")

    # save_fig(fig, path_test, "test_mosaique_hist", save_format="html")
    fig2 = read_from_html(path_fig)
    assert fig.data == fig2.data and fig.layout == fig2.layout


def test_mosaique_nsim():
    """Create a subplot of the outcomes with different colors for multiple simulations

    Parameters
    ----------
    outputs : 3D array
        the output of the simulations
    ncol : int,
        number of columns in the mosaique
    title : string,
        give the title
    varnames : str list,

    Returns
    -------
    subplot : plotly figure
    """
    path = os.getcwd()
    path_test = os.sep.join([path, "fig_unit_test"])
    path_fig = os.sep.join([path_test, "test_mosaique_nsim.html"])

    outputs = np.arange(8).reshape((2, 2, 2))
    varnames = ["a", "b"]
    fig = mosaique_nsim(outputs, 2, "test", varnames)
    # fig.show()
    # save_fig(fig, path_test, "test_mosaique_nsim", save_format="html")
    fig2 = read_from_html(path_fig)
    assert fig.data == fig2.data and fig.layout == fig2.layout


def test_mosaique_hist_nsim():
    """Create a subplot of histogrames of the outcomes with different colors 
    for multiple simulations

    Parameters
    ----------
    outputs : 3D array
        the output of the simulations
    ncol : int,
        number of columns in the mosaique
    title : string,
        give the title
    varnames : str list,

    Returns
    -------
    subplot : plotly figure
    """
    path = os.getcwd()
    path_test = os.sep.join([path, "fig_unit_test"])
    path_fig = os.sep.join([path_test, "test_mosaique_hist_nsim.html"])

    outputs = np.arange(8).reshape((2, 2, 2))
    varnames = ["a", "b"]
    fig = mosaique_hist_nsim(outputs, 2, "test", varnames)
    # fig.show()
    # save_fig(fig, path_test, "test_mosaique_hist_nsim", save_format="html")
    fig2 = read_from_html(path_fig)
    return fig.data == fig2.data and fig.layout == fig2.layout


def test_mosaique_transient():
    """Create a subplot of transient analyses based on multiple simulations

    Parameters
    ----------
    outputs : 3D array
        the output of the simulations
    sign : float,
        the significance of the test
        ex 0.1-> 90% confidence interval
    ncol : int,
        number of columns in the mosaique
    title : string,
        give the title
    varnames : str list,

    Returns
    -------
    subplot : plotly figure
    """
    path = os.getcwd()
    path_test = os.sep.join([path, "fig_unit_test"])
    path_fig = os.sep.join([path_test, "test_mosaique_transient.html"])

    outputs = np.arange(8).reshape((2, 2, 2))
    varnames = ["a", "b"]
    fig = mosaique_transient(outputs, 1, 2, "test", varnames)
    # fig.show()
    # save_fig(fig, path_test, "test_mosaique_transient", save_format="html")
    fig2 = read_from_html(path_fig)
    return fig.data == fig2.data and fig.layout == fig2.layout


def test_color_set():
    """Associate a color to the variable for adapted_mosaique

    Parameters
    ----------
    name : string,
        name of the variable
    k : the number of the curve in the plot
    c: int,
        adjustment parameter

    Returns
    -------
    ... : string
    """
    c1 = color_set("unemployment_rate", 1, 0)
    c2 = color_set("x", 0, 0)
    c3 = color_set("x", 1, 0)
    c4 = color_set("x", 2, 0)
    assert all([c1 == "darkorchid", c2 == "green", c3 == "goldenrod", c4 == "purple"])


def test_type_line():
    """Associate a type of line to the variable for adapted_mosaique

    Parameters
    ----------
    name_col : string,
        name of the variable
    k : the number of the curve in the plot
    c: int,
        adjustment parameter

    Returns
    -------
    ... : string
    """
    t1 = type_line("x_tgt")
    t2 = type_line("x")
    assert all([t1 == "dashdot", t2 == "solid"])


def test_my_scatter_adaptmos():
    """Create a customized go.Scatter
    It's used int the adapted mosaiques

    Parameters
    ----------
    x : np.array,
        x axis
    y: np.array,
        y axis
     k: int,
        number of the variable in the group
    varname:str,
        name of the variable
    b_lower:bool,
        tell if the curve is the lowest in a confidence interval plot
    opa:float,
        opacity
    width:float,
        witdth of the line
    c:int,
        color parameter for automatic color choice
    Returns
    -------
    ... : plotly graph object
    """
    go1 = my_scatter_adaptmos([0, 1], [1, 1], 0, "test")
    go2 = go.Scatter(
        x=[0, 1],
        y=[1, 1],
        name="test",
        opacity=0.5,
        line={"color": "green", "dash": "solid", "shape": "linear", "width": 1.5},
    )
    assert go1 == go2


def test_add_trace_adaptmos():
    """add a figure to a subplot at the right position
    with the right y axis
    
    Parameters
    ----------
    subplot : plotly figure,
        subplot
    varname : str,
        name of the variable
    go_fig : plotly graph object,
        to add
    i : int,
         row-1
    j : int,
        col-1
    no Return
    ---------
    """
    go1 = go.Scatter(x=[0, 1], y=[1, 0], xaxis="x", yaxis="y")
    subplot = make_subplots(rows=1, cols=1)
    varname = "test"
    add_trace_adaptmos(subplot, varname, go1, 0, 0)
    assert subplot.data[0] == go1


def test_add_traces_adaptmos():
    """add figures to a subplot at the right position
    with the right y axis
    
    Parameters
    ----------
    subplot : plotly figure,
        subplot
    varname : str,
        name of the variable
    lst_go_fig : plotly graph object,
        to add
    i : int,
         row-1
    j : int,
        col-1
    no Return
    ---------
    """
    go1 = go.Scatter(x=[0, 1], y=[1, 0], xaxis="x", yaxis="y")
    go2 = go.Scatter(x=[0, 1], y=[1, 1], xaxis="x", yaxis="y")
    subplot = make_subplots(rows=1, cols=1)
    varname = "test"
    add_traces_adaptmos(subplot, varname, [go1, go2], 0, 0)
    assert subplot.data == (go1, go2)


def test_init_subplot_adapted():
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
    return init_subplot_adapted(1, ["test"], 1).layout.xaxis2.domain == (0.0, 0.94)


def test_construct_adaptmos():
    """Create an adapted mosaique based on a certain function add_plot

    Parameters
    ----------
    ncol : int,
        number of columns
    title : str,
        title of the mosaique
    groupnames : str list,
        names of the groups
    groups: str list list,
        liste of the groups of variables
    add_plot : function,
        add a certain plot to a subplot
    width:int,
        width of the figure
    height:int,
        height of the figure
    vspace : float,
        verticale space between plots
    hspace : float,
        horizontal space between plots
        
    Returns
    -------
    subplot : plotly figure
    """
    go1 = go.Scatter(x=[0, 1], y=[0, 1], xaxis="x", yaxis="y")
    go2 = go.Scatter(x=[0, 1], y=[1, 1], xaxis="x2", yaxis="y3")
    lst_go_fig = [go1, go2]

    def add_plot(subplot, i, j, isim):
        add_trace_mos(subplot, lst_go_fig[i * 2 + j], i, j)

    subplot = construct_adaptmos(2, "test", ["groupname"], [["a"], ["b"]], add_plot)
    assert subplot.data == (go1, go2)


def test_adapted_mosaique():
    """Create a subplot of the outcomes with different colors, by groups

    Parameters
    ----------
    output : dataframme,
        of the output of a simulation
    ncol : int,
        number of columns in the mosaique
    name : string,
        give the title
    groupnames : list string,
        name of the subplots
    groups : list list string,
        give how to group the plots
    Returns
    -------
    fig : go figure
    """
    path = os.getcwd()
    path_test = os.sep.join([path, "fig_unit_test"])
    path_fig = os.sep.join([path_test, "test_adapted_mosaique.html"])

    output = pd.DataFrame({"a": [0, 1], "b": [1, 1]})
    fig = adapted_mosaique(
        output, 3, "test_adapted_mosaique", ["a and b"], [["a", "b"]]
    )

    # save_fig(fig, path_test, "test_adapted_mosaique", save_format="html")
    fig2 = read_from_html(path_fig)
    assert fig.data == fig2.data and fig.layout == fig2.layout


def test_adapted_mosaique_transient():
    """Create a subplot of the outcomes with different colors, by groups
    Parameters
    ----------
    mean : np.array,
        means of the output of the simulations
    d : np.array,
        confidence interval width
    ncol : int,
        number of columns in the mosaique
    title : string,
        give the title
    varnames:list[str],
        names of the variables
    groupnames : list string,
        title of the subplots
    groups : list list string,
        give how to group the plots
    width:int,
        width of the figure
    height:int,
        height of the figure
    vspace : float,
        verticale space between plots
    hspace : float,
        horizontal space between plots
    Returns
    -------
    subplot : plotly figure
    """
    path = os.getcwd()
    path_test = os.sep.join([path, "fig_unit_test"])
    path_fig = os.sep.join([path_test, "test_adapted_mosaique_transient.html"])

    mean = np.arange(4).reshape((2, 2))
    d = np.arange(4).reshape((2, 2))

    fig = adapted_mosaique_transient(
        mean,
        d,
        2,
        "test_adapted_mosaique_transient",
        ["a", "b"],
        ["a and b"],
        [["a", "b"]],
    )
    fig2 = read_from_html(path_fig)
    # fig.show()
    # fig2.show()
    # save_fig(fig, path_test, "test_adapted_mosaique_transient", save_format="html")
    assert fig.data == fig2.data and fig.layout == fig2.layout


def test_r2_for_polyreg():
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
    df = pd.DataFrame({"a": [0, 1, 4, 9]})
    assert r2_for_polyreg(df.index, df[["a"]].values) == 1


def test_get_philips_okun_curves():
    """Create a subplot with :
        - Wage and Price Phillips curves, based on firms' micro data. 
        - Okun's Law with unemployment Rate and Change in unemployment Rate, based on macro data
    !!!! 
        The micro-data is not saved on databases,
        So it has to be calculated before using this function 
        To have r2 option, you would need to have f_n, hh_n, t_end (ex: 10, 10, 30)
    !!!!

    Parameters
    ----------
    m : Gross2020 model,

    newpath_fig : str,
        path to save figures
    save : bool,
        if you want to save
    image_format : str,
        format of the of the backup : png, jpeg, pdf
    Returns
    -------
    fig : matplotlib figure
    """
    parameters = {
        "b_margin": 0.05,
        "hh_cons_propensity": 1,
        "norisk_interest": 0,
        "tgt_capital_ratio": 0.1,
        "smooth_interest": 15,
        "beta": 1,  # add to the model
    }
    hyper_parameters = {"f_n": 30, "hh_n": 30, "t_end": 50, "seed": 0}
    initial_values = {
        "wages_0": 1,
        "wages_1": 1,
        "prices_0": 1.01,
        "prices_1": 1,
    }

    m = model(parameters, hyper_parameters, initial_values)
    output = m.simulate(False, False)

    path = os.getcwd()
    path_test = os.sep.join([path, "fig_unit_test"])
    path_fig = os.sep.join([path_test, "test_get_philips_okun_curves.png"])
    path_fig2 = os.sep.join([path_test, "test_get_philips_okun_curves2.png"])
    # fig = get_philips_okun_curves(
    #     m, path_test, "test_get_philips_okun_curves", save=True, image_format="png"
    # )
    fig2 = get_philips_okun_curves(
        m, path_test, "test_get_philips_okun_curves2", save=True, image_format="png"
    )

    im = plt.imread(path_fig)
    im2 = plt.imread(path_fig2)

    # d1 = ((im != im2).sum()/(im == im2).sum())
    d2 = 2 * np.linalg.norm(im2 - im) / (np.linalg.norm(im) + np.linalg.norm(im2))
    return d2 < 0.1


def test_intersection():
    """Give the intersection of two list in the order of the first one
    
    Parameters
    ----------
    lst1 : list,
        
    lst2 : list,
        
    Returns
    -------
    res : list,
    """
    assert intersection([1, 2, 3], [3, 2, 1, 5, 2]) == [1, 2, 3]


def test_transfert_x_lst_to_lst():
    """ Transfert all occurences of x from lst1 to lst2
    
    Parameters
    ----------
    lst1 : list,
        
    lst2 : list,
    
    x : ...,
        
    Returns
    -------
    ... : (list, list)
    """
    assert transfert_x_lst_to_lst([1, 2, 1], [3], 1) == ([2], [3, 1, 1])


def test_transfert_List_to_List():
    """Transfert all occurences of x in L, from lst1 to lst2
    
    Parameters
    ----------
    lst1 : list,
        
    lst2 : list,
    
    lst : list,
        
    Returns
    -------
    res : (list, list)
    """
    assert transfert_List_to_List([3, 2, 1, 5, 2], [1, 2, 3], [2, 1]) == (
        [3, 5],
        [1, 2, 3, 2, 2, 1],
    )


def test_separate_suffix():
    """Decompose a variable's name in two list prefix, suffix.
    The prefix is associated of the type of variable.
    The suffixis composed only of elements of lst_suffix.
    
    Parameters
    ----------
    varname : string,
        
    lst_suffix : list,
        ex: ["energy", "resources", "goods"]
        
    Returns
    -------
    prefix, suffix : (list, list)
    """
    assert separate_suffix("x_a_b", ["a", "b"]) == (["x"], ["a", "b"])


def test_auto_groups():
    """Add the varname to the right group based on its decomposition and original dimension.
    
    Parameters
    ----------
    varname : string,
        
    lst_suffix : list,
        ex: ["energy", "resources", "goods"]
    
    dct_dct : dict dict,
        for the grouping
        
    no Return
    ---------
    """
    return auto_groups(["x_a", "x_b", "y_a", "y_b"], ["a", "b"]) == (
        ["x", "y"],
        [["x_a", "x_b"], ["y_a", "y_b"]],
    )


def test_auto_mosaique():
    """Create an adapted mosaique with the auto grouping from auto_groups
    
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
    path = os.getcwd()
    path_test = os.sep.join([path, "fig_unit_test"])
    path_fig = os.sep.join([path_test, "test_auto_mosaique.html"])

    output = pd.DataFrame({"x_a": [0, 1], "x_b": [0, 2], "y_a": [1, 1], "y_b": [2, 2]})
    fig = auto_mosaique(output, ["a", "b"], "test_auto_mosaique")

    # save_fig(fig, path_test, "test_auto_mosaique", save_format="html")
    fig2 = read_from_html(path_fig)
    return fig.data == fig2.data and fig.layout == fig2.layout


###############
#  gross2022  #
###############


def test_build_for_initialization():
    """Build the objects used for the data of the simulation

    parameters
    ----------
    parameters : dict
        dictionary of the named parameters of the model
    hyper_parameters : dict,
        dictionary of hyper-parameters related to the model
    initial_values : dict,
        dictionary of the models initial values
    varnames : dict,
        dictionary of the variables of the model

    Returns
    -------
    struct_para: tuple,
        t_end : int,
            time_end of the simulation
        n_hh : int,
            number of households
        n_f : int,
            number of firms

    varnames : dict,


    hh_data : tuple,

        hh_cash : (t_end, n_hh) array,
            matrix of the cash owned by the houshold
        f_n_hh : (1, n_f) array,
            number of employee for a firm
        hh_id_f : (1, n_hh) array,
            id of the employer for an houshold

    nohh_data : tuple,
        firms : (t_end, n_f, nb_micro_var) array,
            storage for the micro-variables
        mainv : (t_end,nb_used_macro_variable)
            storage for the macro-variables used in the core of the simulation
        optv : (t_end,nb_optional_macro_variable)
            storage for the optional macro-variables
    """
    parameters = {
        "b_margin": 0.05,
        "hh_cons_propensity": 1,
        "norisk_interest": 0,
        "tgt_capital_ratio": 0.1,
        "smooth_interest": 15,
        "beta": 1,  # add to the model
    }
    hyper_parameters = {"f_n": 2, "hh_n": 3, "t_end": 1, "seed": 0}
    initial_values = {
        "wages_0": 1,
        "wages_1": 1,
        "prices_0": 1.01,
        "prices_1": 1,
    }
    varnames = {
        "firms": [
            "active",  # 0
            "inflation",  # 1
            "wage_inflation",  # 2
            "wage",  # 3
            "f_cash",  # 4
            "loan",  # 5
            "demand",  # 6
            "price",  # 7
            "interest_expense",  # 8
            "new_loan",  # 9
            "top_up",  # 10
            "f_net_worth",  # 11
            "f_capr",  # 12
        ],
        "macro_main": [
            "mean_wage",  # 0
            "wage_inflation",  # 1
            "mean_price",  # 2
            "inflation",  # 3
            "loans",  # 4
            "b_lossnet",  # 5
            "PD",  # 6 : probability of default
            "LGD",  # 7
            "interest",  # 8
        ],
        "macro_optional": ["a"],
    }
    (
        struct_para,
        varnames2,
        (hh_cash, f_n_hh, hh_id_f),
        (firms, mainv, optv),
    ) = model._build_for_initialization(
        parameters, hyper_parameters, initial_values, varnames
    )

    assert all(
        [
            struct_para == (1, 3, 2),
            varnames2 == varnames,
            (hh_cash == np.zeros(3)).all(),
            all(f_n_hh == np.array([2, 1])),
            all(hh_id_f == np.array([0, 1, 0])),
            (
                firms
                == np.array(
                    [
                        [
                            [1, 0, 0, 1, 0, 0, 0, 1.01, 0, 0, 0, 0, 1],
                            [1, 0, 0, 1, 0, 0, 0, 1.01, 0, 0, 0, 0, 1],
                        ]
                    ]
                )
            ).all(),
            (mainv == np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0.05]])).all(),
            (optv == np.array([[0]])).all(),
        ]
    )


def test_initialization():
    """define what are the variables of the simulation,
    and initialize the object for the data of the simulation
    
    parameters
    ----------
    parameters : dict
        dictionary of the named parameters of the model
    hyper_parameters : dict,
        dictionary of hyper-parameters related to the model
    initial_values : dict,
        dictionary of the models initial values
        
    Returns
    -------
    struct_para: tuple,
        t_end : int,
            time_end of the simulation
        n_hh : int,
            number of households
        n_f : int,
            number of firms
        
    varnames : dict,
        names of variables
        
    hh_data : tuple,
        hh_cash : (t_end, n_hh) array,
            matrix of the cash owned by the houshold
        f_n_hh : (1, n_f) array,
            number of employee for a firm
        hh_id_f : (1, n_hh) array,
            id of the employer for an household
    
    nohh_data : tuple,
        firms : (t_end, n_f, nb_micro_var) array,
            storage for the micro-variables
        mainv : (t_end,nb_used_macro_variable)
            storage for the macro-variables used in the core of the simulation
        optv : (t_end,nb_optional_macro_variable)
            storage for the optional macro-variables
    """
    parameters = {
        "b_margin": 0.05,
        "hh_cons_propensity": 1,
        "norisk_interest": 0,
        "tgt_capital_ratio": 0.1,
        "smooth_interest": 15,
        "beta": 1,  # add to the model
    }
    hyper_parameters = {"f_n": 2, "hh_n": 3, "t_end": 1, "seed": 0}
    initial_values = {
        "wages_0": 1,
        "wages_1": 1,
        "prices_0": 1.01,
        "prices_1": 1,
    }
    (
        struct_para,
        varnames,
        (hh_cash, f_n_hh, hh_id_f),
        (firms, mainv, optv),
    ) = model._initialization(parameters, hyper_parameters, initial_values)
    assert all(
        [
            struct_para == (1, 3, 2),
            len(varnames["firms"]) + len(varnames["macro_main"]) == 29,
            (hh_cash == np.zeros(3)).all(),
            all(f_n_hh == np.array([2, 1])),
            all(hh_id_f == np.array([0, 1, 0])),
            (
                firms
                == np.array(
                    [
                        [
                            [1, 0, 0, 1, 0, 0, 0, 1.01, 0, 0, 0, 0, 1],
                            [1, 0, 0, 1, 0, 0, 0, 1.01, 0, 0, 0, 0, 1],
                        ]
                    ]
                )
            ).all(),
            (
                mainv == np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0, 0, 0]])
            ).all(),
            (optv == np.array([[0]])).all(),
        ]
    )


def test_optional_var_update():
    """Update the aggregated optional variables.
    All the variables defined here can be deleted
    without disturbing the simulation

    parameters
    ----------

    t: int,
        step of the simulation
    n_f : int,
        number of firms
    data : tuple,
        all the data of the simulation
        (hh_data, nohh_data)
    lst_active:  int list,
        tells if its an active firm

    count_stats: tuple,
        nb bankrupt,
        nb rolling,
        nb paid_annuity

    no Return
    -------
    """
    firms = np.ones((1, 1, 20))
    mainv = np.ones((1, 30))
    optv = np.zeros((1, 50))
    n_f = 1
    lst_active = np.array([True])
    f_n_hh = np.array([2])
    hh_id_f = np.array([0, 0])
    hh_cash = np.ones((1, 2))

    data = ((hh_cash, f_n_hh, hh_id_f), (firms, mainv, optv))
    count_stats = (1, 2, 3)
    model._update_optional_var(
        0, n_f, data, lst_active, count_stats,
    )
    assert (
        np.nan_to_num(optv)
        == [
            [
                0,
                1,
                1,
                2,
                3,
                1,
                0,
                1,
                0,
                2,
                -1,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                2,
                2,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ]
    ).all()


def test_update_wage():
    """
    definitions
    -new-active : inactive at t-1 and active at t
    -elder-active : active at t-1 and at t
    
    function steps
    -Define the new-active firms and the other active (elder-active)
    -Calculate the wage of elder-active based on inflation
    -Calculate the wage for new-active firms as the average of elder-active

    !!! Inflation is seen as a fraction and is not the log of it contrary to the article!!!
    
    parameters
    ----------
    t: int,
        step of the simulation
    mainv : (t_end,nb_used_macro_variable)
        storage for the macro-variables used in the core of the simulation
    firms : (t_end, n_f, nb_micro_var) array,
        storage for the micro-variables
    beta : float in (0,1),
        variable added to the simulation for more control of the wage inflation
        
    no Return
    -------
    """
    firms = np.ones((3, 1, 4))
    mainv = 0.5 * np.ones((3, 16))
    model._update_wage(2, mainv, firms)
    assert all(
        [
            (firms == np.array([[[1, 1, 1, 1]], [[1, 1, 1, 1]], [[1, 1, 0, 1]]])).all(),
            (mainv == 0.5 * np.ones((3, 16))).all(),
        ]
    )


def test_update_interest():
    """
    Calculate the default probability and LGD, then the interest rate
    
    definitions
    -LGD : Losses Given Default

    parameters
    ----------
    t: int,
        step of the simulation
    mainv : (t_end,nb_used_macro_variable)
        storage for the macro-variables used in the core of the simulation
    firms : (t_end, n_f, nb_micro_var) array,
        storage for the micro-variables
    smooth_interest : int,
        time considered for mean average
    no Return
    -------
    """
    firms = np.ones((3, 1, 4))
    mainv = 0.5 * np.ones((3, 16))
    mainv2 = mainv.copy()
    # 6 : PD # 7 : LGD # 8 : interest
    mainv2[2, 6:9] = 1
    model._update_interest(2, mainv, firms)
    assert all([(firms == np.ones((3, 1, 4))).all(), (mainv == mainv2).all()])


def test_update_wage_inflation():
    """
    update price mean and wage inflation
    
    parameters
    ----------
    t: int,
        step of the simulation
    mainv : (t_end,nb_used_macro_variable)
        storage for the macro-variables used in the core of the simulation
    firms : (t_end, n_f, nb_micro_var) array,
        storage for the micro-variables
    is_active: np.array,
        tells if its an active firm
        
    no Return
    -------
    """

    firms = np.ones((3, 1, 4))
    mainv = 0.5 * np.ones((3, 16))
    is_active = np.array([True])
    model._update_wage_inflation(2, mainv, firms, is_active)
    mainv2 = mainv.copy()
    # 0 : mean_wage
    mainv2[2, 0] = 1
    # 1 : wage_inflation
    mainv2[2, 1] = 1
    assert all([(firms == np.ones((3, 1, 4))).all(), (mainv == mainv2).all()])


def test_wage_payement_and_newloans():
    """
    Firms ask for new loans if they don't have enough cash to pay wages
    Then they pay households.
    
    function steps
    -Calculate the new_loan 
    -Update firms' loan
    -Update firms cash for the payement of wages
    -Update HH's cash with their salary
    
    parameters
    ----------
    t: int,
        step of the simulation
    firms : (t_end, n_f, nb_micro_var) array,
        storage for the micro-variables
    hh_data : tuple,
        (hh_cash, hh_id_f, f_n_hh)
    is_active : np.array,
        tells if its an active firm
        
    no Return
    -------
    """
    firms = np.ones((2, 1, 10))
    hh_id_f = np.array([0, 0])
    firms2 = firms.copy()
    # 4 : f_cash
    firms2[1, :, 4] = 0
    # 5 : loan
    firms2[1, :, 5] = 2
    is_active = np.array([True])
    f_n_hh = np.array([2])
    hh_cash = np.zeros((2, 2))

    hh_data = (hh_cash, f_n_hh, hh_id_f)

    model._wage_payement_and_newloans(1, firms, hh_data, is_active)
    assert all(
        [(firms == firms2).all(), (hh_cash == np.array([[0, 0], [1, 1]])).all(),]
    )


def test_aux_hh_choose():
    """Intermediate function to compute the consumptions decision of the houshold i.
    It fills the row i of the matrix aux_mat, where the columns are associated to the firms.

    Parameters
    ----------
    i: int,
        houshold index
    aux_mat : (n_hh, n_f) array
        matrix for the hh consumption decision calculation
    hh_choices : np.array,
        give the firms choosen by the housholds
    hh_cash_t : np.array,
        give the current cash that the houshold can used to consume
    
    no Return
    ---------
    """
    aux_mat = np.zeros((1, 2))
    hh_choices = np.array([1])
    hh_cash_t = np.array([5])
    model._aux_hh_choose(0, aux_mat, hh_choices, hh_cash_t)
    assert (aux_mat == np.array([[0, 5]])).all()


def test_update_hh_demand():
    """Compute the demand of consumptions of households for th firms

    Parameters
    ----------
    t: int,
        step of the simulation
    n_hh : int, 
        number of households
    n_f : int, 
        number of firms
    firms : (t_end, n_f, nb_micro_var) array,
        storage for the micro-variables
    hh_cash : (t_end, n_hh) array,
        matrix of the cash owned by the houshold
    is_active : np.array,
        tells if its an active firm
    alpha : float,
        households' consumption propension
    no Return
    ---------
    """
    n_f = 2
    firms = np.zeros((2, 2, 7))
    is_active = np.array([True, False])
    n_hh = 3

    hh_cash = 2 * np.ones((2, 3))
    alpha = 0.5
    firms2 = firms
    # 6 : demand
    firms2[0, 0, 6] = 3
    model._update_hh_demand(0, n_hh, n_f, firms, hh_cash, is_active, alpha)
    # 6 : demand
    assert (firms == firms2).all()


def test_update_prices_and_inflations():
    """Firms adapt their prices to the demand

    Parameters
    ----------
    t: int,
        step of the simulation
    mainv : (t_end,nb_used_macro_variable)
        storage for the macro-variables used in the core of the simulation
    firms : (t_end, n_f, nb_micro_var) array,
        storage for the micro-variables
    hh_cash : (t_end, n_hh) array,
        matrix of the cash owned by the houshold
    f_n_hh : (1, n_f) array,
        number of employee for a firm
    alpha : float,
        households' consumption propension

    no Return
    ---------
    """
    firms = np.zeros((3, 1, 8))
    f_n_hh = np.array([3])
    # 6 : demand
    firms[2, 0, 6] = 3
    hh_cash = 2 * np.ones((3, 3))
    alpha = 0.5
    mainv = np.ones((3, 4))

    firms2 = firms.copy()
    # 7 : price
    firms2[2, 0, 7] = 1
    # 4 : f_cash
    firms2[2, 0, 4] = 3
    hh_cash2 = hh_cash.copy()
    hh_cash2[2, :] = 1

    model._update_prices_and_inflations(2, mainv, firms, hh_cash, f_n_hh, alpha)
    assert all(
        [
            (firms == firms2).all(),
            (
                hh_cash == hh_cash2
            ).all(),  # (hh_cash == np.array([[0, 0], [1, 1]])).all(),
        ]
    )


def test_update_loans_and_capr():
    """update the current total loans of the firms, the bank's networth and its capital ratio

    Parameters
    ----------
    t: int,
        step of the simulation
    mainv : (t_end,nb_used_macro_variable)
        storage for the macro-variables used in the core of the simulation
    firms : (t_end, n_f, nb_micro_var) array,
        storage for the micro-variables
    hh_cash : (t_end, n_hh) array,
        matrix of the cash owned by the houshold
        
    no Return
    ---------
    """
    firms = np.zeros((1, 1, 8))
    # 5 : loan
    firms[0, 0, 5] = 10
    # 4 : f_cash
    firms[0, 0, 4] = 1
    mainv = np.zeros((1, 11))
    hh_cash = 2 * np.ones((1, 2))

    firms2 = firms.copy()
    mainv2 = mainv.copy()
    # 4 : loans
    mainv2[0, 4] = 10
    # 9 : b_networth
    mainv2[0, 9] = 5
    # 10 : capital_ratio
    mainv2[0, 10] = 0.5
    hh_cash2 = hh_cash.copy()

    model._update_loans_and_capr(0, mainv, firms, hh_cash)
    assert all(
        [(firms == firms2).all(), (mainv == mainv2).all(), (hh_cash == hh_cash2).all()]
    )


def test_dividends_payements():
    """The bank distribute dividends based on its current capital ratio and the capital ratio targeted
    The Bank's networth is not updated directly. Indeed, the Bank's networth is only updated for the capital ratio calculation,
    thanks to the stock flow consistency of the balance sheets
    
    Parameters
    ----------
    t: int,
        step of the simulation
    n_hh : int, 
        number of households
    mainv : (t_end,nb_used_macro_variable)
        storage for the macro-variables used in the core of the simulation
    hh_cash : (t_end, n_hh) array,
        matrix of the cash owned by the houshold
    tgt_capr : float,
        Bank's capital ratio targeted
        
    no Return
    ---------
    """
    n_hh = 2
    mainv = np.ones((1, 12))
    tgt_capr = 0.5
    hh_cash = np.zeros((1, 2))

    mainv2 = mainv.copy()
    # 11 : dividends
    mainv2[0, 11] = 0.5
    hh_cash2 = hh_cash.copy()
    hh_cash2 = 0.25
    model._dividends_payements(0, n_hh, mainv, hh_cash, tgt_capr)
    assert all([(mainv == mainv2).all(), (hh_cash == hh_cash2).all()])


def test_annuity_payement_and_newloans():
    """
    Firms calculate the interest they have to pay.
    Based on it, they are in bankruptcy, 
    they roll the rest of their debt or they pay all their debts.
    
    definition
    top_up : new loans created to roll the debt
    
    function steps
    -Calculate interest_expense, the interest they have to pay
    -Calculate the stats boolean arrays : is_bankrupt, is_rolling...
    -Update the activity of the firms for the next step
    -Update firms' loan and cash, and banks' networths and losses, according to the situation.
        ! firms' loan and cash is updated after for bankrupt,
        in order to have the capital ratio calculation !
    
    parameters
    ----------
    t: int,
        step of the simulation
    t_end : int,
        time_end of the simulation
    mainv : (t_end,nb_used_macro_variable)
        storage for the macro-variables used in the core of the simulation
    firms : (t_end, n_f, nb_micro_var) array,
        storage for the micro-variables
    is_active:  int list,
        tells if its an active firm

    Returns
    -------
    count_stats: tuple,
        nb bankrupt,
        nb rolling,
        nb paid_annuity
    is_rolling : np.array,
        tells if the firm is rolling is debt
    is_no_bankrupt : np.array,
        tells if the firm doesn't go bankrupt
    """
    firms = np.ones((2, 2, 13))
    mainv = np.ones((2, 14))
    lst_active = np.array([True, False])
    firms2 = firms.copy()
    # 4 : f_cash
    firms2[1, 0, 4] = 0
    mainv2 = mainv.copy()
    # 5 : b_lossnet  # 13 : b_lossgross
    mainv2[1, [5, 13]] = 0
    count_stats, is_rolling, is_no_bankrupt = model._annuity_payement_and_newloans(
        1, 1, mainv, firms, lst_active
    )
    assert all(
        [
            count_stats == (0, 1, 0),
            (is_rolling == lst_active).all(),
            (is_no_bankrupt == lst_active).all(),
            (firms == firms2).all(),
            (mainv == mainv2).all(),
        ]
    )


def test_update_fcapr_interest_expense():
    """Update firms' networth and capital ratio, 
    then compute the mean of the capital ratios and the sum of interest expenses

    Parameters
    ----------
    t: int,
        step of the simulation
    mainv : (t_end,nb_used_macro_variable)
        storage for the macro-variables used in the core of the simulation
    firms : (t_end, n_f, nb_micro_var) array,
        storage for the micro-variables
    optv : (t_end,nb_optional_macro_variable)
        storage for the optional macro-variables
    is_rolling : np.array,
        tells if the firm is rolling is debt
    is_no_bankrupt : np.array,
        tells if the firm doesn't go bankrupt
        
    no Return
    ---------
    """
    is_rolling = np.array([True, False])
    is_no_bankrupt = np.array([True, False])
    firms = np.ones((1, 2, 13))
    # 4 : f_cash
    firms[0, is_no_bankrupt, 4] = 3
    mainv = np.ones((1, 13))
    optv = np.zeros((1, 27))
    firms2 = firms.copy()
    # 11 : f_net_worth # 12: f_capr
    firms2[0, is_no_bankrupt, [11, 12]] = 2

    mainv2 = mainv.copy()
    # 12 interest_expenses
    mainv2[0, 12] = 2

    optv2 = optv.copy()
    # 26 : mean_f_capr
    optv2[0, 26] = 2
    model._update_fcapr_interest_expense(
        0, mainv, firms, optv, is_rolling, is_no_bankrupt
    )
    assert all(
        [(firms == firms2).all(), (mainv == mainv2).all(), (optv == optv2).all()]
    )


def test_creat_output_and_output_micro():
    """Reformat the data to create two dataframes, the outputs of the simulation
    
    Parameters
    ----------
    t_end: int,
        number of steps of the simulation
    mainv : (t_end,nb_used_macro_variable)
        storage for the macro-variables used in the core of the simulation
    firms : (t_end, n_f, nb_micro_var) array,
        storage for the micro-variables
    optv : (t_end,nb_optional_macro_variable)
        storage for the optional macro-variables
    varnames : dict,
        names of variables
        
    Return
    ------
    output : pd.DataFrame,
        time-series of the model simulation
        
    output_micro (not returned) : pd.DataFrame,
        time-series of unemployements ,and firm's wages and prices
    """
    t_end = 2
    firms = np.arange(6).reshape((2, 1, 3))
    mainv = np.ones((2, 2))
    mainv[:, 1] = 2
    optv = np.zeros((2, 2))
    varnames = {"macro_main": ["a", "unemployment_rate"], "macro_optional": ["b", "c"]}
    output, output_micro = model._creat_output_and_output_micro(
        2, 1, mainv, firms, optv, varnames
    )
    output2 = pd.DataFrame(
        {"a": [1, 1], "unemployment_rate": [2, 2], "b": [0, 0], "c": [0, 0],}
    )
    output_micro2 = pd.DataFrame(
        {
            "t": [0, 1],
            "inflation": [1, 4],
            "wage_inflation": [2, 5],
            "unemployment_rate": [2, 2],
        }
    )

    assert all([all(output == output2), all(output_micro == output_micro2)])


def test_fist_steps():
    """Execute the first two steps of the simulation
    
    parameters
    ----------
    t_end : int,
        time_end of the simulation
    n_hh : int,
        number of households
    n_f : int,
        number of firms
    hh_data : tuple,
        hh_cash : (t_end, n_hh) array,
            matrix of the cash owned by the houshold
        f_n_hh : (1, n_f) array,
            number of employee for a firm
        hh_id_f : (1, n_hh) array,
            id of the employer for an houshold
    nohh_data : tuple,
        firms : (t_end, n_f, nb_micro_var) array,
            storage for the micro-variables
        mainv : (t_end,nb_used_macro_variable)
            storage for the macro-variables used in the core of the simulation
        optv : (t_end,nb_optional_macro_variable)
            storage for the optional macro-variables
    alpha : float,
        households' consumption propension
    tgt_capr : float,
        Bank's capital ratio targeted
    Returns
    -------
    struct_para: tuple,
        
        
    varnames : dict,
        names of variables
        
    hh_data : tuple,
        hh_cash : (t_end, n_hh) array,
            matrix of the cash owned by the houshold
        f_n_hh : (1, n_f) array,
            number of employee for a firm
        hh_id_f : (1, n_hh) array,
            id of the employer for an houshold
    
    nohh_data : tuple,
        firms : (t_end, n_f, nb_micro_var) array,
            storage for the micro-variables
        mainv : (t_end,nb_used_macro_variable)
            storage for the macro-variables used in the core of the simulation
        optv : (t_end,nb_optional_macro_variable)
            storage for the optional macro-variables
    """
    parameters = {
        "b_margin": 0,
        "hh_cons_propensity": 1,
        "norisk_interest": 0,
        "tgt_capital_ratio": 0,
        "smooth_interest": 15,
        "beta": 1,  # add to the model
    }
    hyper_parameters = {"f_n": 1, "hh_n": 2, "t_end": 3, "seed": 0}
    initial_values = {
        "wages_0": 1,
        "wages_1": 1,
        "prices_0": 1,
        "prices_1": 1,
    }
    ((t_end, n_hh, n_f), varnames, hh_data, nohh_data,) = model._initialization(
        parameters, hyper_parameters, initial_values
    )
    hh_data2 = deepcopy(hh_data)
    nohh_data2 = deepcopy(nohh_data)
    nohh_data2[0][0, 0, [6, 9]] = 2
    nohh_data2[0][1, 0, [6, 9]] = 2
    nohh_data2[1][0, 2] = 1
    nohh_data2[1][1, [0, 2]] = 1
    nohh_data2[1][[0, 1], 4] = 2

    nohh_data2[2][0, [1, 4, 5, 6, 7, 8, 9, 11, 13, 16, 18, 30, 31, 32]] = np.array(
        [1, 1, 2, 0, 1, 0, 2, 1, 0, 1, 2, 2, 1, 0]
    )

    nohh_data2[2][1, [1, 4, 5, 6, 7, 8, 9, 11, 13, 16, 18, 30, 31]] = np.array(
        [1, 1, 2, 1, 1, 0, 2, 1, 0, 1, 2, 2.0, 1.0]
    )
    model._fist_steps(t_end, n_hh, n_f, hh_data, nohh_data, 1, 0)
    assert all(
        [
            (hh_data[0] == hh_data2[0]).all(),
            (hh_data[1] == hh_data2[1]).all(),
            (hh_data[2] == hh_data2[2]).all(),
            (nohh_data[0] == nohh_data2[0]).all(),
            (nohh_data[1] == nohh_data2[1]).all(),
            (np.nan_to_num(nohh_data[2]) == nohh_data2[2]).all(),
        ]
    )


def test_run_simulation():
    """Run the actual simulation of the model and give a dataframe as output.

    no Parameter
    ----------
    Return
    -------
    output : pd.DataFrame
        time-series of the model simulation
    output_micro (not returned) : pd.DataFrame
        time-series of unemployements ,and firm's wages and prices
    """

    parameters = {
        "b_margin": 0,
        "hh_cons_propensity": 1,
        "norisk_interest": 0,
        "tgt_capital_ratio": 0,
        "smooth_interest": 15,
        "beta": 1,  # add to the model
    }
    hyper_parameters = {"f_n": 1, "hh_n": 2, "t_end": 3, "seed": 0}
    initial_values = {
        "wages_0": 1,
        "wages_1": 1,
        "prices_0": 1,
        "prices_1": 1,
    }
    m = model(parameters, hyper_parameters, initial_values)
    output = m._run_simulation()
    micro_output = m.output_micro
    mat1 = np.array(
        [
            [
                0,
                0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0,
                1,
                0,
                0,
                1,
                2,
                0,
                1,
                0,
                2,
                0,
                1,
                0,
                0,
                0,
                0.0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                1.0,
                0,
                0.0,
            ],
            [
                1,
                0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0,
                1,
                0,
                0,
                1,
                2,
                1,
                1,
                0,
                2,
                0,
                1,
                0,
                0,
                0,
                0.0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                1.0,
                0,
                0.0,
            ],
            [
                1,
                0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0,
                1,
                0,
                0,
                1,
                2,
                1,
                1,
                0,
                2,
                0,
                1,
                0,
                0,
                0,
                0.0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                1.0,
                0,
                0.0,
            ],
        ]
    )

    mat2 = np.zeros(np.shape(micro_output))
    mat2[:, 0] = np.arange(3)

    assert all(
        [
            (np.nan_to_num(output.values) == mat1).all(),
            (np.nan_to_num(micro_output.values) == mat2).all(),
        ]
    )

###################
#  dash function  #
###################


def test_inverse_and_concate():
    """ inverse then concatenate a list of list
    
    Parameters
    ----------
    lst_lst : list list,
        
    Returns
    -------
    lst_res : list,
    
    """
    assert inverse_and_concate([[1, 2], [3, 4]]) == [3, 4, 1, 2]


def test_position_sliders():
    """ Inverse then concatenate a list of list
    
    Parameters
    ----------
    params : dict,
        all parameters in undimensional form
    ncol : int, 
        number of columns of sliders 
    ndim : int,
        number of dimensionally different parameters
    ndico :int, 
        number of initial dictionaries for parameters
        
    Returns
    -------
    ... : list list,
        list of blocks which are list of Columns
    """
    params = {
        "p1_a": 0,
        "p1_b": 0,
        "p1_c": 0,
        "p2_a": 0,
        "p2_b": 0,
        "p2_c": 0,
        "p3_a": 0,
        "p3_b": 0,
    }

    assert position_sliders(params, 2, 3) == [
        [["p1_a", "p1_c"], ["p1_b"]],
        [["p2_a", "p2_c"], ["p2_b"]],
        [["p3_a"], ["p3_b"]],
    ]


def test_add_index_and_exp():
    """ Add to a latex formula an index and an exponent
    
    Parameters
    ----------
    letter : string,
        initial latex name
    index : string, 
        index to add
    exponent : string,
        exponent to add
        
    Returns
    -------
    res : string,
        
    """
    assert add_index_and_exp("l", "i", "e") == "$l_{i}^{e}$"


def test_decompose_mainname():
    """ Decompose the main name into the core name and the exponent
    ex : cons_hh -> (cons, hh)
    
    Parameters
    ----------
    mainname : string,
        initial name

    Returns
    -------
    ...: string,
        
    """
    core, exponent = decompose_mainname("hh_id_b_d_tgt")
    assert core == "id d" and exponent == "hh,b,tgt"


def test_varname_to_latex():
    """ Convert the name of a parameter into the right latex formula  
    !!! To adapt the conversion, please change:
        dct_conv,
        special_exp,
        special_index                             !!!
    
    Parameters
    ----------
    var : string,
        initial name

    Returns
    -------
    ...: string,
        
    """
    r1 = varname_to_latex("p1__hh_cons_propensity")
    r2 = varname_to_latex("p3__wages_0")
    r3 = varname_to_latex("p1__tgt_capital_ratio")
    assert all(
        [
            r1 == "$\\alpha_{}^{hh}$",
            r2 == "$w_{0 }^{}$",
            r3 == "$\\mathcal{CAR}_{}^{tgt}$",
        ]
    )


def test_myslider2():
    """ Create a dcc.Slider in a standard format for dashboard construction
    
    Parameters
    ----------
    name : string,
        name of the parameter associated with the Slider
    minv : float,
        minimum value of the Slider
    maxv : float,
        maximum value of the Slider
    n : int,
        number of steps 
    v0 : float,
        initial value of the Slider 
        
    Returns
    -------
    slider : dcc.Slider html_object
    
    """
    slider = myslider2("test", 1, 2, 4, 4)

    assert all(
        [
            slider.min == 1,
            slider.max == 2,
            slider.step == 0.25,
            slider.marks == {1: "1", 0.5: "0.5", 2: "2"},
            slider.tooltip == {"always_visible": True, "placement": "bottomLeft"},
            slider.id == {"type": "slider", "index": "test"},
        ]
    )


def test_myinput():
    """ Create a dbc.Input in a standard format for dashboard construction
    Parameters
    ----------
    name : string,
        name of the parameter associated with the Slider
    
    v0 : float,
        initial value of the Slider 
        
    Returns
    -------
    ... : dbc.Input html_object
    
    """
    inp = myinput("test", 1)
    assert all(
        [
            inp.id == {"type": "input", "index": "test"},
            inp.className == "mb-4 bg-light text-center",
            inp.placeholder == "input number",
            inp.size == "sm",
            inp.type == "number",
            inp.value == 1,
        ]
    )


def test_mybutton():
    """ Create a dbc.Button in a standard format for dashboard construction
    Parameters
    ----------
    name : string,
        name of the parameter associated with the Slider
        
    Returns
    -------
    ... : dbc.Button html_object
    
    """
    # return mybutton("test")
    button = mybutton("test")
    assert all(
        [
            button.children == "test",
            button.id == "test",
            button.className == "me-1",
            button.color == "secondary",
            button.outline is True,
            button.size == "sm",
        ]
    )


def test_myradio():
    """ Create a dbc.RadioItems with it's label in a standard format for dashboard construction
    Parameters
    ----------
    name : string,
        name of the parameter associated with the Slider
    lst_values : list string, 
        list of the possible values of the radioitems
        
    Returns
    -------
    ... : dbc.RadioItems html_object
    
    """
    rad = myradio("test", [1, 2, 3]).children[1]
    assert all(
        [
            rad.id == "test",
            rad.inline is True,
            rad.inputCheckedClassName == "border border-success bg-success",
            rad.labelCheckedClassName == "text-success",
            rad.options
            == [
                {"label": 1, "value": 1},
                {"label": 2, "value": 2},
                {"label": 3, "value": 3},
            ],
            rad.value == 1,
        ]
    )


def test_columns_of_sliders():
    """ Create a list of html object in order to build a column of Sliders with labels on the top 
                               
    Parameters
    ----------
    lst_var : list,
        list of the variables in the column
    params : dict,
        all parameters in undimensional form
    parameters_Sliders : dict,
        parameters of the Slidders { var : [min, max, n] }
        
    Returns
    -------
    lst_res : list html_object, 
    """
    lst_var = ["p1__x"]
    params = {"p1__x": 0}
    params_sliders = {"p1__x": [0, 1, 1]}
    col = columns_of_sliders(lst_var, params, params_sliders)
    label = col[0]
    slider = col[1]
    assert all(
        [
            label.mathjax is True,
            label.children == "$x_{}^{}$",
            label.style == {"text-align": "center", "margin-top": "20px"},
            slider.min == 0,
            slider.max == 1,
            slider.step == 1,
            slider.marks == {0: "0", 0.5: "0.5", 1: "1"},
            slider.value == 0,
            slider.tooltip == {"always_visible": True, "placement": "bottomLeft"},
            slider.id == {"type": "slider", "index": "p1__x"},
        ]
    )


def test_columns_of_inputs():
    """ Create a list of html object in order to build a column of inputs with labels on the top

    Parameters
    ----------
    lst_var : list,
        list of the variables in the column
    params : dict,
        all parameters in undimensional form
        
    Returns
    -------
    lst_res : list html_object, 
    """
    lst_var = ["p1__x"]
    params = {"p1__x": 0}
    col = columns_of_inputs(lst_var, params)
    label = col[0]
    ipt = col[1]
    assert all(
        [
            label.mathjax is True,
            label.children == "$x_{}^{}$",
            label.style == {"text-align": "center", "margin-top": "20px"},
            ipt.id == {"type": "input", "index": "p1__x"},
            ipt.className == "mb-4 bg-light text-center",
            ipt.placeholder == "input number",
            ipt.size == "sm",
            ipt.type == "number",
            ipt.value == 0,
        ]
    )


def test_top_buttons():
    """ Create a block of buttons
                               
    Parameters
    ----------
    lst_name : list,
        list of the names of the buttons
        
    Returns
    -------
    ... : Div html_object, 
    """
    lst_name = ["p1__x"]
    button = top_buttons(lst_name).children[0]
    assert all(
        [
            button.children == "p1__x",
            button.id == "p1__x",
            button.className == "me-1",
            button.color == "secondary",
            button.outline is True,
            button.size == "sm",
        ]
    )


def test_sidebar_container():
    """ Create a container of the controls of the parameters, based on a function func_col
    ex: func_col(col) = columns_of_sliders(col, params, params_sliders)
                               
    Parameters
    ----------
    params : dict,
        all parameters in undimensional form
    dct_id :,
        give the name of the initial dictionary ,
        ex { "1": "Parameters", "2": "Hyper_Parameters", "3":"Initial Values"}
    ncol : int,
        number of columns of sliders
    func_col : funct,
        ex: col -> columns_of_sliders(col, params, params_sliders)
    id_container : str,
        ex: slider-container
    Returns
    -------
    ... : Div html_object, 
    """
    params = {"p1__x": 0, "p2__y": 0}
    dct_id = {"1": "Parameters", "2": "hyper_parameters"}
    ncol = 1
    func_col = lambda col: columns_of_inputs(col, params)
    id_container = "inputs-container"
    div = sidebar_container(params, dct_id, ncol, func_col, id_container)
    assert all([len(div.children) == 6, div.id == "inputs-container",])


def test_html_struct_mixte_dbc():
    """ Final function of the dashboard layout build. 
    It monitors the application's global html structure.
    
    Parameters
    ----------
    params : dict,
        all parameters in undimensional form
    params_sliders : dict,
        parameters of the Sliders { var : [min, max, n] }
    ncol : int,
        number of columns of sliders
    title : string, 
        title of the app 
        
    Returns
    -------
    div_res :  html.Div() html_object
    
    """
    params = {"p1__x": 0}
    params_sliders = {"p1__x": [0, 1, 1]}
    ncol = 1
    title = "test"
    cont = html_struct_mixte_dbc(params, params_sliders, ncol, title)
    row1 = cont.children[0]
    row2 = cont.children[1]
    assert all(
        [
            len(cont.children) == 2,
            cont.fluid is True,
            len(row1.children) == 1,
            len(row2.children) == 2,
            row2.align == "start",
        ]
    )


def test_app_automosaique():
    """ Final function to create the application.
    It controls the plots and the update system.
    
    Parameters
    ----------
    
    parameters: dict,
    
    hyper_parameters: dict,
    
    initial_values: dict,
    
    params_sliders : dict,
        parameters of the Sliders { var : [min, max, n] }
        
    model : model object,
        ex: threesector()
    
    Returns
    -------
    app : dash object
    
    """
    parameters = {
        "b_margin": 0.05,
        "hh_cons_propensity": 1,
        "norisk_interest": 0,
        "tgt_capital_ratio": 0.1,
        "smooth_interest": 15,
        "beta": 1,  # add to the model
    }
    hyper_parameters = {"f_n": 2, "hh_n": 3, "t_end": 2, "seed": 0}
    initial_values = {
        "wages_0": 1,
        "wages_1": 1,
        "prices_0": 1.01,
        "prices_1": 1,
    }
    params_sliders = {
        "p1__b_margin": [0, 1, 1],
        "p1__hh_cons_propensity": [0, 1, 1],
        "p1__norisk_interest": [0, 1, 1],
        "p1__tgt_capital_ratio": [0, 1, 1],
        "p1__smooth_interest": [0, 1, 1],
        "p1__beta": [0, 1, 1],
        "p2__f_n": [0, 1, 1],
        "p2__hh_n": [0, 1, 1],
        "p2__t_end": [0, 1, 1],
        "p2__seed": [0, 1, 1],
        "p3__wages_0": [0, 1, 1],
        "p3__wages_1": [0, 1, 1],
        "p3__prices_0": [0, 1, 1],
        "p3__prices_1": [0, 1, 1],
    }
    dct_groups = {"test": ["p1__x"]}

    app = app_automosaique(
        parameters, hyper_parameters, initial_values, params_sliders, model, dct_groups,
    )
    conf = app.config
    assert all(
        [
            conf["name"] == "dash_functions_gross",
            conf["assets_url_path"] == "assets",
            conf["title"] == "Dash",
            conf["serve_locally"] is True,
            conf["meta_tags"]
            == [{"name": "viewport", "content": "with=device_with initial-scale=1.0"}],
            len(app.callback_map) == 2,
            len(app.layout.children) == 2,
            app.layout.fluid is True,
            app.title == "Dash",
        ]
    )
    
#################################
#  statistical_functions_gross  #
#################################

def test_get_mean_and_var():
    """ Give means and variances from outputs 
    
    Parameters
    ----------
    outputs : (t_end, nvar, nsim) array,
        gathers all the simulations' output

    Returns
    -------
    mean : (t_end, nvar) array,
    var : (t_end, nvar) array,
    """
    outputs = np.arange(4).reshape((1,2,2))
    mean, var = get_mean_and_var(outputs)
    assert all(
        [
            (mean == np.array([[0.5, 2.5]])).all(),
            (var == np.array([[0.25, 0.25]])).all()
        ]
    )
    
def test_get_batch_means():
    """ Give means and variances from outputs 
    
    Parameters
    ----------
    mainv0 : (t_end, nb_used_macro_variable)
        storage for the macro-variables used in the core of the simulation
    bs:int,
        number of batch
        
    Returns
    -------
    res : (bs, nb_used_macro_variable) array,
    """
    output0 = pd.DataFrame({"a":[1,2,3,4], "b":[2,4,6,8]})
    res = get_batch_means(output0, 2)
    assert (res == np.array([[1.5, 3. ],[3.5, 7. ]])).all()

def test_get_batch_means_agreg():
    """ Give means and variances from outputs 
    
    Parameters
    ----------
    mainv0 : (t_end, nb_used_macro_variable)
        storage for the macro-variables used in the core of the simulation
    bs:int,
        number of batch
        
    Returns
    -------
    res : (bs, nb_used_macro_variable) array,
    """
    outputs0 = np.arange(16).reshape((4,2,2))
    res = get_batch_means_agreg(outputs0, 2)
    assert (res == np.array([[2.5 , 4.5],[10.5, 12.5]])).all() 

def test_iid_ci_width():
    """ Give confidence interval width of significance sign
    
    Parameters
    ----------
    var : array
        variances
    nsim :int,
        number of simulations
    sign :significance of the t_test associated
        ex: 0.1 -> 90% confidence interval
        
    Returns
    -------
    ... :  array,
    """
    res = iid_ci_width(np.array([1,4]),2,0.1)
    assert (np.round(res,2)==[ 8.93, 17.86]).all()
    
def test_rech_dycho_closest_inf():
    """ Give the index of the closest inferior value from lst to x
    Or the closest value if x is not in the range of the list
    
    Parameters
    ----------
    x : float,
        value to approxiate
    lst : list,
        list of reference
        
    Returns
    -------
    g :  int,
        index in lst
    """
    lst = [1, 2.5, 4]
    i1 = rech_dycho_closest_inf(0,lst)
    i2 = rech_dycho_closest_inf(1,lst)
    i3 = rech_dycho_closest_inf(2.4,lst)
    i4 = rech_dycho_closest_inf(2.5,lst)
    i5 = rech_dycho_closest_inf(2.6,lst)
    i6 = rech_dycho_closest_inf(5,lst)
    assert [i1,i2,i3,i4,i5,i6] == [0, 0, 0, 1, 1, 2]

# print(test_rech_dycho_closest_inf())
# batch_test
# batch_test_agreg
# batch_test_agreg_pearson
# kolmogorov_smirnov_test
# transient_analysis
# asymptotical_analysis

#####################
#  iterators_gross  #
#####################

def test_n_simulate():
    """ Give nsim simulations from the model m
    
    Parameters
    ----------
    m: Model,
        ex Gross2022
    nsim: int,
        number of simulations
    lst_seeds: list,
        list of seeds, for personalized generations
    overwrite: bool, optional
        if you want to replace existing data by this simulation
    save : bool, optional
        if you want to save the data
    t_end : int, optional
        Total runtime of the model

    Returns
    -------
    outputs : (t_end, nvar, nsim2) array,
        gathers all the simulations' output
    
    """
    parameters = {
        "b_margin": 0,
        "hh_cons_propensity": 1,
        "norisk_interest": 0,
        "tgt_capital_ratio": 0,
        "smooth_interest": 15,
        "beta": 1,  # add to the model
    }
    hyper_parameters = {"f_n": 1, "hh_n": 2, "t_end": 3, "seed": 0}
    initial_values = {
        "wages_0": 1,
        "wages_1": 1,
        "prices_0": 1,
        "prices_1": 1,
    }
    m = model(parameters, hyper_parameters, initial_values)
    outputs = n_simulate(m,2)
    mat1 = np.array(
        [
            [
                0,
                0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0,
                1,
                0,
                0,
                1,
                2,
                0,
                1,
                0,
                2,
                0,
                1,
                0,
                0,
                0,
                0.0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                1.0,
                0,
                0.0,
            ],
            [
                1,
                0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0,
                1,
                0,
                0,
                1,
                2,
                1,
                1,
                0,
                2,
                0,
                1,
                0,
                0,
                0,
                0.0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                1.0,
                0,
                0.0,
            ],
            [
                1,
                0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0,
                1,
                0,
                0,
                1,
                2,
                1,
                1,
                0,
                2,
                0,
                1,
                0,
                0,
                0,
                0.0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                1.0,
                0,
                0.0,
            ],
        ]
    )

    assert all(
        [
            (np.nan_to_num(outputs)[:,:,0] == mat1).all(),
            (np.nan_to_num(outputs)[:,:,1] == mat1).all()
        ]
    )
    
def test_add_simulation():
    """ Add a simulation to an existing ouputs from the model m
    
    Parameters
    ----------
    m: Model,
        ex Gross2022
    outputs0 : (t_end, nvar, nsim) array,
        gathers all the simulations' output
    seed : int,
        seed for the generation of the simulation
    overwrite: bool, optional
        if you want to replace existing data by this simulation
    save : bool, optional
        if you want to save the data
    t_end : int, optional
        Total runtime of the model

    Returns
    -------
    outputs : (t_end, nvar, nsim+1) array,
        gathers all the simulations' output
    
    """
    parameters = {
        "b_margin": 0,
        "hh_cons_propensity": 1,
        "norisk_interest": 0,
        "tgt_capital_ratio": 0,
        "smooth_interest": 15,
        "beta": 1,  # add to the model
    }
    hyper_parameters = {"f_n": 1, "hh_n": 2, "t_end": 3, "seed": 0}
    initial_values = {
        "wages_0": 1,
        "wages_1": 1,
        "prices_0": 1,
        "prices_1": 1,
    }
    m = model(parameters, hyper_parameters, initial_values)
    outputs0 = n_simulate(m)
    outputs = add_simulate(m,outputs0,1)
    mat1 = np.array(
        [
            [
                0,
                0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0,
                1,
                0,
                0,
                1,
                2,
                0,
                1,
                0,
                2,
                0,
                1,
                0,
                0,
                0,
                0.0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                1.0,
                0,
                0.0,
            ],
            [
                1,
                0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0,
                1,
                0,
                0,
                1,
                2,
                1,
                1,
                0,
                2,
                0,
                1,
                0,
                0,
                0,
                0.0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                1.0,
                0,
                0.0,
            ],
            [
                1,
                0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0,
                1,
                0,
                0,
                1,
                2,
                1,
                1,
                0,
                2,
                0,
                1,
                0,
                0,
                0,
                0.0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                1.0,
                0,
                0.0,
            ],
        ]
    )

    assert all(
        [
            (np.nan_to_num(outputs)[:,:,0] == mat1).all(),
            (np.nan_to_num(outputs)[:,:,1] == mat1).all()
        ]
    )
    
def test_aux_simulate():
    """ return a simulation output according to an args tuple
        This form of function is useful the Pool instance and parallelization 
        
    Parameters
    ----------
    args tuple,
        m: Model,
            ex Gross2022
        overwrite: bool, optional
            if you want to replace existing data by this simulation
        save : bool, optional
            if you want to save the data
        sim_id : int,
            if you want to change the saving id of the simulation 
            (useful against IntegrityError)
        t_end : int, optional
            Total runtime of the model
        s : int,
            seed for the generation of the simulation

    Returns
    -------
    output : pd.DataFrame
        simulation result
    """
    parameters = {
        "b_margin": 0,
        "hh_cons_propensity": 1,
        "norisk_interest": 0,
        "tgt_capital_ratio": 0,
        "smooth_interest": 15,
        "beta": 1,  # add to the model
    }
    hyper_parameters = {"f_n": 1, "hh_n": 2, "t_end": 3, "seed": 0}
    initial_values = {
        "wages_0": 1,
        "wages_1": 1,
        "prices_0": 1,
        "prices_1": 1,
    }
    m = model(parameters, hyper_parameters, initial_values)
    args = (m,False,False,1,3,0)
    output = aux_simulate(args)
    mat1 = np.array(
        [
            [
                0,
                0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0,
                1,
                0,
                0,
                1,
                2,
                0,
                1,
                0,
                2,
                0,
                1,
                0,
                0,
                0,
                0.0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                1.0,
                0,
                0.0,
            ],
            [
                1,
                0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0,
                1,
                0,
                0,
                1,
                2,
                1,
                1,
                0,
                2,
                0,
                1,
                0,
                0,
                0,
                0.0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                1.0,
                0,
                0.0,
            ],
            [
                1,
                0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                0,
                1,
                0,
                0,
                1,
                2,
                1,
                1,
                0,
                2,
                0,
                1,
                0,
                0,
                0,
                0.0,
                1,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                1.0,
                0,
                0.0,
            ],
        ]
    )

    assert all(
        [  
            (np.nan_to_num(output) == mat1).all(),
        ]
    )

def test_aux_zip_para():
    """ Create a zip of args tuples to iterate the previous simulation function
        It also manage the sim_ids in order to avoid 
        the IntegrityError caused by simultaneous savings
        
    Parameters
    ----------
    fix_args tuple,
        m: Model,
            ex Gross2022
        overwrite: bool, optional
            if you want to replace existing data by this simulation
        save : bool, optional
            if you want to save the data
        t_end : int, optional
            Total runtime of the 
            
    seeds : np.array,
        seeds for the generation of the simulation

    Returns
    -------
    ... : zip object
    args tuples for simulations
    """
    parameters = {
        "b_margin": 0,
        "hh_cons_propensity": 1,
        "norisk_interest": 0,
        "tgt_capital_ratio": 0,
        "smooth_interest": 15,
        "beta": 1,  # add to the model
    }
    hyper_parameters = {"f_n": 1, "hh_n": 2, "t_end": 3, "seed": 0}
    initial_values = {
        "wages_0": 1,
        "wages_1": 1,
        "prices_0": 1,
        "prices_1": 1,
    }
    m = model(parameters, hyper_parameters, initial_values)
    fix_args = (m,False,False,3)
    seeds = np.arange(2)
    args1, args2 = aux_zip_para(fix_args, seeds)
    assert all(
        [  
            args1[0] == m,
            args2[0] == m,
            sum(args1[1:2])+sum(args2[1:2]) == 0,
            args1[4] == 3,
            args2[4] == 3,
            args1[5] == 0,
            args2[5] == 1,
        ]
    )
    
def test_aux_pool_sim():
    """ Create a Pool of simulations and returns the outputs if the simulations passed
        
    Parameters
    ----------
    n_cores : int,
        number of cpu cores used
    zip_para : zip tuples,
        arguments of the simulations in zipped format

    Returns
    -------
    ... : (t_end, nvar, n_cores) array,
        gathers all the simulations' output
    """
    
    if __name__ == '__main__':
        parameters = {
            "b_margin": 0,
            "hh_cons_propensity": 1,
            "norisk_interest": 0,
            "tgt_capital_ratio": 0,
            "smooth_interest": 15,
            "beta": 1,  # add to the model
        }
        hyper_parameters = {"f_n": 1, "hh_n": 2, "t_end": 3, "seed": 0}
        initial_values = {
            "wages_0": 1,
            "wages_1": 1,
            "prices_0": 1,
            "prices_1": 1,
        }
        m = model(parameters, hyper_parameters, initial_values)
        fix_args = (m,False,False,3)
        seeds = np.arange(2)
        zip_para = aux_zip_para(fix_args, seeds)
        outputs = aux_pool_sim(2,zip_para, (2, 3, 50) )
        mat1 = np.array(
            [
                [
                    0,
                    0,
                    1,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    2,
                    0,
                    1,
                    0,
                    2,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0.0,
                    1,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    1.0,
                    0,
                    0.0,
                ],
                [
                    1,
                    0,
                    1,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    2,
                    1,
                    1,
                    0,
                    2,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0.0,
                    1,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    1.0,
                    0,
                    0.0,
                ],
                [
                    1,
                    0,
                    1,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    2,
                    1,
                    1,
                    0,
                    2,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0.0,
                    1,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    1.0,
                    0,
                    0.0,
                ],
            ]
        )
        assert all(
            [
                (np.nan_to_num(outputs)[:,:,0] == mat1).all(),
                (np.nan_to_num(outputs)[:,:,1] == mat1).all()
            ]
        )
def test_n_sim_multiproc():
    """ Give nsim simulations from the model m
    
    Parameters
    ----------
    m: Model,
        ex Gross2022
    nsim: int,
        number of simulations
    lst_seeds: list,
        list of seeds, for personalized generations
    overwrite: bool, optional
        if you want to replace existing data by this simulation
    save : bool, optional
        if you want to save the data
    t_end : int, optional
        Total runtime of the model

    Returns
    -------
    outputs : (t_end, nvar, nsim2) array,
        gathers all the simulations' output
    
    """
    if __name__ == '__main__':
        parameters = {
            "b_margin": 0,
            "hh_cons_propensity": 1,
            "norisk_interest": 0,
            "tgt_capital_ratio": 0,
            "smooth_interest": 15,
            "beta": 1,  # add to the model
        }
        hyper_parameters = {"f_n": 1, "hh_n": 2, "t_end": 3, "seed": 0}
        initial_values = {
            "wages_0": 1,
            "wages_1": 1,
            "prices_0": 1,
            "prices_1": 1,
        }
        m = model(parameters, hyper_parameters, initial_values)
        outputs = n_sim_multiproc(m,2, t_end = 3, nvar = 50 )
        mat1 = np.array(
            [
                [
                    0,
                    0,
                    1,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    2,
                    0,
                    1,
                    0,
                    2,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0.0,
                    1,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    1.0,
                    0,
                    0.0,
                ],
                [
                    1,
                    0,
                    1,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    2,
                    1,
                    1,
                    0,
                    2,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0.0,
                    1,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    1.0,
                    0,
                    0.0,
                ],
                [
                    1,
                    0,
                    1,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    2,
                    1,
                    1,
                    0,
                    2,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0.0,
                    1,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    1.0,
                    0,
                    0.0,
                ],
            ]
        )
    
        assert all(
            [
                (np.nan_to_num(outputs)[:,:,0] == mat1).all(),
                (np.nan_to_num(outputs)[:,:,1] == mat1).all()
            ]
        )

# print(test_n_sim_multiproc())
    

    # auto_transient,
    # get_relaxation_time,
    # get_relaxation_time_agreg,

# print(test_app_automosaique())

########################
#  data_collect_gross  #
########################

