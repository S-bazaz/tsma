# -*- coding: utf-8 -*-
"""
These functions allow for the exploration of parameters through statistical analyses,
 with the option for automatic saving

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
import time as t
from tqdm import tqdm
import numpy as np
import scipy.stats as ss

#################
#  importations #
#################

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from tsma.basics.transfers import update_dct_from_list, update_dct_from_dct
from tsma.basics.text_management import encode, decode
from tsma.collect.output_management import get_save_path
from tsma.collect.iterators import nsim_mproc
from tsma.visuals.fig_management import save_transient, transient_default

##########################
#  exploration functions #
##########################


def get_nset(
    m_ref,
    nvar: int,
    nsim: int = 1,
    ndays: int = 0,
    nhours: int = 0,
    nmin: int = 0,
    ns: int = 10,
) -> int:
    """ Compute one or several standard simulations in order to get the maximum number of sets of parameters 
    that one can simulate in a given amount of time
    
    Parameters
    ----------
    m: Model,
        ex Gross2022
    nsim: int,
        number of simulations
    t_end : int, optional
        Total runtime of the model
    mode_nsim : bool,
        if the estimation is based on nsim simulations or only one
        
    ndays:int,
        number of days
    nhours:int,
        number of hours
    nmin:int,
        number of minutes
    ns:int,
        number of seconds
        
    Returns
    -------
    int:
        number of sets of parameters
    """

    tot_time = ((ndays * 24 + nhours) * 60 + nmin) * 60 + ns

    t_end = m_ref.hyper_parameters["t_end"]

    t1 = t.time()
    output = m_ref.simulate(True, True)
    t2 = t.time()
    dt = t2 - t1
    nvar = len(list(output))

    if nsim > 1:
        t3 = t.time()
        outputs = nsim_mproc(m_ref, nvar, t_end, nsim, save=True, overwrite=True)
        t4 = t.time()
        dt = t4 - t3
        return int(tot_time / dt)
    else:
        t1 = t.time()
        output = m_ref.simulate(True, True)
        t2 = t.time()
        dt = t2 - t1
        return int(tot_time / dt)


def mat_para(
    nset: int, lst_bounds: list[list[float]], scramble: bool = True
) -> np.array:
    """
    Compute a parameters exploration, according to a given number of sets that one wants to test,
    and ranges to explore for every parameters.
    
    Parameters
    ----------
    
    nset : int
        Number of sets of parameters to test.
    lst_bounds : list[list[float]]
        List of lists with the minimum and maximum values for each parameter.
    scramble : bool, optional
        Randomized Sobol sequence or deterministic Sobol sequence. Default is True (randomized).
        
    Returns
    -------
    np.array
        Matrix of parameters.
    """
    normalized_mat = ss.qmc.Sobol(len(lst_bounds), scramble=scramble, seed=None).random(
        nset
    )
    # print(normalized_mat)
    a = lst_bounds[:, 0]
    b = lst_bounds[:, 1]
    f = lambda x: a + (b - a) * x
    return f(normalized_mat)


####################
#  data collection #
####################


def oneset_collect(
    model,
    parameters: dict,
    initial_values: dict,
    hyper_parameters: dict,
    nsim: int,
    seeds: list = [],
    maxcore: int = -1,
    save: bool = True,
    overwrite: bool = False,
    save_fig: bool = True,
    fig_format: str = "png",
    dct_groups: dict = {},
    sign: float = 0.1,
    ncol: int = 3,
    nskip: int = 1,
    ncol_para: int = 3,
    f_save=save_transient,
    f_fig=transient_default,
) -> np.array:
    """ Simulate with a specific set of parameters, then compute a transient analysis and save the figure.
    
    Parameters
    ----------
    model : callable
        A model object, such as Gross2022 or threesector
    parameters : dict
        Dictionary of parameters of the model
    hyper_parameters : dict
        Dictionary of hyper-parameters related to the model
    initial_values : dict
        Dictionary of the models initial values
    nsim : int
        Number of simulations
    seeds : List[int], optional
        List of seeds for the simulations.
        Defaults to an empty list, in which case random seeds are generated.
    maxcore : int, optional
        Maximum number of cores to use for parallel simulations.
        Defaults to -1, which means use all available cores.
    save : bool, optional
        If True, save the data from the simulations.
        Defaults to True.
    overwrite : bool, optional
        If True, overwrite existing simulation data.
        Defaults to False.
    save_fig : bool, optional
        If True, save the figure from the transient analysis.
        Defaults to True.
    fig_format : str, optional
        Format of the figure to save. Can be "png", "pdf", or "jpeg".
        Defaults to "png".
    dct_groups : dict, optional
        Dictionary for grouping the curves in the transient analysis.
        Defaults to an empty dictionary.
    sign : float,
        significance of the test associated with a transient analysis for insteance
        ex: 0.1 -> 90% confidence interval
    ncol : int, optional
        Number of columns in the plot. Defaults to 3.
    nskip : int, optional
        Number of empty columns to skip at the beginning of the figure, by default 0.
    ncol_para : int, optional
        Number of columns in the parameter table. Defaults to 3.
    f_save : callable, optional
        Function for saving the figure
    f_fig : callable, optional
        Function for creating the figure
    Returns
    -------
    outputs : np.ndarray
        Outputs of the simulations
    """
    t_end = hyper_parameters["t_end"]
    m = model(parameters, hyper_parameters, initial_values)
    path_figures = os.sep.join([get_save_path(m), "figures"])

    output = m.simulate(save=True, overwrite=False, t_end=t_end)
    sim_id = m.sim_id
    varnames = list(output)
    nvar = len(varnames)

    outputs = nsim_mproc(
        m,
        nvar,
        t_end,
        nsim=nsim,
        seeds=seeds,
        save=save,
        overwrite=overwrite,
        maxcore=maxcore,
    ).copy()

    if save_fig:
        f_save(
            parameters,
            hyper_parameters,
            initial_values,
            outputs,
            varnames,
            dct_groups,
            sim_id,
            path_figures,
            fig_format,
            sign=sign,
            ncol=ncol,
            nskip=nskip,
            ncol_para=ncol_para,
            f_fig=f_fig,
        )
    return outputs


def para_exploration(
    model: type,
    m_ref,
    nvar: int,
    nsim: int,
    dct_bounds: dict[str, list[float]],
    ndays: int = 0,
    nhours: int = 0,
    nmin: int = 0,
    ns: int = 10,
    scramble: bool = True,
    seeds: list[int] = [],
    maxcore: int = -1,
    save: bool = True,
    overwrite: bool = False,
    save_fig: bool = True,
    fig_format: str = "png",
    dct_groups: dict[str, list[str]] = {},
    sign: float = 0.1,
    ncol: int = 3,
    nskip: int = 1,
    ncol_para: int = 3,
    f_save=save_transient,
    f_fig=transient_default,
) -> None:
    """One of the final functions of the data collection.
    It computes a parameter exploration, does the simulations, (saves the data)
    Then performs the transient analysis and saves the figure.
    
    Parameters
    ----------
    model: class object,
        example: Gross2022
    m_ref: model object,
        example: Gross2022()
        
    nsim: int,
        number of simulations
    dct_bounds: dict[str, List[float]],
        dictionary of the parameter bounds to explore
    ndays: int,
        number of days to simulate
    nhours: int,
        number of hours to simulate
    nmin: int,
        number of minutes to simulate
    ns: int,
        number of seconds to simulate
    scramble: bool,
        whether to use a randomized or deterministic Sobol sequence
    seeds: List[int],
        list of seed values to use for each simulation
    maxcore: int,
        maximum number of cores to use
    save: bool,
        whether to save the data from the simulations
    overwrite: bool,
        whether to overwrite existing data
    save_fig: bool,
        whether to save the figure
    fig_format: str,
        format to save the figure (e.g. "png", "pdf", "jpeg")
    dct_groups: Dict[str, List[str]],
        dictionary for grouping curves in the figure
    sign: float,
        the percentage of the range to include above and below the data in the figure
    ncol: int,
        number of columns for subplots in the figure
    nskip: int,
        number of rows to skip between subplots in the figure
    ncol_para: int,
        number of columns to use for displaying parameter values in the figure title
    f_save: Callable,
        function to use for saving the figure
    f_fig: Callable,
        function to use for generating the figure
        
    Returns
    -------
    None
    """
    nset = get_nset(
        m_ref, nvar, nsim=nsim, ndays=ndays, nhours=nhours, nmin=nmin, ns=ns,
    )

    lst_bounds = np.array(list(dct_bounds.values()))
    lst_para = mat_para(nset, lst_bounds, scramble=scramble)
    print("----------------------------------------------------------------------")
    print(f"number of sets of parameters {nset}")
    print("----------------------------------------------------------------------")

    params = encode(
        m_ref.parameters, m_ref.hyper_parameters, m_ref.initial_values, m_ref.agent_ids
    )
    for i in tqdm(range(nset)):

        dct_para = update_dct_from_list(dct_bounds, lst_para[i])
        parameters, hyper_parameters, initial_values = decode(
            update_dct_from_dct(params, dct_para), m_ref.agent_ids
        )

        outputs = oneset_collect(
            model,
            parameters,
            initial_values,
            hyper_parameters,
            nsim,
            seeds,
            maxcore,
            save,
            overwrite,
            save_fig,
            fig_format,
            dct_groups,
            sign,
            ncol,
            nskip,
            ncol_para,
            f_save,
            f_fig,
        )
