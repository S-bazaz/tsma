# -*- coding: utf-8 -*-
"""
implementation of statistical analyses taken from : Vandin & al (2021)
http://arxiv.org/abs/2102.05405
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
import numpy as np

#################
#  importations #
#################

path = os.path.dirname(os.getcwd())
sys.path.append(path)

from analyses.statistics import get_mean_and_var, iid_ci_width
from collect.iterators import nsimulate, add_simulate, nsim_mproc

#######################
# transient analysis  #
#######################

def transient_analysis(outputs:np.array, sign : float)->(np.array, np.array):
    """
    Do the transient analysis from several simulations
    
    Parameters
    ----------
    mainv : pd.DataFrame,
        results of a simulation
    sign : float,
        significance of the t_test associated
        ex: 0.1 -> 90% confidence interval
        
    Returns
    -------
    mean : (t_end, nvar) array,
        means
    d : (t_end, nvar) array,
        confidence interval width
    """
    mean, var = get_mean_and_var(outputs)
    d = iid_ci_width(var, len(outputs[0][0]), sign)
    return mean, d

def auto_transient(
    m,
    sign: float,
    delta: float,
    overwrite: bool = False,
    save: bool = False,
    t_end: int = None,
) -> (np.array, np.array, np.array):
    """ Add simulations until the confidence interval width is below delta times the mean
    
    Parameters
    ----------
    m: Model,
        ex Gross2022
    sign : float,
        significance of the t_test associated
        ex: 0.1 -> 90% confidence interval
    delta : float,
        threshold for the confidence interval width ratio
    overwrite: bool, optional
        if you want to replace existing data by this simulation
    save : bool, optional
        if you want to save the data
    t_end : int, optional
        Total runtime of the model

    Returns
    -------
    ... : bool np.array
        tell if the threshold is respected (the programme can stop before)
    mean : (t_end, nvar) array,
        means
    d : (t_end, nvar) array,
        confidence interval widths
    outputs : (t_end, nvar, nsim2) array,
        gathers all the simulations' output
    """
    # initialization
    nsim = 3
    outputs = nsimulate(m, nsim, overwrite=overwrite, save=save, t_end=t_end)
    mean, d = transient_analysis(outputs, sign)

    lst_pos = mean != 0
    while (d[lst_pos] / np.abs(mean[lst_pos]) > delta).any() and nsim < 1000:
        nsim += 1
        outputs = add_simulate(m, outputs, seed=nsim)
        mean, d = transient_analysis(outputs, sign)
        lst_pos = mean != 0
        print(f"max ratio {(d[lst_pos] / mean[lst_pos]).max()}, max width :{d.max()}")
    return (d[lst_pos] / np.abs(mean[lst_pos]) > delta), mean, d, outputs

####################
# relaxation time  #
####################

def get_relaxation_time(
    m,
    lst_thrs: list[float],
    f_test,
    lst_vars: list[int],
    b: int = 4,
    bs0: int = 4,
    n_batch: int = 64,
    max_count: int = 12,
) -> (bool, int, np.array):
    """ Give the relaxation time of a unique simulation, if it existes,
     based on a certain test defined by f_test

    Parameters
    ----------
    m: Model,
        ex Gross2022
    lst_thrs: list[float],
        list of threshold for the given tests
    f_test : function,
        test function
    lst_vars : list[int],
        list of the variables considered for the test
    b: int,
        number of batches,subdivisions cuted for the stationarity test
    bs0: int,
        initial btach length
    n_batch: int,
        number of batches for the subdivision
    max_count: int,
        maximum number of step
    
    Returns
    -------
    None if not found or:
    ... : int,
        potential relaxation time

    mainv : (t_end, 16) array,
        outputs of the main variables 
    """
    bs = bs0
    mainv = m.simulate(t_end=n_batch * bs).iloc[:, lst_vars]
    count = 0
    while f_test(mainv, lst_thrs, b, bs) and (count < max_count):
        bs *= 2
        mainv = m.simulate(t_end=n_batch * bs).iloc[:, lst_vars]
        count += 1
    if count >= max_count:
        return None
    return b * bs, mainv


def get_relaxation_time_agreg(
    m,
    nsim: int,
    lst_thrs: list[float],
    f_test,
    lst_vars: list[int] = np.arange(16),
    b: int = 4,
    bs0: int = 4,
    n_batch: int = 64,
    max_count: int = 12,
):
    """ Give the relaxation time of the mean of the simulations, if it existes,
     based on a certain test defined by f_test

    Parameters
    ----------
    m: Model,
        ex Gross2022
    nsim: int,
        number of simulations
    lst_thrs: list[float],
        list of thresholds for the given tests
    f_test : function,
        test function
    lst_vars : list[int],
        list of the variables considered for the test
    b: int,
        number of batches,subdivisions cuted for the stationarity test
    bs0: int,
        initial batch length
     n_batch: int,
         number of batches,subdivisions
    max_count: int,
        maximum number of step

    Returns
    -------
    None if not found or:
    ... : int,
        potential relaxation time

    mainv : (t_end, len(lst_vars), nsim) array,
        outputs of the main variables 
    """
    bs = bs0
    ouputs = nsimulate(m, nsim=nsim, t_end=n_batch * bs)
    mainv = ouputs[:, lst_vars, :]
    count = 0
    while f_test(mainv, lst_thrs, b, bs) and (count < max_count):
        bs *= 2
        ouputs = nsim_mproc(m, nsim=nsim, t_end=n_batch * bs)
        mainv = ouputs[:, lst_vars, :]
        count += 1
    if count >= max_count:
        return None
    return b * bs, ouputs

###########################
#  asymptotical analysis  #
###########################

def asymptotical_analysis(t_relax, mainv)->(float):
    """
    Do the asymptotical_analysis: the mean after relaxation time
    is an estimation of the steady state values
    
    Parameters
    ----------
    t_relax : int,
        relaxation time
    mainv : pd.DataFrame,
        results of a simulation
        
    Returns
    -------
    mean : float,

    """
    cut_mainv = mainv[t_relax:, :].copy()
    return np.nanmean(cut_mainv, axis=(1))

def asymptotical_analysis_agreg(t_relax, mainv)->(float):
    """
    Do the asymptotical_analysis on the mean of the simulations
    
    Parameters
    ----------
    t_relax : int,
        relaxation time
    mainv : pd.DataFrame,
        results of a simulation
        
    Returns
    -------
    mean : float,

    """
    cut_mainv = mainv[t_relax:, :, :].copy()
    return np.nanmean(cut_mainv, axis=(1, 2))
