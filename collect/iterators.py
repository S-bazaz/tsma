# -*- coding: utf-8 -*-
"""
functions used to gather simulations results

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
from tqdm import tqdm
from itertools import repeat
from multiprocessing import Pool, cpu_count

#################
#  Importation  #
#################

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from tsma.collect.output_management import get_max_id

######################
#  Simple iterators  #
######################

def nsimulate(
    m,
    nsim: int = 1,
    lst_seeds: list = [],
    overwrite: bool = False,
    save: bool = False,
    t_end: int = None,
) -> np.array:
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

    f_aux = lambda s: m.simulate(overwrite=overwrite, save=save, t_end=t_end, seed=s)
    nsim2 = nsim
    seed = 0
    # The mode of generation is given by lst_seeds
    # If the latter is empty, the seeds are given by range(nsim)

    b_mode = len(lst_seeds) > 0
    if b_mode:
        nsim2 = len(lst_seeds)
        seed = lst_seeds[0]
    output = f_aux(seed)
    t_end = len(output)
    nvar = len(list(output))

    outputs = np.zeros((t_end, nvar, nsim2))
    outputs[:, :, 0] = output.values
    for i in tqdm(range(1, nsim2)):
        if b_mode:
            seed = lst_seeds[i]
        else:
            seed = i
        outputs[:, :, i] = f_aux(seed).values
    return outputs


def add_simulate(
    m,
    outputs0: np.array,
    seed: int = 0,
    overwrite: bool = False,
    save: bool = False,
    t_end: int = None,
) -> np.array:
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
    (t_end, nvar, nsim0) = np.shape(outputs0)
    outputs = np.zeros((t_end, nvar, nsim0 + 1))
    outputs[:, :, :nsim0] = outputs0
    outputs[:, :, nsim0] = m.simulate(
        overwrite=overwrite, save=save, t_end=t_end, seed=seed
    )
    return outputs


def aux_simulate(args: tuple) -> ():
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
    m, overwrite, save, sim_id, t_end, s = args
    output = m.simulate(
        overwrite=overwrite, save=save, sim_id=int(sim_id), t_end=t_end, seed=s
    ).values
    return output


def aux_zip_para(fix_args: tuple, seeds: np.array) -> ():
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
    m, overwrite, save, t_end = fix_args
    id0 = get_max_id(m)
    if id0 is None:
        sim_ids = np.arange(len(seeds))
    else:
        sim_ids = np.arange(id0 + 1, len(seeds) + id0 + 1)
    return zip(
        repeat(m), repeat(overwrite), repeat(save), sim_ids, repeat(t_end), seeds
    )


def aux_pool_sim(n_cores: int, zip_para, tgt_shape: tuple) -> np.array:
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
    res = Pool(n_cores).map(aux_simulate, zip_para)
    print(f"\nshape of the pool outputs: {np.shape(res)}")
    print(f"targeted shape: {tgt_shape}")
    if np.shape(res) != tgt_shape:
        return np.nan
    return np.transpose(np.array(res), (1, 2, 0))


def nsim_mproc(
    m,
    nvar: int,
    t_end: int,
    nsim: int = 1,
    seeds: list = [],
    overwrite: bool = False,
    save: bool = False,
    maxcore: int = -1,
) -> np.array:
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

    f_zip = aux_zip_para
    f_aux = aux_pool_sim
    fix_args = (m, overwrite, save, t_end)

    if len(seeds) > 0:
        seeds2 = seeds.copy()
    else:
        seeds2 = np.arange(nsim)
    nsim2 = len(seeds2)

    n_cores = min(cpu_count(), nsim2)
    if maxcore > 0:
        n_cores = min(n_cores, maxcore)
    npool = nsim2 // n_cores
    nsim_lastpool = nsim2 % n_cores

    print(f"number of cores used {n_cores}")
    print(f"number of pools {npool+int(nsim_lastpool>0)}")

    outputs = np.zeros((t_end, nvar, nsim2))

    for i in tqdm(range(npool)):
        res_shape = (n_cores, t_end, nvar)
        zip_para = f_zip(fix_args, seeds2[i * n_cores : (i + 1) * n_cores])
        outputs[:, :, i * n_cores : (i + 1) * n_cores] = f_aux(
            n_cores, zip_para, res_shape
        )
    if nsim_lastpool > 0:
        res_shape = (nsim_lastpool, t_end, nvar)
        zip_para = f_zip(fix_args, seeds2[npool * n_cores :])
        outputs[:, :, npool * n_cores :] = f_aux(n_cores, zip_para, res_shape)
    return outputs
