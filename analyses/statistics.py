# -*- coding: utf-8 -*-
"""
statistics and test used for statistical analyses : Mean, Stationarity tests, ...
certain tests are taken from Vandin & al (2021) : http://arxiv.org/abs/2102.05405
"""
__author__ = ["Samuel Bazaz"]
__credits__ = ["Samuel Bazaz"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Samuel Bazaz"]

##############
#  Packages  #
##############

import numpy as np
import scipy.stats as ss
from statsmodels.tsa import stattools

##################
#  mean and var  #
##################

def get_mean_and_var(outputs: np.array) -> (np.array, np.array):
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
    mean = np.nanmean(outputs, axis=2)
    var = np.nanvar(outputs, axis=2)
    return mean, var


def get_batch_means(ouput0, bs: int) -> np.array:
    """ Give means by batch for every variables
    With the data of a unique simulation
    
    Parameters
    ----------
    ouput0 : pd.DataFrame
        certains variables of the output of a simulation
    bs:int,
        number of batches
        
    Returns
    -------
    res : (bs, nvar) array,
    """
    ouput = ouput0.copy()
    t_end, nvar = ouput.shape
    n_batch = t_end // bs
    res = np.zeros((n_batch, nvar))
    for i in range(n_batch - 1):
        res[i, :] = ouput.iloc[i * bs : (i + 1) * bs, :].mean(axis=0, skipna=True)
    res[n_batch - 1, :] = ouput.iloc[(n_batch - 1) * bs :, :].mean(axis=0, skipna=True)
    return res


def get_batch_means_agreg(ouputs0: np.array, bs: int) -> np.array:
    """ Give means by batch for every variables
    With the data of several simulations
    
    Parameters
    ----------
    ouputs0 : (t_end, nvar, nsim) array
        certains variables of the output of a simulation
    bs:int,
        number of batch
        
    Returns
    -------
    res : (bs, nvar) array,
    """
    ouputs = ouputs0.copy()
    t_end, nvar, nsim = np.shape(ouputs)
    n_batch = t_end // bs
    res = np.zeros((n_batch, nvar, nsim))
    for i in range(n_batch - 1):
        res[i, :, :] = np.nanmean(ouputs[i * bs : (i + 1) * bs, :, :], axis=0)
    res[n_batch - 1, :, :] = np.nanmean(ouputs[(n_batch - 1) * bs :, :, :], axis=0)
    return np.nanmean(res, axis=2)


#########################
#  confidence interval  #
#########################

def iid_ci_width(var: np.array, nsim: int, sign: float) -> np.array:
    """ Give confidence interval width of significance sign from
    independent variables.
    
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
    q = ss.t(nsim - 1).ppf(1 - sign / 2)
    return 2 * q * np.sqrt(var / nsim)

########################
#  Stationarity tests  #
########################

def rech_dycho_closest_inf(x: float, lst: list) -> int:
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
    g = 0
    d = len(lst) - 1
    if x <= lst[0]:
        return 0
    elif x > lst[d]:
        return len(lst) - 1
    while d - g > 1:
        m = (d + g) // 2

        if x > lst[m]:
            g = m
        elif x == lst[m]:
            return m
        else:
            d = m
    return g


def batch_test(ouput, lst_thrs:list[float], b: int = 4, bs: int = 4)->bool:
    """Stationarity test, based on a normality test on batch means 
    and a lag 1 autocorelation threshold
    
    Parameters
    ----------
    ouput : pd.DataFrame,
        result of a simulation
    lst_thrs : list[float],
        list of thresholds for the given tests
    b: int,
        number of batches,subdivisions cuted for the stationarity test
    bs: int,
        initial batch length
        
    Returns
    -------
    ...: bool,
        if true the stationarity hypothesis is rejected
    """
    sign_thrs = lst_thrs[0]
    corr_thrs = lst_thrs[1]

    batch_means = get_batch_means(ouput, bs)
    batch_means = batch_means[(b - 1) :, :]

    critical_values = [0.984, 0.827, 0.709, 0.591, 0.519]
    lst_sign = [1.0, 2.5, 5.0, 10.0, 15.0]
    stats = [ss.anderson(batch_means[:, i], dist="norm").statistic for i in range(16)]

    stat_thrs = critical_values[rech_dycho_closest_inf(sign_thrs, lst_sign)]
    reject_normality = max(stats) > stat_thrs


    lag_1_corr = [stattools.acf(batch_means[:, i], nlags=1)[1] for i in range(16)]
    reject_nocorr = max(np.abs(lag_1_corr)) > corr_thrs

    print(f"normality statistic:{max(stats)}, autocorrelation:{max(np.abs(lag_1_corr))}")
    return reject_normality or reject_nocorr

def kolmogorov_smirnov_test(ouput, lst_thrs:list[float], b: int = 4, bs: int = 4)->bool:
    """Stationarity test, based on a two parts Kolmogorov Smirnov test of equal distribution
    and a lag 1 autocorelation threshold
    
    Parameters
    ----------
    ouput : pd.DataFrame,
        results of a simulation
    lst_thrs : list[float],
        list of thresholds for the given tests
    b: int,
        number of batches,subdivisions cuted for the stationarity test
    bs: int,
        initial batch length
        
    Returns
    -------
    ...: bool,
        if true the stationarity hypothesis is rejected
    """
    sign_thrs = lst_thrs[0]/100
    corr_thrs = lst_thrs[1]

    ouput_cut = ouput.iloc[(b - 1) :, :]
    sample1 = ouput_cut.iloc[: len(ouput_cut) // 2, :]
    sample2 = ouput_cut.iloc[len(ouput_cut) // 2 :, :]
    pvals = np.array(
        [
            100
            * ss.ks_2samp(
                np.array(sample1.iloc[:, i]).flatten(),
                np.array(sample2.iloc[:, i]).flatten(),
                alternative="two-sided",
            ).pvalue
            for i in range(16)
        ]
    )
    pvals = pvals[~np.isnan(pvals)]
    reject_equal_cpf = pvals.min() < sign_thrs

    batch_means = get_batch_means(ouput_cut, bs)
    lag_1_corr = np.array(
        [stattools.acf(batch_means[:, i], nlags=1)[1] for i in range(16)]
    )
    lag_1_corr = lag_1_corr[~np.isnan(lag_1_corr)]
    reject_nocorr = np.abs(lag_1_corr).max() > corr_thrs
    print(f"distribution min pval:{pvals.min()}, autocorrelation:{np.abs(lag_1_corr).max()}")
    return reject_equal_cpf or reject_nocorr



def batch_test_agreg(mainv:np.array, lst_thrs:list[float], b: int = 4, bs: int = 4)->bool:
    """Stationarity test of the mean of the simulations, based on a normality test on batch means 
    and a lag 1 autocorelation threshold
    
    Parameters
    ----------
    mainv : pd.DataFrame,
        results of a simulation
    lst_thrs : list[float],
        list of thresholds for the given tests
    b: int,
        number of batches,subdivisions cuted for the stationarity test
    bs: int,
        initial batch length
        
    Returns
    -------
    ...: bool,
        if true the stationarity hypothesis is rejected
    """
    sign_thrs = lst_thrs[0]
    corr_thrs = lst_thrs[1]
    nvar = np.shape(mainv)[1] 
    batch_means = get_batch_means_agreg(mainv, bs)
    batch_means = batch_means[(b - 1) :, :]

    critical_values = [0.984, 0.827, 0.709, 0.591, 0.519]
    lst_sign = [1.0, 2.5, 5.0, 10.0, 15.0]
    stats = [ss.anderson(batch_means[:, i], dist="norm").statistic for i in range(nvar)]

    stat_thrs = critical_values[rech_dycho_closest_inf(sign_thrs, lst_sign)]
    reject_normality = max(stats) > stat_thrs

    lag_1_corr = np.array(
        [stattools.acf(batch_means[:, i], nlags=1)[1] for i in range(nvar)]
    )
    lag_1_corr = lag_1_corr[~np.isnan(lag_1_corr)]
    reject_nocorr = np.abs(lag_1_corr).max() > corr_thrs
    print(f"normality statistic:{max(stats)}, autocorrelation:{max(np.abs(lag_1_corr))}")
    return reject_normality or reject_nocorr


def pearson_test_agreg(mainv, lst_thrs:list[float], b: int = 4, bs: int = 4)->bool:
    """Stationarity test of the mean of the simulations,
   based on a Pearson normality test on batch means
   and a lag 1 autocorelation threshold
    
    Parameters
    ----------
    mainv : pd.DataFrame,
        results of a simulation
    lst_thrs : list[float],
        list of thresholds for the given tests
    b: int,
        number of batches,subdivisions cuted for the stationarity test
    bs: int,
        initial batch length
        
    Returns
    -------
    ...: bool,
        if true the stationarity hypothesis is rejected
    """
    sign_thrs = lst_thrs[0]
    corr_thrs = lst_thrs[1]
    nvar = np.shape(mainv)[1] 
    batch_means = get_batch_means_agreg(mainv, bs)
    batch_means = batch_means[(b - 1) :, :]

    pvals = np.array([ss.normaltest(batch_means[:, i]).pvalue for i in range(nvar)])
    pvals = pvals[~np.isnan(pvals)]
    reject_normality = pvals.min() < sign_thrs

    lag_1_corr = np.array(
        [stattools.acf(batch_means[:, i], nlags=1)[1] for i in range(nvar)]
    )
    lag_1_corr = lag_1_corr[~np.isnan(lag_1_corr)]
    reject_nocorr = np.abs(lag_1_corr).max() > corr_thrs

    print(f"normality min pval:{pvals.min()}, autocorrelation:{np.abs(lag_1_corr).max()}")
    return reject_normality or reject_nocorr


def kolmogorov_smirnov_test_agreg(mainv:np.array, lst_thrs:list[float], b: int = 4, bs: int = 4)->bool:
    """Stationarity test of the mean of the simulations, 
    based on a Kolmogorov Smirnov test of equal distribution
    and a lag 1 autocorelation threshold
    
    Parameters
    ----------
    mainv : pd.DataFrame,
        results of a simulation
    lst_thrs : list[float],
        list of thresholds for the given tests
    b: int,
        number of batches,subdivisions cuted for the stationarity test
    bs: int,
        initial batch length
        
    Returns
    -------
    ...: bool,
        if true the stationarity hypothesis is rejected
    """
    sign_thrs = lst_thrs[0]
    corr_thrs = lst_thrs[1]

    mainv_cut = mainv[(b - 1) :, :, :]
    sample1 = mainv_cut[: len(mainv_cut) // 2, :, :]
    sample2 = mainv_cut[len(mainv_cut) // 2 :, :, :]
    pvals = np.array(
        [
            100
            * ss.ks_2samp(
                sample1[:, i, :].flatten(),
                sample2[:, i, :].flatten(),
                alternative="two-sided",
            ).pvalue
            for i in range(16)
        ]
    )
    pvals = pvals[~np.isnan(pvals)]
    reject_equal_cpf = pvals.min() < sign_thrs

    batch_means = get_batch_means_agreg(mainv_cut, bs)
    lag_1_corr = np.array(
        [stattools.acf(batch_means[:, i], nlags=1)[1] for i in range(16)]
    )
    lag_1_corr = lag_1_corr[~np.isnan(lag_1_corr)]
    reject_nocorr = np.abs(lag_1_corr).max() > corr_thrs
    print(f"distribution min pval:{pvals.min()}, autocorrelation:{np.abs(lag_1_corr).max()}")
    return reject_equal_cpf or reject_nocorr

