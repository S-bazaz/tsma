# -*- coding: utf-8 -*-
"""
functions for clustering

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
import numpy as np
import pandas as pd
import tslearn.clustering as ts
import networkx as nx
import seaborn as sns

from tqdm import tqdm

from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset
from iisignature import sig

from sklearn import cluster, mixture
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.neighbors import kneighbors_graph

# from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from tslearn.metrics import cdist_dtw

##################
#  importations  #
##################

path = os.path.dirname(os.getcwd())
sys.path.append(path)

from basics.transfers import add_seeds_to_list, sep, join

from collect.output_management import (
    get_save_path,
    query_nparameters,
    query_parameters_specific,
    query_simulations,
    load_temp_outputs,
)
from basics.text_management import (
    get_cluster_codes_and_label,
    para_to_string,
    decode_label,
    get_nclust,
    
)
from visuals.figures import my_heatmap, distance_hist


######################
#  data importation  #
######################


def get_data_for_clustering(
    m,
    nsim: int,
    lst_sim_ids: list[int] = [],
    sim_id0: int = 0,
    step: int = 50,
    t_end: int = 500,
    warmup: int = 0,
    csv_name: str = "",
) -> tuple[pd.DataFrame, np.array, str]:
    """ Extract data from the database for the clustering and visualization
    
    Parameters
    ----------
    m: model object,
        ex Gross2022()
    varnames0: list[str],
        list of variables
    nsim: int,
        number of simulations
    lst_sim_ids : list[int], optional
        list of sim_id
    sim_id0 : int, optional
        starting sim_id
    step: int, optional
        step to increase sim_id
    t_end : int, optional
        Total runtime of the model
    warmup: int, optional
        warmup period to be removed from the data
    csv_name : str, optional
        name of the csv file containing the data
        
    Returns
    -------
    df_params : pd.DataFrame
        dataframe of the parameters for each simulation
    outputs0 : np.array
        array of the outputs of the simulations
    path_figures : str
        path to the figures directory
    """
    path_figures = os.sep.join([get_save_path(m), "figures"])

    if len(lst_sim_ids) > 0:
        nseeds_per_sim = nsim // len(lst_sim_ids)
        print(f"number of seeds per set of parameters : {nseeds_per_sim}")
        sim_ids = add_seeds_to_list(nseeds_per_sim, lst_sim_ids)

        df_params = query_parameters_specific(m, sim_ids)
        df_distinct_params = df_params.loc[
            :, (df_params.columns != "sim_id") * (df_params.columns != "p2__seed")
        ].drop_duplicates()
        df_distinct_params["params_id"] = np.arange(len(df_distinct_params))
        df_params = pd.merge(
            df_params,
            df_distinct_params,
            how="left",
            on=list(df_distinct_params.iloc[:, :-1].columns),
        )
    else:
        df_params = query_nparameters(
            m, nsim=nsim, sim_id0=sim_id0, step=step, t_end=t_end
        )
    sim_ids = df_params["sim_id"]

    if csv_name != "":
        outputs0 = load_temp_outputs(m, csv_name)[warmup:, :, :]
    else:
        outputs0 = query_simulations(m, sim_ids)[warmup:, :, :]
    sim_ids = "S" + np.array(sim_ids.astype(str), dtype=object)
    df_params["sim_id"] = sim_ids
    df_params.index = sim_ids

    return df_params, outputs0, path_figures


###############
#  embedding  #
###############


def embedding_lead_lag(outputs: np.array, nlag: int = 1) -> np.array:
    """Compute a lead-lag embedding of the given outputs.

    Parameters
    ----------
    outputs : np.array
        Simulation outputs
    nlag : int
        Number of lags to consider
    
    Returns
    -------
    outputs_embeded : np.array
        Lead-lag embedded simulation outputs
    """
    (t_end, nvar, nsim) = np.shape(outputs)
    outputs_embeded = np.zeros((t_end, nvar * (1 + nlag) + 1, nsim))
    outputs_embeded[:, 0, :] = np.repeat([np.arange(t_end)], [nsim], axis=0).reshape(
        (t_end, nsim)
    )
    outputs_embeded[:, 1 : (nvar + 1), :] = outputs.copy()
    for lag in range(1, nlag + 1):
        start_i = lag * nvar + 1
        end_i = (lag + 1) * nvar + 1
        outputs_embeded[:, start_i:end_i, :] = np.roll(outputs, lag, axis=0)
        outputs_embeded[:lag, start_i:end_i, :] = 0
    return outputs_embeded


###################
#  preprocessing  #
###################


def select_var_and_no_nan(
    varnames: list[str], outputs0: np.array, lst_var: list[int] = []
) -> tuple[list[str], np.array]:
    """Select variables in the output and replace NaN values by 0.
   
   Parameters
   ----------
   varnames: List[str]
       List of names of variables in the outputs.
   outputs0: np.array
       Array of outputs.
   lst_var: Optional[List[int]], default []
       List of indices of variables to select in the outputs.
       If not provided, all variables are selected.
       
   Returns
   -------
   Tuple[List[str], np.array]
       Tuple containing the list of selected variable names 
       and the array of selected and cleaned outputs.
   """
    outputs = np.nan_to_num(outputs0.copy())
    if lst_var == []:
        return varnames, outputs
    return varnames[lst_var], outputs[:, lst_var, :]


def drop_divergences(
    outputs: np.ndarray,
    lst_sim: list[str],
    divergence_thrs: float = 1e9,
    ndiv_thrs: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove simulations that have diverged, as defined by having more than `ndiv_thrs` 
    time steps where the value of a variable exceeds `divergence_thrs` in absolute value.
    
    Parameters
    ----------
    outputs : np.ndarray
        Array of simulation outputs of shape (time steps, variables, simulations)
    lst_sim : List[str]
        List of simulation IDs corresponding to `outputs`
    divergence_thrs : float, optional
        Threshold for defining divergence (default is 1e9)
    ndiv_thrs : int, optional
        Number of time steps that must exceed `divergence_thrs` 
        for a simulation to be considered diverged (default is 2)
        
    Returns
    -------
    outputs_dtw : np.ndarray
        Array of simulation outputs with divergences removed
    lst_sim : np.ndarray
        Array of simulation IDs with divergences removed
    """
    no_divergence = np.sum((np.abs(outputs) > divergence_thrs), axis=(0, 1)) < ndiv_thrs
    outputs_dtw = outputs[:, :, no_divergence].copy()
    return outputs_dtw, np.array(lst_sim[no_divergence])


def normalized_time_series(outputs: np.array, minmax_mode: bool = False) -> np.array:
    """ Normalize the time series dataset, either using MinMaxScaler or StandardScaler

    Parameters
    ----------
    outputs : np.array
        array of the time series data, of shape (t_end, nvar, nsim)
    minmax_mode : bool, optional
        whether to use MinMaxScaler or StandardScaler, by default False
        
    Returns
    -------
    np.array
        Normalized time series data, of shape (nsim, t_end, nvar)
    """
    norm_outputs = to_time_series_dataset(
        np.transpose(outputs, axes=[2, 0, 1])
    )  # (nsim, t_end, nvar)
    if minmax_mode:
        return TimeSeriesScalerMinMax().fit_transform(norm_outputs)
    return TimeSeriesScalerMeanVariance().fit_transform(norm_outputs)


def standardize_vect_data(lst_vects: list[np.ndarray]) -> list[np.ndarray]:
    """Standardize a list of numpy arrays.
    
    Parameters
    ----------
    lst_vects: list of numpy arrays
        The list of numpy arrays to standardize.
        
    Returns
    -------
    list of numpy arrays
        The list of standardized numpy arrays.
    """
    return [StandardScaler().fit_transform(vects) for vects in lst_vects]


## signature transform
def get_length_sign(nvar: int, depth: int) -> int:
    """
    Calculate the length of the signature vector based on
    the number of variables and depth of the signature.
    
    Parameters
    ----------
    nvar: int
        Number of variables in the time series dataset.
    depth: int
        Depth of the signature.
    
    Returns
    -------
    res: int
        Length of the signature vector.
    """
    res = 0
    for k in range(1, depth + 1):
        res += nvar ** k
    return res


def get_signatures(norm_outputs: np.ndarray, depth: int = 2) -> np.ndarray:
    """
    Compute signatures of time series data.
    
    Parameters
    ----------
    norm_outputs : np.ndarray
        Normalized time series data with shape (t_end, nvar, nsim)
    depth : int, optional
        Depth of the signature transform
        
    Returns
    -------
    x_sign : np.ndarray
        Array of signatures with shape (nsim, length of signatures)
    """
    nvar, nsim = np.shape(norm_outputs)[1:]
    x_sign = np.zeros((nsim, get_length_sign(nvar, depth)))
    for i in range(nsim):
        x_sign[i, :] = sig(norm_outputs[:, :, i], depth)
    return x_sign


def add_preproc(
    preprocessing: dict[str, [list, np.ndarray]],
    sim_ids0: np.ndarray,
    varnames0: list[str],
    outputs0: np.ndarray,
    dct_v: dict[str, list[int]],
    dct_e: dict,
    dct_m: dict[str, object],
    dct_select: dict[str, dict[str, list[str]]],
) -> None:
    """ Add the preprocessed data in a dictionary, with their name.
    
    Parameters
    ----------
    preprocessing: Dict[str, Union[list, np.ndarray]],
        dictionary that will contain the results
    sim_ids0 : np.ndarray,
        list of ids of simulations
    varnames0 : list[str],
        list of variables names
    outputs0 : np.ndarray,
        outputs of the simulations
    dct_v : Dict[str, list[int]],
        dictionary to select variables in the outputs
    dct_e : Dict[str, Callable[[np.ndarray], np.ndarray]],
        dictionary to select embeddings in the outputs
    dct_m : Dict[str, object],
        dictionary to select metrics object
        to apply the method on the outputs
    dct_select : Dict[str, Dict[str, list[str]]],
        dictionary that contains the selection of the methods that you want to apply
        
    Returns
    -------
    None
    """
    for m, dct in dct_select.items():
        for v, lst_e in dct.items():
            for e in lst_e:
                t0 = t.time()

                partial_code = join([v, e])
                code_name = join([m, partial_code])

                varnames, xtrain = select_var_and_no_nan(varnames0, outputs0, dct_v[v])

                xtrain = dct_e[e](xtrain)
                sim_ids = sim_ids0.copy()

                xtrain, sim_ids = dct_m[m].preproc(xtrain, sim_ids, partial_code)

                t1 = t.time()
                preprocessing[code_name] = [sim_ids, varnames, xtrain, t1 - t0]


##################################
#  preprocessing visualizations  #
##################################


def cross_similarity_matrix(
    outputs: np.ndarray,
    code_name: str,
    sim_ids: list = [],
    is_vectors: bool = True,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Calculate the pairwise distance between rows of outputs.
    
    Parameters
    ----------
    outputs : np.ndarray
        A 2-D array with `m` rows and `n` columns.
    code_name : str
        A string to be used in the plot title.
    sim_ids : list, optional
        A list with the IDs of the rows in `outputs`.
        If not provided, the row indices will be used.
    is_vectors : bool, optional
        If True, the Euclidean distance will be calculated.
        If False, the DTW distance will be calculated.
    plot : bool, optional
        If True, a heatmap of the distance matrix will be plotted.
    
    Returns
    -------
    pd.DataFrame
        The distance matrix, with row and column labels.
    """

    if is_vectors:
        mat = euclidean_distances(outputs)
    else:
        mat = cdist_dtw(outputs)
    df = pd.DataFrame(mat)
    df.columns = np.array(sim_ids)
    df.index = np.array(sim_ids)

    if plot:
        fig = my_heatmap(
            df, f"Cross similarity matrix, vectors = {is_vectors} " + code_name
        )
        fig.show()
    return df


def show_cross_similarities(
    preprocessing: dict, df_params: pd.DataFrame, plot_hist: bool = True
) -> None:
    """
    Show cross similarities of the given time series data.
    
    Parameters:
    ----------
    - preprocessing: a dictionary of preprocessed time series data

    - df_params: a dataframe containing the sim_ids and the corresponding parameter values
    - plot_hist: a boolean indicating whether to plot a histogram of the distances
    
    Returns:
    -------
    None
    """
    for code_name, lst_preproc in preprocessing.items():
        metric_code = code_name.split("_")[0]
        sim_ids = lst_preproc[0]
        xtrain = lst_preproc[2]
        if metric_code in ["m1"]:
            print(np.shape(xtrain))
            df = cross_similarity_matrix(xtrain, code_name, sim_ids, False)
            print(f"no infinit distances : {len(df) == len(sim_ids) }")
        else:
            df = cross_similarity_matrix(xtrain, code_name, sim_ids, True)
        if plot_hist:
            df["sim_id"] = df.index
            df = pd.merge(
                df, df_params.loc[:, ["sim_id", "params_id"]], how="left", on="sim_id"
            )
            fig = distance_hist(df, "signature distance " + code_name)
            fig.show()


################
#  clustering  #
################


def set_clust_algo(args_clust: dict[str, [list[str], int, float]]) -> dict:
    """
    Set the list of clustering algorithms to use.

    Parameters
    ----------
    args_clust: dict
        Dictionary containing the configuration for the clustering algorithms.
        
    Returns
    -------
    dct_algo: dict
        Dictionary containing the clustering algorithms.
    """
    nclust = args_clust["n_clusters"]
    lst_model = args_clust["lst_models"]
    dct_algo = {}

    for mod_name in lst_model:
        if mod_name == "TsKMeans":
            dct_algo[mod_name] = lambda x, p: ts.TimeSeriesKMeans(
                n_clusters=nclust, metric=p, max_iter=30, n_jobs=-1
            )
        elif mod_name == "KMeans":
            dct_algo[mod_name] = lambda x, p: cluster.KMeans(
                n_clusters=nclust, n_init=10, max_iter=300
            )
        elif mod_name == "MeanShift":

            def mean_shilft(x, p):
                bandwidth = cluster.estimate_bandwidth(
                    x, quantile=args_clust["quantile"]
                )
                return cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

            dct_algo[mod_name] = mean_shilft
        elif mod_name == "Ward":

            def ward(x, p):
                connectivity = kneighbors_graph(
                    x, n_neighbors=args_clust["n_neighbors"], include_self=False
                )
                connectivity = 0.5 * (connectivity + connectivity.T)
                return cluster.AgglomerativeClustering(
                    n_clusters=nclust, linkage="ward", connectivity=connectivity,
                )

            dct_algo[mod_name] = ward
        elif mod_name == "Agglomerative_Clustering":

            def agglo(x, p):
                connectivity = kneighbors_graph(
                    x, n_neighbors=args_clust["n_neighbors"], include_self=False
                )
                connectivity = 0.5 * (connectivity + connectivity.T)
                return cluster.AgglomerativeClustering(
                    linkage="average",
                    affinity="cityblock",
                    n_clusters=nclust,
                    connectivity=connectivity,
                )

            dct_algo[mod_name] = agglo
        elif mod_name == "Spectral_Clustering":
            dct_algo[mod_name] = lambda x, p: cluster.SpectralClustering(
                n_clusters=nclust, eigen_solver="arpack", affinity="nearest_neighbors",
            )
        elif mod_name == "MiniBatch_KMeans":
            dct_algo[mod_name] = lambda x, p: cluster.MiniBatchKMeans(n_clusters=nclust)
        elif mod_name == "DBSCAN":
            dct_algo[mod_name] = lambda x, p: cluster.DBSCAN(eps=args_clust["eps"])
        elif mod_name == "OPTICS":
            dct_algo[mod_name] = lambda x, p: cluster.OPTICS(
                min_samples=args_clust["min_samples"],
                xi=args_clust["xi"],
                min_cluster_size=args_clust["min_cluster_size"],
            )
        elif mod_name == "Affinity_Propagation":
            dct_algo[mod_name] = lambda x, p: cluster.AffinityPropagation(
                damping=args_clust["damping"],
                preference=args_clust["preference"],
                random_state=0,
            )
        elif mod_name == "BIRCH":
            dct_algo[mod_name] = lambda x, p: cluster.Birch(n_clusters=nclust)
        else:
            dct_algo[mod_name] = lambda x, p: mixture.GaussianMixture(
                n_components=nclust, covariance_type=args_clust["covariance_type_gmm"],
            )
    return dct_algo


def get_list_models(dct_m: dict) -> list[str]:
    """
    Get list of unique clustering algorithms compatible with the metrics
    
    Parameters:
    ----------
    dct_m: dict,
        dictionary of metrics
    
    Returns:
    -------
    lst_mod: list of strings
        list of unique clustering algorithms
    """
    lst_mod = []
    for m in dct_m.values():
        lst_mod = lst_mod + m.algo
    return list(set(lst_mod))


def m_to_algopara(metric, algo: str) -> str:
    """
    Convert a metric to the parameter of an algorithm.

    Parameters
    ----------
    metric: Union[Metric, str]
        Metric object or name of the metric.
    algo: str
        Name of the algorithm.

    Returns
    -------
    str
        The corresponding parameter for the algorithm.
    """
    if metric.mtype == "":
        return "no_para"
    elif algo == "TsKMeans":
        return metric.mtype
    else:
        return "no_para"


def clust_1step(
    clusters: dict,
    clust_name: str,
    nclust: int,
    algo: str,
    x_train: np.ndarray,
    metric,
    dct_algo: dict,
    f_para=m_to_algopara,
) -> None:
    """
    Clusters data into `nclust` number of clusters using algorithm `algo`
    and stores the results in the dictionary `clusters`.
    
    Parameters
    ----------
    clusters: Dict[str, Union[cluster.BaseEstimator, np.ndarray, float, int]]
        Dictionary to store the results of clustering.
    clust_name: str
        Name to give to the cluster.
    nclust: int
        Number of clusters to form.
    algo: str
        Clustering algorithm to use.
    x_train: np.ndarray
        Data to cluster.
    metric: Metric,
        Metric to use for clustering.
    dct_algo: Dict[str, Callable[[np.ndarray, Union[str, callable]], cluster.BaseEstimator]]
        Dictionary of clustering algorithms.
    f_para: Optional[Callable[[Union[str, callable], str], Union[str, callable]]] = m_to_algopara
        Function to map the metric and algorithm to their corresponding parameters.
    """
    t0 = t.time()
    mod = dct_algo[algo](x_train, f_para(metric, algo))

    mod.fit(x_train)
    t1 = t.time()

    if hasattr(mod, "labels_"):
        groups = mod.labels_.astype(int)
    else:
        groups = mod.predict(x_train)
    clusters[clust_name] = [mod, groups, (t1 - t0), nclust]


def add_clusters_1step(
    clusters: dict,
    preprocessing: dict,
    args_clust: dict,
    session_name: str,
    dct_m: dict,
    f_algo=set_clust_algo,
    f_para=m_to_algopara,
) -> None:
    """
    Add clusters to the input clusters dictionary, 
    using the given preprocessing data, clustering arguments, 
    session name, and mapping of model names to model objects.
    
    Parameters:
    ----------
    clusters (dict),
        dictionary to store the generated clusters
    preprocessing (dict),
        dictionary containing preprocessed data for clustering
    args_clust (dict),
        dictionary of arguments for clustering
    session_name (str),
        name of the current session
    dct_m (dict): 
        dictionary of metrics
    f_algo (function, optional)
        function to set the clustering algorithm. Defaults to set_clust_algo.
    f_para (function, optional),
        function to set the parameters for the clustering algorithm.
        Defaults to m_to_algopara.
    
    Returns:
    -------
    None
    """
    nclust = args_clust["n_clusters"]
    lst_model = args_clust["lst_models"]
    dct_algo = f_algo(args_clust)

    k = "k" + str(nclust)
    for code, lst_preproc in preprocessing.items():
        m = code.split("_")[0]
        metric = dct_m[m]
        x_train = lst_preproc[2]

        for algo in metric.algo:
            if algo in lst_model:
                clust_name = join([code, k, algo, session_name])
                clust_1step(
                    clusters,
                    clust_name,
                    nclust,
                    algo,
                    x_train,
                    metric,
                    dct_algo,
                    f_para=f_para,
                )


def add_clusters(
    clusters: dict,
    preprocessing: dict,
    args_clust: dict,
    session_name: str,
    dct_m: dict,
    f_algo=set_clust_algo,
    f_para=m_to_algopara,
) -> None:
    """
    Add clusters to the `clusters` dictionary using the different models specified in
    `args_clust["lst_models"]` and the number of clusters specified in `args_clust["lst_nclust"]`.
    The preprocessing steps specified in `preprocessing` are used to obtain the data to cluster.
    The session name is used to differentiate the clusters obtained in different runs.
    
    Parameters
    ----------
    clusters: dict
        Dictionary to store the clusters.
    preprocessing: dict
        Dictionary containing the preprocessing steps to be applied to the data.
    args_clust: dict
        Dictionary containing the arguments for clustering,
        including the list of models and number of clusters to be used.
    session_name: str
        Name of the current session, used to differentiate the clusters obtained in different runs.
    dct_m: dict
        Dictionary containing the different metrics to be used.
    f_algo: function, optional
        Function to set the clustering algorithms. Defaults to `set_clust_algo`.
    f_para: function, optional
        Function to obtain the algorithm-specific parameters. Defaults to `m_to_algopara`.
    
    Returns
    -------
    None
    """
    for nclust in args_clust["lst_nclust"]:
        args_clust["n_clusters"] = nclust
        add_clusters_1step(
            clusters,
            preprocessing,
            args_clust,
            session_name,
            dct_m,
            f_algo=f_algo,
            f_para=f_para,
        )


##################################
#  clustering results processing #
##################################


def clean_df_params(df_params0: pd.DataFrame, drop_col: list[str]) -> pd.DataFrame:
    """
    Remove specified columns from the DataFrame and update the remaining column names.
    
    Parameters:
    df_params0 (pd.DataFrame),
        The original DataFrame containing the parameters.
    drop_col (List[str]),
        A list of column names to be dropped.
    
    Returns:
        pd.DataFrame,
    The modified DataFrame with updated column names and dropped columns.
    """
    df_params = df_params0.drop(columns=drop_col)
    df_params.columns = list(
        map(lambda x: x if len(sep(x, 2)) == 1 else sep(x, 2)[1], df_params.columns,)
    )
    return df_params


def is_good_clustering(groups: list[int], ncluster: int) -> bool:
    """
    Determines if a clustering is acceptable.

    Parameters
    ----------
    groups : List[int]
        A list of integers representing the cluster assignments for each data point.
    ncluster : int
        The number of clusters that the data has been partitioned into.

    Returns
    -------
    bool
        True if the clustering is acceptable, False otherwise.
    """
    if groups is None:
        return False
    elif len(set(groups)) == 1:
        return False
    else:
        temp = np.array(groups)
        temp = (temp < 0) + (temp >= ncluster + 1)
        return not (temp > 0).any()


def clusterings_to_df(clusterings: dict[str, list]) -> pd.DataFrame:
    """Convert a dictionary of clusterings to a Pandas DataFrame.

    Parameters
    ----------
    clusterings: Dict[str, List]
        Dictionary of clusterings. Each value is a list containing the model, the cluster labels,
        the clustering time, and the number of clusters. The key is the name of the clustering

    Returns
    -------
    df_clustering: pd.DataFrame
        DataFrame with columns 'name', 'model', 'res', 'clust_time', 'nclusters',
        and 'good_cluster'.
        The 'good_cluster' column indicates whether the clustering is considered valid or not.
    """
    df_clustering = pd.DataFrame(
        [[name] + lst for name, lst in clusterings.items()],
        columns=["name", "model", "res", "clust_time", "nclusters"],
    )
    df_clustering["good_cluster"] = [
        is_good_clustering(df_clustering["res"][i], df_clustering["nclusters"][i])
        for i in range(len(df_clustering.index))
    ]
    return df_clustering


def get_df_clusters(
    df_params2: pd.DataFrame, df_clustering: pd.DataFrame, preprocessing: dict
) -> pd.DataFrame:
    """
    Return a DataFrame of clustering results
    where each row represents a simulation
    
    Parameters
    ----------
    df_params2 : pd.DataFrame
        Dataframe containing the parameters for each simulation id.
    df_clustering : pd.DataFrame
        Dataframe containing the results of the clustering.
    preprocessing : dict
        Dictionary containing the preprocessing results for each time series.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the cluster assignment for each simulation id.
    """
    nrows = len(df_clustering.index)
    is_good_cluster = df_clustering.iloc[:, 5]
    res = pd.DataFrame({}, index=df_params2["sim_id"])
    for i in tqdm(range(nrows)):
        clustering_name = df_clustering.loc[i, "name"]
        lst_info = clustering_name.split("_")
        if is_good_cluster[i]:
            code_name = "_".join(lst_info[:3])
            clust_res = df_clustering.iloc[i, 2]
            sim_ids = preprocessing[code_name][0]

            para_clusters = (np.max(clust_res) + 1) * np.ones(len(df_params2))
            para_clusters[df_params2["sim_id"].isin(sim_ids)] = clust_res
            res[clustering_name] = para_clusters
    return res


def get_clustering_dfs(
    preprocessing: dict,
    clusterings: dict,
    df_params0: pd.DataFrame,
    drop_col: list[str] = [],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function takes in preprocessing data, clustering data,
    and a DataFrame of parameters and returns three DataFrames:
        one of the cleaned parameters,
        one of the clustering metadata,
        one of the cluster assignments for each simulation.
    
    Parameters:
    preprocessing (dict):
        A dictionary with keys as preprocessing codes and 
        values as lists containing the sim_id's, data, 
        and target for each preprocessing.
    clusterings (dict):
        A dictionary with keys as clustering names and 
        values as lists containing the model, cluster assignments,
        time taken, and number of clusters for each clustering.
    df_params0 (pd.DataFrame): 
        A DataFrame of the original simulation parameters.
    drop_col (List[str]):
        A list of column names to drop from the parameter DataFrame.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
        A tuple containing three DataFrames:
            one of the cleaned parameters,
            one of the clustering metadata,
            and one of the cluster assignments for each simulation.
    """
    df_params = clean_df_params(df_params0, drop_col)
    df_params.index = df_params["sim_id"]
    df_clusterings = clusterings_to_df(clusterings)
    df_clusters = get_df_clusters(df_params, df_clusterings, preprocessing)

    return df_params, df_clusterings, df_clusters


############################
#  clustering preselection #
############################


def add_scores(
    df_clustering: pd.DataFrame,
    preprocessing: dict,
    df_params2: pd.DataFrame,
    time_importance: float,
) -> None:
    """
    Add scores to the given `df_clustering` dataframe.

    Parameters
    ----------
    df_clustering : pd.DataFrame
        DataFrame containing clusterings.
    preprocessing : dict
        Dictionary with preprocessing information.
    df_params2 : pd.DataFrame
        DataFrame with parameters.
    time_importance : float
        Weight to give to the time in 'myscore'

    Returns
    -------
    None
    """
    nrows = len(df_clustering.index)
    silhouettes = -2 * np.ones(nrows)
    silh_para = -2 * np.ones(nrows)
    tottime = 1e9 * np.ones(nrows)
    is_good_cluster = df_clustering.iloc[:, 5]

    for i in tqdm(range(nrows)):
        clustering_name = df_clustering.loc[i, "name"]
        lst_info = clustering_name.split("_")

        if is_good_cluster[i]:
            code_name = "_".join(lst_info[:3])
            metric_code = lst_info[0]
            xtrain = preprocessing[code_name][2]
            sim_ids = preprocessing[code_name][0]

            para_clusters = -1 * np.ones(len(df_params2))
            para_clusters[df_params2["sim_id"].isin(sim_ids)] = df_clustering.iloc[i, 2]
            silh_para[i] = silhouette_score(
                df_params2.iloc[:, 1:-1], para_clusters, metric="euclidean"
            )
            tottime[i] = (
                df_clustering.loc[i, "clust_time"] + preprocessing[code_name][3]
            )

            if metric_code in ["m1"]:
                silhouettes[i] = ts.silhouette_score(
                    xtrain, df_clustering.iloc[i, 2], metric="dtw"
                )
            else:
                silhouettes[i] = silhouette_score(
                    xtrain, df_clustering.iloc[i, 2], metric="euclidean"
                )
    df_clustering["silhouette"] = silhouettes
    df_clustering["silh_para"] = silh_para
    df_clustering["tottime"] = tottime
    df_clustering["myscore"] = 0

    df_clustering.loc[is_good_cluster, "myscore"] = minmax_scale(
        silhouettes[is_good_cluster], feature_range=(0, 1), axis=0
    ) + time_importance * minmax_scale(
        -tottime[is_good_cluster], feature_range=(0, 1), axis=0
    )


def get_best_aux(
    df_clustering: pd.DataFrame,
    score: str,
    ninfo: int = 2,
    nbest: int = 2,
    largest: bool = True,
) -> list[int]:
    """
    Returns the indexes of the top `nbest` entries
    in `df_clustering` for each type.
    
    Parameters
    ----------
    df_clustering: pd.DataFrame
        DataFrame containing the clustering information.
    score: str
        The name of the column in `df_clustering` to use as the score.
    ninfo: int, optional
        The number of elements in the `name` column of `df_clustering` to use as the type.
        The default is 2.
    nbest: int, optional
        The number of top entries to select for each type. The default is 2.
    largest: bool, optional
        If True, select the top entries by finding the largest values of `score`.
        If False,select the top entries by finding the smallest values of `score`.
        The default is True.
        
    Returns
    -------
    List[int]
        The indexes of the top `nbest` entries in `df_clustering` for each type.
    """
    df = df_clustering.copy()
    df["type"] = np.array(
        list(map(lambda s: "_".join(str(s).split("_")[:ninfo]), df["name"]))
    )
    if largest:
        indexes = df.groupby(by="type")[score].nlargest(nbest).index
    else:
        indexes = df.groupby(by="type")[score].nsmallest(nbest).index
    indexes = list(map(lambda x: x[1], list(indexes)))

    return indexes


def preselection(
    preprocessing: dict,
    df_clustering: pd.DataFrame,
    df_params: pd.DataFrame,
    time_coef: float = 0,
    nbest1: int = 5,
    nbest2: int = 2,
) -> pd.DataFrame:
    """
    Select the best clusterings from `df_clustering` using the `myscore` column.

    Parameters
    ----------
    preprocessing : dict
        Dictionary with the preprocessed data.
    df_clustering : pd.DataFrame
        DataFrame with the results of the clusterings.
    df_params : pd.DataFrame
        DataFrame with the simulation parameters.
    time_coef : float, optional
        Coefficient to adjust the importance of the time taken for the clustering.
        Default is 0.
    nbest1 : int, optional
        Number of best clusterings to consider for the first selection. Default is 5.
    nbest2 : int, optional
        Number of best clusterings to consider for the second selection. Default is 2.

    Returns
    -------
    pd.DataFrame
        DataFrame with the results of the selected clusterings.
    """
    add_scores(df_clustering, preprocessing, df_params, time_coef)
    indexes = get_best_aux(
        df_clustering, "myscore", ninfo=3, nbest=nbest1, largest=True
    )
    indexes2 = get_best_aux(
        df_clustering.loc[indexes, :], "silh_para", ninfo=1, nbest=nbest2, largest=True
    )
    return df_clustering.loc[
        indexes2, ["name", "silhouette", "tottime", "myscore", "silh_para", "model"]
    ]


##################
#  phase diagram #
##################


def get_clustered_params(
    df_params: pd.DataFrame, clusterings: dict, preprocessing: dict, clust_code: str,
) -> pd.DataFrame:
    """
    Return a copy of `df_params` with an additional column 'cluster'
    that reflects the clusters obtained from the clustering model
    identified by `clust_code`.

    Parameters
    ----------
    df_params: pd.DataFrame
        DataFrame containing the parameters of the simulations.
    clusterings: dict
        Dictionary containing the information about the clusterings obtained.
    preprocessing: dict
        Dictionary containing the information about the preprocessing performed.
    clust_code: str
        Code identifying the clustering model to use.

    Returns
    -------
    pd.DataFrame
        Copy of `df_params` with an additional column 'cluster'.
    """
    clusters = clusterings[clust_code][1]
    preproc_code = join(sep(clust_code)[:3])
    sim_ids = preprocessing[preproc_code][0]
    df_params2 = df_params.copy()
    df_params2 = df_params2.drop(columns=["seed"])

    new_clust = max(clusters) + 1
    para_clusters = new_clust * np.ones(len(df_params2))
    para_clusters[df_params2["sim_id"].isin(sim_ids)] = clusters
    df_params2["cluster"] = para_clusters
    return df_params2


def get_best_vars_4_proj(
    df_params: pd.DataFrame,
    clusterings: dict,
    nvar: int,
    clust_code: str,
    heatmap: bool = True,
    size: int = 12,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Get best variables for projection of the parameter space.

    Parameters
    ----------
    df_params: pd.DataFrame
        DataFrame with the parameters of the simulations.
    clusterings: dict
        Dictionnary with the clusters for each code.
    nvar: int
        Number of variables to keep for the projection.
    clust_code: str
        Code of the clustering to consider.
    heatmap: bool, optional
        Display a heatmap of the silouhette scores.
    size: int, optional
        Size of the heatmap.

    Returns
    -------
    df_params: pd.DataFrame
        DataFrame with the parameters of the simulations.
    list_columns: List[str]
        List of the variables to keep for the projection.
    """
    para_clusters = df_params["cluster"]
    df_aux = df_params.copy().drop(columns=["cluster", "sim_id"])
    cols = list(df_aux)
    ncol = len(cols)
    df_score = pd.DataFrame(-1 * np.ones((ncol, ncol)), columns=cols)
    df_score.index = cols
    for i in range(ncol):
        for j in range(ncol):
            if i != j:
                df_score.iloc[i, j] = silhouette_score(
                    df_aux.iloc[:, [i, j]], para_clusters, metric="euclidean"
                )
    if heatmap:
        df_score2 = df_score.copy()
        for i in range(ncol):
            df_score2.iloc[i, i] = np.nan
        my_heatmap(
            df_score2,
            "Silouhette score for parameters space projections " + clust_code,
            size=size,
        )
    sorted_coordinates = np.argsort(-(df_score.values).reshape(ncol * ncol))
    id_best_proj = []
    if nvar > ncol:
        return df_params, cols
    rank = 0
    count = 0
    while count < nvar:
        v = sorted_coordinates[rank]
        rank += 1
        i, j = v // ncol, v % ncol
        if i not in id_best_proj:
            id_best_proj.append(i)
            count += 1
        if count < nvar:
            if j not in id_best_proj:
                id_best_proj.append(j)
                count += 1
    return df_params, ["sim_id"] + list(np.array(cols)[id_best_proj]) + ["cluster"]


#################################
#  clustering characterization  #
#################################


def get_df_clusters2(df_clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the cluster assignments in `df_clusters` into a more usable form.

    Parameters
    ----------
    df_clusters: pd.DataFrame
        DataFrame with cluster assignments, where 
        each column corresponds to a different clustering method and
        each row corresponds to a simulation.
        The entries are integers indicating the cluster assignment for that
        simulation.

    Returns
    -------
    df_clusters2: pd.DataFrame
        DataFrame with one row for each cluster assignment,
        with the following columns:
        'cluster_id': str
            Unique identifier for the cluster.
        'ordercode': int
            Integer used to sort the clusters in a natural order.
        'label': str
            Human-readable label for the cluster.
        'clust_number': int
            Cluster number within the clustering method.
        'clustering_code': str
            Clustering method used.
        'sim_ids': List[int]
            List of simulation ids belonging to the cluster.
    """
    clusters_ids = []
    ordercodes = []
    labels = []
    clust_numbers = []
    clustering_codes = []
    sim_ids = []

    for code in list(df_clusters):
        nclust = get_nclust(code)
        for i in range(nclust):
            clust_numbers.append(i)
            cluster_id, ordercode, label = get_cluster_codes_and_label(code, f"g{i}")
            clusters_ids.append(cluster_id)
            ordercodes.append(ordercode)
            labels.append(label)
            clustering_codes.append(code)
            sim_ids.append(df_clusters.index[np.where(df_clusters[code] == i)])
    df_clusters2 = pd.DataFrame(
        {
            "cluster_id": clusters_ids,
            "ordercode": ordercodes,
            "label": labels,
            "clust_number": clust_numbers,
            "clustering_code": clustering_codes,
            "sim_ids": sim_ids,
        }
    ).sort_values("ordercode")
    df_clusters2.index = df_clusters2["cluster_id"]
    return df_clusters2


def get_faux_clust_ts(df_params: pd.DataFrame, outputs0: np.ndarray) -> tuple:
    """
    Return two functions for generating plots of time series for a given list of simulation ids.
    
    Parameters:
    - df_params: DataFrame of simulation parameters
    - outputs0: 3D numpy array of time series data
    
    Returns:
    - Tuple of two functions, fout and ftitle, where:
      - fout: function that takes a list of simulation ids
          and returns a 2D numpy array of time series data
      - ftitle: function that takes a list of simulation ids
          and a figure name, and returns a string title for the plot
    """
    df_params["row"] = np.arange(len(df_params))
    fid = lambda sim_ids: df_params.loc[sim_ids, "row"].values
    fout = lambda sim_ids: outputs0[:, :, fid(sim_ids)]
    ftitle = lambda sim_ids, figname: para_to_string(
        sim_ids,
        df_params.drop(columns=["sim_id", "seed", "row"]),
        figname.replace("_", " "),
    )
    return fout, ftitle


###########################
#  clustering Comparison  #
###########################


def jaccard(set1: set, set2: set) -> float:
    """
    Calculate the Jaccard index between two sets.
    
    Parameters
    ----------
    set1: Set[Any]
        A set of elements.
    set2: Set[Any]
        A set of elements.
    
    Returns
    -------
    float
        The Jaccard index between set1 and set2.
    """
    if len(set1) == 0 and len(set2) == 0:
        return 1
    intersection_len = len(set1.intersection(set2))
    union_len = len(set1) + len(set2) - intersection_len
    if union_len == 0:
        print(set1, set2)
    return intersection_len / union_len


def get_jaccard_dfs(
    df_clusters2: pd.DataFrame, heatmap: bool = True, size: int = 40
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate Jaccard coefficients between different clusters.
    
    Parameters:
    - df_clusters2: DataFrame containing information about the clusters.
    - heatmap: Boolean flag indicating whether to show a heatmap 
      of the Jaccard coefficients.
    - size: Integer indicating the size of the heatmap figure.
    
    Returns:
    A tuple of two DataFrames: 
        the first one contains the Jaccard coefficients
        between different clusters,
        the second one contains the Jaccard distances
        between different clusters.
    """
    n = len(df_clusters2)
    mat = -1 * np.ones((n, n))
    codes = list(df_clusters2.index)
    for i, code1 in enumerate(codes):
        for j, code2 in enumerate(codes):
            set1 = set(df_clusters2.loc[code1, "sim_ids"])
            set2 = set(df_clusters2.loc[code2, "sim_ids"])
            mat[i, j] = jaccard(set1, set2)
    df_jaccard = pd.DataFrame(
        mat, columns=df_clusters2["label"], index=df_clusters2["label"]
    )
    df_jaccdist = pd.DataFrame(
        1 - mat, columns=df_clusters2["label"], index=df_clusters2["label"]
    )
    if heatmap:
        my_heatmap(df_jaccard, "Jaccard coefficients", size=size)
    return df_jaccard, df_jaccdist


def jacc_net(df_jaccard: pd.DataFrame) -> nx.Graph:
    """
    Create a graph from a Jaccard coefficient matrix.
    
    Parameters:
        df_jaccard (pd.DataFrame): Jaccard coefficient matrix..
        
    Returns:
        nx.Graph: Graph created from the Jaccard coefficient matrix..
    """
    net = nx.from_numpy_matrix(df_jaccard.values)
    mapping = dict(enumerate(df_jaccard.index))
    #{i: code for i, code in enumerate(df_jaccard.index)}
    net = nx.relabel_nodes(net, mapping, copy=False)
    return net


def aux_df_labels(mat: np.ndarray, labels: list[str], i: int):
    """
    Modify the i-th row of `mat` to contain the decoded version of the label at index i in `labels`.
    
    Parameters
    ----------
    mat: np.ndarray
        The matrix to be modified.
    labels: List[str]
        A list of labels.
    i: int
        The index of the row in `mat` to be modified and the label in `labels` to be decoded.
    """
    mat[i] = np.array(decode_label(labels[i]))


def get_df_labels(net: nx.Graph) -> pd.DataFrame:
    """
    Returns a Pandas DataFrame object with
    the labels of the nodes in the graph net.

    Parameters:
    net: nx.Graph
        NetworkX graph object.
    
    Returns:
    pd.DataFrame
        DataFrame object with the labels of the nodes in `net`.
    """
    labels = list(net.nodes)
    n_node = len(labels)
    mat = np.repeat(np.array(["", -1, -1, -1, -1]), n_node).reshape((n_node, 5))
    f_aux = lambda i: aux_df_labels(mat, labels, i)
    np.vectorize(f_aux)(np.arange(n_node))
    df_labels = pd.DataFrame(mat, columns=["algo", "m", "v", "e", "k"], index=labels)
    return df_labels


def add_nodes_styles(df_labels: pd.DataFrame, pal: str = "Spectral") -> pd.DataFrame:
    """
    Add node colors and shapes to df_labels
    based on values of 'algo' and 'm' columns.
    The color palette is determined by 'pal' argument.
    
    Copy code
    Parameters
    ----------
    df_labels: pd.DataFrame
        DataFrame with labels of nodes.
    pal: str
        Color palette name.
    
    Returns
    -------
    df_labels2: pd.DataFrame
        DataFrame with the same structure as df_labels
        but with added 'node_color' and 'node_shape' columns.
    """
    df_labels2 = df_labels.copy()
    df_labels2["label"] = df_labels.index

    color_temp = df_labels2.loc[:, ["algo", "m"]].drop_duplicates()
    color_temp["node_color"] = sns.color_palette(pal, len(color_temp)).as_hex()
    df_labels2 = pd.merge(df_labels2, color_temp, how="inner", on=["algo", "m"])

    edgecol_temp = df_labels.loc[:, ["e"]].drop_duplicates()
    edgecol_temp["edgecolors"] = sns.color_palette(
        "magma", len(edgecol_temp)
    ).as_hex()  # crest
    df_labels2 = pd.merge(df_labels2, edgecol_temp, how="inner", on=["e"])

    shapes = np.array(list("sdoph8^>v<"))
    df_labels2["node_shape"] = shapes[np.array(df_labels2["v"], dtype=int) - 1]
    df_labels2.index = df_labels2["label"]

    return df_labels2
