# -*- coding: utf-8 -*-
"""
classes used to create clustering approaches

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
import pandas as pd

#################
#  importations #
#################

path = os.path.dirname(os.getcwd())
sys.path.append(path)

from visuals.figures import my_heatmap
from analyses.clustering import (
    drop_divergences,
    normalized_time_series,
    get_signatures,
    )

class Metric:
    """ Class template for metric creation and unsupervised clustering

    Attributes
    ----------
    code: str,
        code associated with the metric
    name: str,
        name of the metric
        for example, 'Euclidean distance'
    algo: list[str],
        list of the algorithms compatible with the metric
    mtype : str,
        code to detect special cases during clustering computation

    Public Methods
    --------------
    preproc
        do the right preprocessing for the given metric
    """

    def __init__(
        self,
        code: str = 'm',
        name: str = '',
        algo: list[str] = [],
        mtype = "",
    ):
        self.code = code
        self.name = name
        self.algo = algo
        self.mtype = mtype
        
    def preproc(self, xtrain, sim_ids, partial_code):
        """Run a simulation of the model. Implemented by each of the child
        classes separately

        Returns
        -------
        output : numpy array

        """
        raise NotImplementedError
        
        
class Ts_Metric(Metric):
    """ Time series metric : a distance beetween time series is directly computed.

    Attributes
    ----------
    code: str,
        code associated with the metric
    name: str,
        name of the metric
        for example, 'Euclidean distance'
    algo: list[str],
        list of the algorithms compatible with the metric
    mtype : str,
        code to detect special cases during clustering computation
    div_thrs : float,
        thresold to consider a value as a divergence
    ndiv_thrs : int,
        number of divergences accepted for a given simulation
        If this limit is exceeded, the simulation is dropped
    minmax : bool,
        whether the data is normalized with a minmax norm
        if not the mean and the variance are used
    
    Public Methods
    --------------
    preproc
        do the right preprocessing for the metric
    """
    
    def __init__(
        self,
        code: str = 'm1',
        name: str = 'Dynamic Time Warping',
        mtype:str = "dtw",
        div_thrs:float = 1e9,
        ndiv_thrs:int = 2,
        minmax:bool = False,
    ):
        """Initialization of the model class

        parameters
        ----------

        """
        super().__init__(code, name, ["TsKMeans"], mtype)
        self.div_thrs = div_thrs
        self.ndiv_thrs = ndiv_thrs
        self.minmax = minmax
        
    def preproc(self, xtrain, sim_ids, partial_code):

        xtrain, sim_ids = drop_divergences(
            xtrain, sim_ids, divergence_thrs=self.div_thrs, ndiv_thrs = self.ndiv_thrs
        )
        xtrain = normalized_time_series(xtrain, self.minmax)
        return xtrain, sim_ids
    
    
class Sig_Metric(Metric):
    """ Signature transform metric : 
        based on euclidian distance of the signature representation of the time series

    Attributes
    ----------
    code: str,
        code associated with the metric
    name: str,
        name of the metric
        for example, 'Euclidean distance'
    algo: list[str],
        list of the algorithms compatible with the metric
    mtype : str,
        code to detect special cases during clustering computation
    depth : int,
        depth of the signature, ie number of terms considered in the representation
    special_depths : dict,
        dictionary of special cases for depth
        for instance {'m2_e1_v1' : 3}
    minmax : bool,
        whether the data is normalized with a minmax norm
        if not the mean and the variance are used
    plot_sign : bool,
        whether the signature is visualized or not
    
    Public Methods
    --------------
    preproc
        do the right preprocessing for the metric
    """
    def __init__(
        self,
        code: str = 'm2',
        name: str = 'Signature metric',
        depth:int = 2,
        special_depths:dict = {},
        minmax:bool = False,
        plot_sign:bool = True,
    ):
        """Initialization of the model class

        parameters
        ----------

        """
        super().__init__(code, name,[
        "KMeans",
        "MiniBatch_KMeans",
        "Affinity_Propagation",
        "MeanShift",
        "Spectral_Clustering",
        "Ward",
        "Agglomerative_Clustering",
        "DBSCAN",
        "OPTICS",
        "BIRCH",
        "Gaussian_Mixture",
    ], "")
        self.depth = depth
        self.special_depths = special_depths
        self.minmax = minmax
        self.plot_sign = plot_sign
        
    def preproc(self, xtrain, sim_ids, partial_code):
        
        xtrain = normalized_time_series(xtrain, self.minmax)
        xtrain = np.transpose(xtrain, axes=[1, 2, 0])
        depth = self.depth
        spe_depth = self.special_depths
        
        if  partial_code in spe_depth:
            depth = spe_depth[partial_code]
        
        xtrain = get_signatures(xtrain, depth)
        if self.plot_sign:
            df_sign = pd.DataFrame(xtrain)
            df_sign.index = sim_ids
            fig = my_heatmap(df_sign, f"Signature coefficients depth:{depth}  " + self.code +"_"+partial_code)
            fig.show()
        xtrain = np.nan_to_num(xtrain)
        return xtrain, sim_ids
    
        
if __name__ == "__main__":
    pass
        