# -*- coding: utf-8 -*-
"""
Generic model class that serves as baseline for the remaining implementations
such as the Gross model
"""
__author__ = ["Karl Naumann-Woleske", "Samuel Jazayeri"]
__credits__ = ["Karl Naumann-Woleske", "Samuel Jazayeri"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Samuel Jazayeri"]


##############
#  Packages  #
##############

import os
import sys
import pandas as pd

#################
#  Importation  #
#################

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from tsma.basics.text_management import encode

from tsma.collect.output_management import (
    initialize_outputs,
    get_save_path,
    get_id_parameters,
    delete_simulation,
    query_simulation,
    save_parameters,
    save_simulation,
)


class Model:
    """ Class template for macro ABM

    Attributes
    ----------
    name: str
        Name of the model, such as "threesector". Used for filenames
    parameters : dict
        Dictionary of all parameters in decoded form 
    hyper_parameters : dict
        Dictionary of all hyperparameters in decoded form
    initial_values : dict
        Dictionary of all initial values in decoded form
    output : pd.DataFrame
        None, or the latest simulation run for given parameters
    debug : bool (default False)
        if true, prints step-wise calculations in the model

    Public Methods
    --------------
    simulate
        checks if simulation already exists, and executes a new simulation
        if necessary or desired
    load
        loads a simulation for the given set of parameters
    """

    def __init__(
        self,
        parameters: dict,
        hyper_parameters: dict,
        initial_values: dict,
        name: str = "",
        agent_ids=[],
    ):
        """Initialization of the model class

        Parameters
        ----------
        parameters : dict
            dictionary of the named parameters of the model
        hyper_parameters : dict
            dictionary of hyper-parameters related to the model
        initial_values : dict
            dictionary of the models initial values
        name : str (default '')
            name of the model (for use in filenaming)
        debug : bool (default False)
            enable to get stepwise printed output
        """
        # Essential attributes
        self.parameters = parameters
        self.hyper_parameters = hyper_parameters
        self.initial_values = initial_values
        self.name = name
        self.agent_ids = agent_ids
        self.sim_id = "S"

        # Attributes generated later on
        self.output = None

    def _simulate(
        self, overwrite: bool = False, save: bool = False, sim_id: int = None
    ) -> pd.DataFrame:
        """Simulate a model run and return it as a pandas dataframe. Simulation
        will be loaded if it exists and overwrite is false, and saved to the 
        database if save is true and it doesn't exist

        Parameters
        ----------
        overwrite : bool (default False)
            overwrite any existing saved simulation file if save is also true.
            Guarantees that a new simulation is run
        save : bool (default False)
            will save the output to the relevant model database            

        Returns
        -------
        output : pd.DataFrame
        """

        params = encode(
            self.parameters, self.hyper_parameters, self.initial_values, self.agent_ids
        )
        self.sim_id = "S"

        noquery = True
        if not os.path.exists(get_save_path(self)):
            self.output = self._run_simulation()
            initialize_outputs(params, list(self.output), self)
        else:

            index = get_id_parameters(params, self)
            if index is None:
                self.output = self._run_simulation()
            elif index[0] is None:
                self.output = self._run_simulation()
            else:
                self.sim_id += str(index[0])

                if overwrite:
                    self.output = self._run_simulation()

                    if save:
                        delete_simulation(self.sim_id, self)
                else:
                    self.output = query_simulation(self.sim_id, self)
                    noquery = False
        if save and noquery:
            if self.sim_id == "S":  # new params
                if sim_id is None:
                    self.sim_id += str(save_parameters(params, self))
                else:
                    self.sim_id += str(save_parameters(params, self, sim_id))
            save_simulation(self.output, self.sim_id, self)
        return self.output

    def load(self) -> pd.DataFrame:
        """Load a simulation from the database for the given set of parameters
        
        Returns
        -------
        output: pd.DataFrame
        """
        raise NotImplementedError

    def _run_simulation(self) -> pd.DataFrame:
        """Run a simulation of the model. Implemented by each of the child
        classes separately

        Returns
        -------
        output : pd.DataFrame

        """
        raise NotImplementedError


if __name__ == "__main__":
    pass
