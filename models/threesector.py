"""
Implementation of the two-sector energy-macroeconomics model

Usage is to `from models.threesector import ThreeSectorModel`
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Samuel Jazayeri"]

from itertools import product

import pandas as pd
import numpy as np
from tqdm import tqdm
from .model import Model


class ThreeSectorModel(Model):
    """ Version 1 of the three-sector model comprising energy, resources, and
    goods in a dynamic economy.

    Attributes
    ----------
    name: str
        Name of the model, such as "threesector". Used for filenames
    parameters : dict
        Dictionary of all parameters in vectorized form 
    hyper_parameters : dict
        Dictionary of all hyperparameters in vectorized form
    initial_values : dict
        Dictionary of all initial values in vectorized form
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
        self, parameters: dict, hyper_parameters: dict, initial_values: dict,
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
        super().__init__(
            parameters,
            hyper_parameters,
            initial_values,
            "threesectors",
            agent_ids=["E", "R", "G"],
        )

    def simulate(
        self,
        overwrite: bool = False,
        save: bool = False,
        sim_id: int = None,
        t_end: int = None,
        seed: int = None,
    ) -> pd.DataFrame:
        """Run a model simulation, and optionally save its output as a tsv file
        otherwise just return a DataFrame.

        Parameters
        ----------
        t_end : int, optional
            Total runtime of the model
        t_print : int, optional
            At which interval to save outputs
        seed : int, optional
            Random seed of the simulation
        seed_init : int, optional
            Random seed for the initial values of the simulation
        run_in_period : int, optional
            How many periods to cut off at the beginning of the simulation.
        save : str, optional
            filename to save the .txt output. 'mark0_covid' is prepended
        output_folder : str, default: 'output'
            where the output .txt is saved

        Returns
        -------
        output : pd.DataFrame
            time-series of the model simulation
        """
        # Adjust the hyperparameters if necessary
        if t_end:
            self.hyper_parameters["t_end"] = t_end
        if seed:
            self.hyper_parameters["seed"] = seed


        # Run the  model
        self._vectorize_params()
        self.output = self._simulate(overwrite, save)

        return self.output  # .loc[:, self.variables]

    def _vectorize_params(self):
        """Convert list items in the parameter dictionary to numpy arrays"""
        for key, par in self.parameters.items():
            if isinstance(par, list):
                self.parameters[key] = np.array(par)

    def _initialize_variables(self, n: int, t: int) -> dict:
        """Initialize all of the empty variables for the model

        Parameters
        ----------
        n : int
            Number of sectors in the model
        t : int
            Number of timesteps in the model

        Returns
        -------
        v : dict
        """

        # Initialize variables by their dimensions
        d1 = ["resources", "population", "wage", "income_hh", "deposits_hh", "cpi"]
        dn = [
            "prod_tgt",
            "expected_sales",
            "prod",
            "dem_labor",
            "labor",
            "capital",
            "debt",
            "prices",
            "income",
            "deposits",
            "dem_hh",
            "dem_hh_tgt",
            "cons_hh",
            "unallocated",
            "inflation",
        ]
        dnn = ["dem_interm", "dem_invest", "cons_interm", "investment", "inventory"]

        v = {}
        for i in d1:
            v[i] = np.zeros((t,))
        for i in dn:
            v[i] = np.zeros((t, n))
        for i in dnn:
            v[i] = np.zeros((t, n, n))
        return v

    def _set_initial_values(self, v: dict) -> dict:
        """Set the initial values of the different variables

        Parameters
        ----------
        v : dict
            dictionary of all variables in the model 

        Returns
        -------
        v : dict
        """
        for key, iv in self.initial_values.items():
            v[key][0] = iv
        return v

    def _compute_cpi(self, t: int, v: dict, p: dict, aux: dict):
        """Compute the consumer price inflation (CPI) based on prior period
        expenditure weights

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        """
        v["inflation"][t] = np.divide(v["prices"][t, :], v["prices"][t - 1, :]) - 1
        cons_dollar = v["cons_hh"][t - 1, :] * v["prices"][t - 1, :]
        mktshares = np.divide(cons_dollar, np.sum(cons_dollar))
        v["cpi"][t] = np.sum(np.multiply(mktshares, v["inflation"][t]))

    def _wage_update(self, t: int, v: dict, p: dict, aux: dict):
        """Update the wage levels based on inflation and labor market tightness

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        """
        part = np.sum(v["labor"][t - 1]) / v["population"][t - 1]
        multiplier = p["wage_part"] * (part - p["participation_nat"])
        multiplier += p["wage_infl"] * v["cpi"][t]
        v["wage"][t] = v["wage"][t - 1] * (1 + multiplier)

    def _production_targets(self, t: int, v: dict, p: dict, aux: dict):
        """ Update the targeted production for the different sectors

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        Returns
        -------
        v : dict

        """
        seen_dem = (
            v["dem_interm"][t - 1].sum(0)
            + v["dem_invest"][t - 1].sum(0)
            + v["dem_hh"][t - 1]
        )
        # print(f"seen dem{seen_dem}",seen_dem)
        memory = (aux["onen"] - p["tgtupdate"]) * v["prod_tgt"][t - 1]
        v["expected_sales"][t] = memory + p["tgtupdate"] * seen_dem
        # print(v["inventory"][t].diagonal(), v["expected_sales"][t-1])
        # print(v["inventory"][t].diagonal()- v["expected_sales"][t-1])
        expected_inventory = np.maximum(
            0, v["inventory"][t].diagonal() - v["expected_sales"][t - 1]
        )
        expected_inventory[0] = 0
        v["prod_tgt"][t] = np.maximum(
            0, p["inv_cov"] * v["expected_sales"][t] - expected_inventory
        )

    def _input_demands(self, t: int, v: dict, p: dict, aux: dict):
        """ Demands for intermediate goods and labor are formed 

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        Returns
        -------
        v : dict

        """
        # Demands for intermediate goods
        effective_tgt = np.minimum(p["kprod"] * v["capital"][t], v["prod_tgt"][t])
        v["dem_labor"][t] = effective_tgt / p["lprod"]

        temp = v["inventory"][t].copy()
        # Not necessarily the case i retains enough of its own for next period
        np.fill_diagonal(temp, 0)
        v["dem_interm"][t] = np.maximum(p["zmat"] * effective_tgt[:, None] - temp, 0)

    def _investment_demands(self, t: int, v: dict, p: dict, aux: dict):
        """ Demands for investment into capital

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        Returns
        -------
        v : dict

        """
        tgt_capital = v["prod_tgt"][t] / (p["kprod"] * p["cutgt"])
        v["dem_invest"][t, :, -1] = tgt_capital - (1 - p["deprate_k"]) * v["capital"][t]
        v["dem_invest"][t] = np.maximum(0, v["dem_invest"][t])

    def _labor_allocation(self, t: int, v: dict, p: dict, aux: dict):
        """ Clearing of the labor market

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        Returns
        -------
        v : dict

        """
        v["labor"][t] = v["dem_labor"][t] * np.minimum(
            1, v["population"][t] / np.sum(v["dem_labor"][t])
        )

    def _household_demand(self, t: int, v: dict, p: dict, aux: dict):
        """ Household income and demand formulation

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        Returns
        -------
        v : dict

        """

        v["income_hh"][t] = (
            v["wage"][t] * np.sum(v["labor"][t]) + p["interest"] * v["deposits_hh"][t]
        )

        budget = v["income_hh"][t] + v["deposits_hh"][t]
        inc_growth = (v["income_hh"][t] / v["income_hh"][t - 1]) - 1

        multiplier = (
            +1
            + (p["elast_price"] @ v["inflation"][t, :, None])[:, 0]
            + p["elast_income"] * inc_growth
        )

        v["dem_hh_tgt"][t] = multiplier * v["dem_hh_tgt"][t - 1]

        exp_cost = np.sum(v["dem_hh_tgt"][t] * v["prices"][t])
        v["dem_hh"][t] = v["dem_hh_tgt"][t] * np.minimum(1, budget / exp_cost)

    def _clear_markets(self, t: int, v: dict, p: dict, aux: dict):
        """ Allocate all of the existing inventory, produce energy, and
        allocate the energy amongst the sectors and household

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        Returns
        -------
        v : dict

        """
        hh_excess_dem = v["dem_hh"][t] - p["mincons"]
        avail = np.diag(v["inventory"][t]) - p["mincons"]
        avail = avail[1:]
        tot_dem = v["dem_interm"][t].sum(0) + hh_excess_dem
        tot_dem[0] -= v["dem_interm"][t, 0, 0]  # Energy sector supplies net only
        al = p["lprod"] * v["labor"][t]

        # Step 1: allocate non-energy items (excl first col)
        pdem = np.minimum(1, avail / tot_dem[1:])
        v["cons_interm"][t, :, 1:] = v["dem_interm"][t, :, 1:] * pdem
        v["cons_hh"][t, 1:] = p["mincons"][1:] + hh_excess_dem[1:] * pdem

        # Step 2: produce energy
        intermediate_budget = (
            v["cons_interm"][t]
            + np.triu(v["inventory"][t], 1)
            + np.tril(v["inventory"][t], -1)
        )
        intermediate_constraints = intermediate_budget / p["zmat"]
        intermediate_constraints[intermediate_budget == 0] = 0
        intermediate_constraints[p["zmat"] == 0] = np.inf

        output = np.min([al[0], np.min(intermediate_constraints[0, 1:])])
        v["prod"][t, 0] = (1 - p["zmat"][0, 0]) * output
        v["cons_interm"][t, 0, 0] = p["zmat"][0, 0] * output

        # Step 3: allocate energy
        pdem = np.minimum(1, (v["prod"][t, 0] - p["mincons"][0]) / tot_dem[0])
        v["cons_interm"][t, 1:, 0] = v["dem_interm"][t, 1:, 0] * pdem
        v["cons_hh"][t, 0] = p["mincons"][0] + hh_excess_dem[0] * pdem

        # Step 4: allocate investment
        avail = (
            v["inventory"][t].diagonal() - v["cons_interm"][t].sum(0) - v["cons_hh"][t]
        )
        avail[0] += output
        tot_dem = np.maximum(0, v["dem_invest"][t].sum(0))
        pdem = np.nan_to_num(np.minimum(1, avail / tot_dem))
        v["investment"][t] = v["dem_invest"][t] * pdem

    def _commodity_production(self, t: int, v: dict, p: dict, aux: dict):
        """ Determine financial performance of the sectors

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        Returns
        -------
        v : dict

        """
        intermediate_budget = (
            v["cons_interm"][t]
            + np.triu(v["inventory"][t], 1)
            + np.tril(v["inventory"][t], -1)
        )

        intermediate_constraints = intermediate_budget / p["zmat"]
        intermediate_constraints[p["zmat"] == 0] = np.inf

        v["prod"][t] = np.min(
            np.vstack(
                [p["lprod"] * v["labor"][t], np.min(intermediate_constraints, axis=1)]
            ).T,
            axis=1,
        )

        # Resource constraint
        v["prod"][t, 1] = max([0, min([v["resources"][t - 1], v["prod"][t, 1]])])

    def _update_inventory(self, t: int, v: dict, p: dict, aux: dict):
        """ Determine financial performance of the sectors

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        Returns
        -------
        v : dict

        """
        unsold = (
            v["inventory"][t].diagonal() - v["cons_interm"][t].sum(0) - v["cons_hh"][t]
        )

        # All inflows and outflows of items produced in the previous period
        v["inventory"][t + 1] = (
            # Inflows from inventory and purchases
            v["cons_interm"][t]
            + np.triu(v["inventory"][t], 1)
            + np.tril(v["inventory"][t], -1)
            # Inflow from unsold items
            + np.diag(unsold)
            # + np.diag(v["prod"][t])
            # Outflows from producing goods
            - np.multiply(p["zmat"], v["prod"][t, None].T)
        )

        # Adjust for inventory depreciation
        v["inventory"][t + 1] = np.multiply(p["deprate_inv"], v["inventory"][t + 1])

        # Add in new production
        np.fill_diagonal(
            v["inventory"][t + 1], v["inventory"][t + 1].diagonal() + v["prod"][t]
        )
        v["inventory"][t + 1, :, 0] = 0

    def _financial_performance(self, t: int, v: dict, p: dict, aux: dict):
        """ Determine financial performance of the sectors

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        Returns
        -------
        v : dict

        """
        sales = v["cons_interm"][t].sum(0) + v["cons_hh"][t] + v["investment"][t].sum(0)
        interm_purchases = (v["cons_interm"][t] * v["prices"][t]).sum(1)
        expenses = v["wage"][t] * v["labor"][t] + p["interest"] * v["debt"][t]
        v["income"][t] = v["prices"][t] * sales - expenses - interm_purchases

        invest_cost = (v["investment"][t] * v["prices"][t]).sum(1)
        temp = v["debt"][t] - v["deposits"][t] - v["income"][t] + invest_cost
        v["debt"][t + 1] = np.maximum(temp, 0)
        v["deposits"][t + 1] = np.maximum(-1 * temp, 0)

        expenses_hh = (v["cons_hh"][t] * v["prices"][t]).sum()
        v["deposits_hh"][t + 1] = v["deposits_hh"][t] + v["income_hh"][t] - expenses_hh

    def _capital_investment(self, t: int, v: dict, p: dict, aux: dict):
        """ Allocate the purchased goods to the capital stock of the sector

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        Returns
        -------
        v : dict

        """
        v["capital"][t + 1] = (1 - p["deprate_k"]) * v["capital"][t] + v["investment"][
            t
        ].sum(1)

    @staticmethod
    def _safe_divide(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Safely divide two vectors element-wise such that NaN and zeros
        are accounted for as follows:
            
        If the numerator is zero, then the result will be considered zero.
        If the denominator is zero, then the result will be infinity. 
        If both are zero, then the result is also set to zero

        Parameters
        ----------
        vec1 : np.ndarray
        vec2 : np.ndarray

        Returns
        -------
        vec : np.ndarray
        """
        vec = vec1 / vec2
        vec[vec2 == 0] = np.inf
        vec[vec1 == 0] = 0
        return vec

    def _price_updates(self, t: int, v: dict, p: dict, aux: dict):
        """ Update the prices based on a markup rule

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        Returns
        -------
        v : dict

        """
        quantity = v["prod"][t]  # + v["inventory"][t].diagonal()
        # print(quantity)
        # Unit costs
        int_cost = (p["zmat"] @ v["prices"][t, :, None])[:, 0]
        lab_cost = v["wage"][t] / p["lprod"]

        dep_cost = p["deprate_k"] * v["capital"][t] * v["prices"][t, -1]
        dep_cost = self._safe_divide(dep_cost, quantity)

        debt_cost = p["interest"] * v["debt"][t]
        debt_cost = self._safe_divide(debt_cost, quantity)

        v["prices"][t + 1] = p["markup"] * (int_cost + lab_cost + dep_cost + debt_cost)

    def _resource_update(self, t: int, v: dict, p: dict, aux: dict):
        """ Update the stock of resources in the environment

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        Returns
        -------
        v : dict

        """
        v["resources"][t] = v["resources"][t - 1] - v["prod"][t, 1]

    def _population_update(self, t: int, v: dict, p: dict, aux: dict):
        """ Update the population of citizens in the model

        Parameters
        ----------
        t : int
	    current period
        v : dict
            dictionary of variables	
        p : dict
            dictionary of parameters	
        aux : dict
            dictionary of helpful vectors

        Returns
        -------
        v : dict

        """
        v["population"][t + 1] = v["population"][t] * (1 + p["pop_grow"])

    def _run_simulation(self) -> pd.DataFrame:
        """Run the actual simulation of the model"""

        ###############################
        #  Initialization of Objects  #
        ###############################
        hp = self.hyper_parameters
        p = self.parameters

        v = self._initialize_variables(hp["n_sectors"], hp["t_end"])
        v = self._set_initial_values(v)

        # Manually move forward one
        v["prices"][1] = v["prices"][0]
        v["capital"][1] = v["capital"][0]
        v["population"][1] = v["population"][0]
        v["inventory"][1] = v["inventory"][0]

        # Auxiliaries
        aux = {
            "onen": np.ones((hp["n_sectors"],)),
            "zeron": np.zeros((hp["n_sectors"],)),
            "onenn": np.ones((hp["n_sectors"], hp["n_sectors"])),
            "zeronn": np.zeros((hp["n_sectors"], hp["n_sectors"])),
            "ixne": (slice(1, hp["n_sectors"]), slice(1, hp["n_sectors"])),
        }

        #######################
        #  Running the model  #
        #######################

        np.random.seed(hp["seed"])

        for t in tqdm(range(1, hp["t_end"] - 1)):

            # print(f"{50*'-'} t={t}")
            self._compute_cpi(t, v, p, aux)
            self._wage_update(t, v, p, aux)
            self._production_targets(t, v, p, aux)
            self._input_demands(t, v, p, aux)
            self._investment_demands(t, v, p, aux)
            self._labor_allocation(t, v, p, aux)
            self._household_demand(t, v, p, aux)
            self._clear_markets(t, v, p, aux)
            self._commodity_production(t, v, p, aux)
            self._update_inventory(t, v, p, aux)
            self._financial_performance(t, v, p, aux)
            self._capital_investment(t, v, p, aux)
            self._price_updates(t, v, p, aux)
            self._resource_update(t, v, p, aux)
            self._population_update(t, v, p, aux)
        ####################################
        #  Combining and Returning Output  #
        ####################################
        self.output = v

        cols = []
        for k, arr in self.output.items():
            if len(arr.shape) == 3:
                c = [f"{k}_{i}_{j}" for i, j in product(hp["sectors"], hp["sectors"])]
                cols.extend(c)
                self.output[k] = arr.reshape((hp["t_end"], hp["n_sectors"] ** 2))
            elif len(arr.shape) == 2 and arr.shape[1] > 1:
                cols.extend([f"{k}_{i}" for i in hp["sectors"]])
            else:
                cols.append(k)
                self.output[k] = arr[:, None]
        data = np.hstack([k for _, k in self.output.items()])
        return pd.DataFrame(data, columns=cols)


if __name__ == "__main__":
    pass
