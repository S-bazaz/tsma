"""
Implementation of the Gross-macroeconomics model of 2022
https://doi.org/10.1016/j.ecolecon.2010.03.021
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
from tqdm import tqdm

#################
#  Importation  #
#################

path = os.path.dirname(os.getcwd())
sys.path.append(path)

from models.model import Model

###############
#  Gross2022  #
###############


class Gross2022(Model):
    """

    Attributes
    ----------
    name : str
        Name of the model, such as "threesector". Used for filenames
    parameters : dict
        Dictionary of all parameters in decoded form
    hyper_parameters : dict
        Dictionary of all hyperparameters in decoded form
    initial_values : dict
        Dictionary of all initial values in decoded form
    output : pd.DataFrame
        None, or the latest simulation run for given parameters
    output_micro : pd.DataFrame
        None, or time-series of unemployements, and firm's wages and prices
    debug : bool (default False)
        if true, prints step-wise calculations in the model

    Public Methods
    --------------
    simulate
        checks if simulation already exists, and executes a new simulation
        if necessary or desired
    load
        loads a simulation for the given set of parameters
    
    Private Methods
    ---------------
    _run_simulation
    
    Static Methods
    --------------
    _build_for_initialization
    _initialization
    _update_optional_var
    _update_wage
    _update_interest
    _update_wage_inflation
    _wage_payement_and_newloans
    _aux_hh_choose
    _update_hh_demand
    _update_prices_and_inflations
    _update_loans_and_capr
    _dividends_payements
    _annuity_payement_and_newloans
    _update_fcapr_interest_expense
    _creat_output_and_output_micro
    _fist_steps
    _run_simulation
    """

    def __init__(
        self, parameters: dict, hyper_parameters: dict, initial_values: dict,
    ):
        """Initialization of the model class

        parameters
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
        output_micro : pd.DataFrame
            None, or time-series of unemployements, and firm's wages and prices
        """
        super().__init__(parameters, hyper_parameters, initial_values, "gross2022")
        self.output_micro = None

    def simulate(
        self,
        overwrite: bool = False,
        save: bool = False,
        sim_id: int = None,
        t_end: int = None,
        seed: int = None,
    ) -> pd.DataFrame:
        """Run a model simulation, and optionally save its output and parameters in sql databases.

        parameters
        ----------
        t_end : int, optional
            Total runtime of the model
        seed : int, optional
            Random seed of the simulation
        save : bool, optional
            if you want to save the data
        overwrite: bool, optional
            if you want to replace existing data by this simulation

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
        self.output = self._simulate(overwrite, save, sim_id=sim_id)

        return self.output

    @staticmethod
    def _build_for_initialization(
        parameters: dict, hyper_parameters: dict, initial_values: dict, varnames: dict
    ) -> (tuple, dict, tuple, tuple):
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

        ### hyper parameters----------------------------------------
        t_end = int(hyper_parameters["t_end"])
        n_hh = int(hyper_parameters["hh_n"])
        n_f = int(hyper_parameters["f_n"])
        struct_para = (t_end, n_hh, n_f)
        ### storages creations--------------------------------------

        ## hh_data--------------------------------------------------
        # no initial cash
        hh_cash = np.zeros((t_end, n_hh))
        # households are fairly distributed to firms
        f_n_hh = np.empty(n_f)
        f_n_hh[: n_hh % n_f + 1] = n_hh // n_f + 1
        f_n_hh[n_hh % n_f :] = n_hh // n_f
        hh_id_f = np.arange(n_hh) % n_f
        # household data
        hh_data = (hh_cash, f_n_hh, hh_id_f)

        ## nohh_data--------------------------------------------------
        firms = np.zeros((t_end, n_f, len(varnames["firms"])))
        mainv = np.zeros((t_end, len(varnames["macro_main"])))
        optv = np.zeros((t_end, len(varnames["macro_optional"])))

        ## firms : firms' micro data
        # firms are active in the begging
        # 0: active
        firms[:, :, 0] = 1
        # they don't have any debt but also any money
        # 12: f_capr
        firms[:, :, 12] = 1
        # wages and prices are in the beginning fixed by parameters
        # 3 : wage
        firms[:, :, 3] = initial_values["wages_1"]
        firms[0, :, 3] = initial_values["wages_0"]
        # 7 : price
        firms[:, :, 7] = initial_values["prices_1"]
        firms[0, :, 7] = initial_values["prices_0"]

        ## mainv : non optional macro data
        # The minimum interest level is determined by parameters
        # 8 : interest
        mainv[:, 8] = parameters["norisk_interest"] + parameters["b_margin"]
        # not household data
        nohh_data = (firms, mainv, optv)

        return (struct_para, varnames, hh_data, nohh_data)

    @staticmethod
    def _initialization(
        parameters: dict, hyper_parameters: dict, initial_values: dict
    ) -> (tuple, dict, tuple, tuple):
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
                id of the employer for an houshold
        
        nohh_data : tuple,
            firms : (t_end, n_f, nb_micro_var) array,
                storage for the micro-variables
            mainv : (t_end,nb_used_macro_variable)
                storage for the macro-variables used in the core of the simulation
            optv : (t_end,nb_optional_macro_variable)
                storage for the optional macro-variables
        """
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
                "f_capr",  # 12 : firms' capital ratio
            ],
            "macro_main": [
                "mean_wage",  # 0
                "wage_inflation",  # 1
                "mean_price",  # 2
                "inflation",  # 3
                "loans",  # 4
                "b_lossnet",  # 5
                "PD",  # 6 : probability of default
                "LGD",  # 7 : loss giving default
                "interest",  # 8
                "b_networth",  # 9
                "capital_ratio",  # 10
                "dividends",  # 11
                "interest_expenses",  # 12
                "b_lossgross",  # 13
                "sPD",  # 14 : smoothed probability of default
                "sLGD",  # 15 : smoothed loss giving default
            ],
            "macro_optional": [
                "unemployment_rate",  # 0
                "n_active",  # 1
                "default_rate",  # 2
                "rollover",  # 3
                "full_debt_service",  # 4
                "GDP",  # 5
                "GDP_growth",  # 6
                "Debt_to_GDP",  # 7
                "loans_growth",  # 8
                "sum_wage",  # 9
                "f_profit_share_GDP",  # 10
                "w_share_GDP",  # 11
                "sum_f_cash",  # 12
                "firms_interest_coverage",  # 13
                "mean_inflation",  # 14
                "mean_wage_inflation",  # 15
                "count_newloans",  # 16
                "count_topups",  # 17
                "mean_demand",  # 18
                "std_demand",  # 19
                "std_price",  # 20
                "std_wage",  # 21
                "mean_f_cash",  # 22
                "std_f_cash",  # 23
                "mean_loan",  # 24
                "std_loan",  # 25
                "mean_f_capr",  # 26
                "mean_hh_cash",  # 27
                "std_hh_cash",  # 28
                "sum_hh_cash",  # 29
                "real_GDP",  # 30
                "real_wage",  # 31
                "real_GDP_growth",  # 32
                "unemployment_change",  # 33
            ],
        }

        return Gross2022._build_for_initialization(
            parameters, hyper_parameters, initial_values, varnames
        )

    @staticmethod
    def _update_optional_var(
        t: int, n_f: int, data: tuple, is_active: np.array, count_stats: tuple,
    ):
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
        is_active: np.array,
            tells if its an active firm
            
        count_stats: tuple,
            nb bankrupt,
            nb rolling,
            nb paid_annuity
            
        no Return
        -------
        """
        hh_cash, f_n_hh = data[0][:2]
        firms, mainv, optv = data[1]

        # 0 : unemployment_rate
        optv[t, 0] = (n_f - np.nansum(is_active)) / n_f
        # 1 : n_active
        optv[t, 1] = np.nansum(is_active)
        # 2 : default_rate
        optv[t, 2] = count_stats[0] / n_f
        # 3 : rollover
        optv[t, 3] = count_stats[1]
        # 4 : full_debt_service
        optv[t, 4] = count_stats[2]
        # GDP is the sum of the demand here
        # 5 : GDP #6 : demand
        optv[t, 5] = np.nansum(firms[t, is_active, 6])
        # 9 : sum_wage # 3: wage
        optv[t, 9] = np.nansum(firms[t, is_active, 3] * f_n_hh[is_active])

        if optv[t, 5] > 0:  # 5 : GDP

            # 7 : Debt_to_GDP # 5 : GDP # 4 : loans
            optv[t, 7] = mainv[t, 4] / optv[t, 5]
            # f_profit_share_GDP is the part of GDP taken by the firms
            # 10 : f_profit_share_GDP #9 : sum_wage # 5 : GDP
            optv[t, 10] = 1 - optv[t, 9] / optv[t, 5]
            # w_share_GDP is the part of GDP used to pay wages
            # 11 : w_share_GDP #9 : sum_wage # 5 : GDP
            optv[t, 11] = optv[t, 9] / optv[t, 5]
        else:
            # 10 : f_profit_share_GDP
            optv[t, 10] = 1
        # 12 : sum_f_cash #4 : f_cash
        optv[t, 12] = np.nansum(firms[t, is_active, 4])

        # 8: interest_expense
        temp = (firms[t - 1, :, 0] == 1) * (firms[t, :, 8] > 0) * is_active == 1
        # firms_interest_coverage is haow many time firms can pay their interest
        # 13 : firms_interest_coverage # 4 : f_cash # 8: interest_expense
        optv[t, 13] = np.nanmedian(firms[t - 1, temp, 4] / firms[t, temp, 8])

        # 14 : mean_inflation # 1 : inflation
        optv[t, 14] = np.nanmean(firms[t, is_active, 1])
        # 15 : mean_wage_inflation # 2 : wage_inflation
        optv[t, 15] = np.nanmean(firms[t, is_active, 2])
        # 16 : count_newloans # 9 : new_loan
        optv[t, 16] = np.nansum((firms[t, is_active, 9] > 0))
        # 17 : count_topups # 10 : top_up
        optv[t, 17] = np.nansum((firms[t, is_active, 10] > 0))
        # 18 : mean_demand  # 6 : demand
        optv[t, 18] = np.nanmean(firms[t, is_active, 6])
        # 19 : std_demand  # 6 : demand
        optv[t, 19] = np.nanstd(firms[t, is_active, 6])
        # 20 : std_price  # 7 : price
        optv[t, 20] = np.nanstd(firms[t, is_active, 7])
        # 21 : std_wage  # 3: wage
        optv[t, 21] = np.nanstd(firms[t, is_active, 3])
        # 22 : mean_f_cash  # 4 : f_cash
        optv[t, 22] = np.nanmean(firms[t, is_active, 4])
        # 23 : std_f_cash  # 4 : f_cash
        optv[t, 23] = np.nanstd(firms[t, is_active, 4])
        # 24 : mean_loan  # 5 : loan
        optv[t, 24] = np.nanmean(firms[t, is_active, 5])
        # 25 : std_loan  # 5 : loan
        optv[t, 25] = np.nanstd(firms[t, is_active, 5])

        # 27 : mean_hh_cash
        optv[t, 27] = np.nanmean(hh_cash[t, :])
        # 28 : std_hh_cash
        optv[t, 28] = np.nanstd(hh_cash[t, :])
        # 29 : sum_hh_cash
        optv[t, 29] = np.nansum(hh_cash[t, :])
        # Because the real production is fixed by the number of workers in a firm
        # real_GDP is given by the number of employed people
        # 30 : real_GDP
        optv[t, 30] = np.nansum(f_n_hh[is_active])
        # 31 : real_wage # 3 : wage # 7 : price
        optv[t, 31] = np.nansum(firms[t, is_active, 3]) / np.nansum(
            firms[t, is_active, 7]
        )

        # 33 : unemployment_change # 0 : unemployment
        optv[t, 33] = optv[t, 0] - optv[t - 1, 0]

        # growths:
        if t > 0:
            # 6 : GDP_growth # 5 : GDP
            optv[t, 6] = optv[t, 5] / optv[t - 1, 5]

            # 8 : loans_growth # 4 : loans
            optv[t, 8] = mainv[t, 4] / mainv[t - 1, 4] - 1

            # 32 : real_GDP_growth
            optv[t, 32] = optv[t, 30] / optv[t - 1, 30] - 1
        optv[0, 6] = np.nan  # 6 : GDP_growth
        optv[0, 8] = np.nan  # 8 : loans_growth
        optv[0, 32] = np.nan  # 32 : real_GDP_growth

    @staticmethod
    def _update_wage(t: int, mainv: np.array, firms: np.array, beta: int = 1):
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
        # Definition of two categories or active firms
        # 0: active
        is_elderactive = (firms[t, :, 0] * firms[t - 1, :, 0]) == 1
        is_newactive = (firms[t, :, 0] * (1 - is_elderactive)) == 1

        # 0: active
        bool_have_old_elder = any(firms[t - 1, :, 0] * firms[t - 2, :, 0])

        # aux_bool verifies if a firm who paid more than two wages exist
        # and if the inflation is high enough to increase wages
        # 3 : inflation  #1 : wage_inflation
        aux_bool = (
            bool_have_old_elder
            and mainv[t - 1, 3] > mainv[t - 1, 1]
            and mainv[t - 1, 3] > 0
        )

        ## new active
        # if the inflation is high enough and has a signification (there are elderactive firms)
        # newactive firms take the global inflation into account
        # 3 : inflation
        if aux_bool:
            new_wg = beta * mainv[t - 1, 3] + 1
        else:
            new_wg = 1
        # newactive firms also fix their wage based on the mean of previous wages
        # 0 : mean_wage
        if bool_have_old_elder:
            newactive_wage = mainv[t - 1, 0] * new_wg
        else:
            newactive_wage = mainv[t - 1, 0]
        # newactive wage and wage inflation update
        # 2 : wage_inflation
        firms[t, is_newactive, 2] = new_wg - 1
        # 3: wage
        firms[t, is_newactive, 3] = newactive_wage

        ## elder active
        # Because they were active, elderactive firms adapt their wage based on their own statistics
        # As before they consider the previous inflation and adapt their wage if it's high enough
        # 1 : inflation
        laged_pg = 1 + firms[t - 1, is_elderactive, 1]
        # 2 : wage_inflation
        laged_wg = 1 + firms[t - 1, is_elderactive, 2]
        elder_wg = np.maximum(laged_pg, 1)
        elder_wg[laged_pg <= laged_wg] = 1

        # adjustment(optional)
        elder_wg = 1 + beta * (elder_wg - 1)
        # elderactive wage and wage inflation update
        # 2 : wage_inflation
        firms[t, is_elderactive, 2] = elder_wg - 1
        # 3: wage
        firms[t, is_elderactive, 3] = firms[t - 1, is_elderactive, 3] * elder_wg

    @staticmethod
    def _update_interest(
        t: int, mainv: np.array, smooth_interest: int = 15
    ):
        """
        Calculate the default probability and LGD, then the interest rate
        
        definition
        -LGD : Losses Given Default

        parameters
        ----------
        t: int,
            step of the simulation
        mainv : (t_end,nb_used_macro_variable)
            storage for the macro-variables used in the core of the simulation
        smooth_interest : int,
            time considered for mean average
        no Return
        -------
        """
        # first to calculate the default probability,
        # the bank calculate the total debt it previously canceled
        # 13 : b_lossgross
        d_flow = mainv[t - 1, 13]

        # the default probability is then the percentage of the debt at t-2 canceled at t-1
        # with no debt there is no resik of bankrupt to PD = 0
        # 6 : PD # 4 : loans
        mainv[t, 6] = d_flow / mainv[t - 2, 4] if mainv[t - 2, 4] > 0 else 0

        # Losses Given Default is the percentage of the canceled loans that the bank really loose
        # because of the lack of collateral
        # 7 : LGD # 5 : b_lossnet
        mainv[t, 7] = mainv[t - 1, 5] / d_flow if d_flow > 0 else 0

        # Finally, the interest is calculated based on smoothed values of PD and LGD
        # 6 : PD
        spd = np.nanmean(mainv[max(t - smooth_interest, 0) : t, 6])
        # 7 : LGD
        slgd = np.nanmean(mainv[max(t - smooth_interest, 0) : t, 7])
        # 14 : sPD
        mainv[t, 14] = spd
        # 15 : sLGD
        mainv[t, 15] = slgd
        # 8 : interest
        mainv[t, 8] += spd * slgd / (1 - spd)
        # only positive values of interest are accepted
        # 8 : interest
        mainv[t, 8] = max(mainv[t, 8], 0)

    @staticmethod
    def _update_wage_inflation(
        t: int, mainv: np.array, firms: np.array, is_active: np.array
    ):
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

        # 0 : mean_wage #3: wage
        mainv[t, 0] = np.nanmean(firms[t, is_active, 3])

        if mainv[t - 1, 0] > 0:  # 0 : mean_wage
            # With no wages there is no wage inflation
            # 1 : wage_inflation # 0 : mean_wage
            mainv[t, 1] = (
                mainv[t, 0] / mainv[t - 1, 0] - 1 if mainv[t - 1, 0] > 0 else 0
            )

    @staticmethod
    def _wage_payement_and_newloans(
        t: int, firms: np.array, hh_data: tuple, is_active: np.array
    ):
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
        hh_cash, f_n_hh, hh_id_f = hh_data
        ## new loans to pay wages---------------------------------------------
        # Based on the wage bill and their cash firms ask for new loans
        # 9 : new_loan # 3: wage # 4 : f_cash
        firms[t, is_active, 9] = np.maximum(
            firms[t, is_active, 3] * f_n_hh[is_active]  # wage bill
            - firms[t - 1, is_active, 4],
            0,
        )
        # Here, all loans are granted
        # so firms loan are updated with the new loans
        # 5 : loan
        firms[t, is_active, 5] = firms[t - 1, is_active, 5] if t > 0 else 0
        # 5 : loan  #9 : new_loan
        firms[t, is_active, 5] += firms[t, is_active, 9]

        # Finally for the firms, the cash that remains after wage payements is calculated
        # 4 : f_cash  # 3: wage
        firms[t, is_active, 4] = np.maximum(
            firms[t - 1, is_active, 4] - firms[t, is_active, 3] * f_n_hh[is_active], 0,
        )
        ## wage payements------------------------------------------------------
        # On the other side, households recieve their wage,
        # so their cash is updated
        hh_cash[t, :] = hh_cash[t - 1, :] if t > 0 else 0
        # 3 : wage # 0 : active
        hh_cash[t, :] += firms[t, hh_id_f, 3] * firms[t, hh_id_f, 0]

    @staticmethod
    def _aux_hh_choose(
        i: int, aux_mat: np.array, hh_choices: np.array, hh_cash_t: np.array
    ):
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
        aux_mat[i, hh_choices[i]] = hh_cash_t[i]

    @staticmethod
    def _update_hh_demand(
        t: int,
        n_hh: int,
        n_f: int,
        firms: np.array,
        hh_cash: np.array,
        is_active: np.array,
        alpha: float,
    ):
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
        if is_active.sum() != 0:
            # first the index of the active firms are listed
            active_index = (np.arange(len(is_active)))[is_active]

            # The households choose a unique firm for their consumption
            # hh_choices = np.random.choice(lst, n_hh)
            hh_choices = active_index[
                np.random.randint(0, len(active_index), size=n_hh)
            ]

            # Then a matrix aux_mat is calculated
            # it gives with the positive values, the choice of every hh
            # but also their current budget with the coefficient
            aux_mat = np.zeros((n_hh, n_f))
            Gross2022._aux_hh_choose(
                np.arange(n_hh), aux_mat, hh_choices, hh_cash[t, :]
            )

            # Finally the demand of each firm can be computed
            # 6 : demand
            firms[t, :, 6] = alpha * (aux_mat.sum(axis=0))

    @staticmethod
    def _update_prices_and_inflations(
        t: int,
        mainv: np.array,
        firms: np.array,
        hh_cash: np.array,
        f_n_hh: np.array,
        alpha: float,
    ):
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
        if t > 1:
            # Because the production is given by the number of workers
            # the price is choosen in order to have the earning equal to the demand
            # 7 : price # 6 : demand
            firms[t, :, 7] = firms[t, :, 6] / f_n_hh

            # For those who had demand in t-1 and thus a positive price, the inflation is updated
            is_had_demand = firms[t - 1, :, 7] > 0
            # 1 : inflation # 7 : price
            firms[t, is_had_demand, 1] = (
                firms[t, is_had_demand, 7] / firms[t - 1, is_had_demand, 7]
            ) - 1
        # Then the mean of the price is computed, based on only on positive demand ie traded goods
        # Indeed, no demand mean no trades
        # 7 : price
        is_have_demand = firms[t, :, 7] > 0

        # 2 : mean_price # 7 : price
        mainv[t, 2] = np.nanmean(firms[t, is_have_demand, 7])

        # inflations at firm level are also updated
        # 3 : inflation # 2 : mean_price
        if t > 0:
            mainv[t, 3] = (mainv[t, 2] / mainv[t - 1, 2]) - 1
        # Finally, households spendings are recieved by the firms
        # hh consume alpha% of their cash
        hh_cash[t, :] *= 1 - alpha
        # firms earnings are their demand by definition of the price
        # 4 : f_cash # 6 : demand
        firms[t, :, 4] += firms[t, :, 6]

    @staticmethod
    def _update_loans_and_capr(
        t: int, mainv: np.array, firms: np.array, hh_cash: np.array
    ):
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
        # Total loans
        # 4 : loans # 5 : loan
        mainv[t, 4] = np.nansum(firms[t, :, 5])

        # Bank networth
        # 9 : b_networth # 4 : loans # 4 : f_cash
        mainv[t, 9] = mainv[t, 4] - np.nansum(firms[t, :, 4]) - np.nansum(hh_cash[t, :])

        # Bank capital ratio
        # 10 : capital_ratio # 9 : b_networth # 4 : loans
        mainv[t, 10] = mainv[t, 9] / mainv[t, 4] if mainv[t, 4] > 0 else 0

    @staticmethod
    def _dividends_payements(
        t: int, n_hh: int, mainv: np.array, hh_cash: np.array, tgt_capr: float
    ):
        """The bank distribute dividends based on its current capital ratio and the capital ratio targeted
        The Bank's networth is not updated directly.
        Indeed, the Bank's networth is only updated for the capital ratio calculation,
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
        # Sum of dividends computation
        # 11 : dividends # 10 : capital_ratio # 4 : loans
        mainv[t, 11] = max((mainv[t, 10] - tgt_capr) * mainv[t, 4], 0,)

        # Distribution of dividends
        # 11 : dividends
        hh_cash[t, :] += mainv[t, 11] / n_hh

    @staticmethod
    def _annuity_payement_and_newloans(
        t: int, t_end: int, mainv: np.array, firms: np.array, is_active: np.array
    ) -> (tuple, np.array, np.array):
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

        ## interests and annuities firms have to pay----------------------
        # 8 : interest #5 : loan
        interest_expense = mainv[t, 8] * firms[t, :, 5]
        annuity = (1 + mainv[t, 8]) * firms[t, :, 5]

        # 8 : interest_expense
        firms[t, :, 8] = interest_expense

        ## Situations of the firms-----------------------------------------
        # If a firms has enough cash it pays its all annuity.
        # If not it pays just the interest or goes bankrupt
        # 4 : f_cash
        is_bankrupt = (interest_expense > firms[t, :, 4]) * is_active
        is_rolling = (
            (interest_expense <= firms[t, :, 4])
            * (firms[t, :, 4] < annuity)
            * is_active
        )
        is_paid_annuity = (firms[t, :, 4] >= annuity) * is_active
        is_no_bankrupt = is_rolling + is_paid_annuity
        count_stats = (is_bankrupt.sum(), is_rolling.sum(), is_paid_annuity.sum())

        ## activity update---------------------------------------------------
        # firms that go bankrupt are inactive for the next step
        if t < t_end - 1:
            # 0 : active
            firms[t + 1, is_bankrupt, 0] = 0
            # 3 : wage
            firms[t + 1, is_bankrupt, 3] = 0
        ## Bank payements, loans and cash update----------------------------
        ## bankruptcy
        # The bank cancel the loan and the lost of that debt is given by b_lossgross
        # 13 : b_lossgross # 5 : loan
        mainv[t, 13] = (firms[t, is_bankrupt, 5]).sum()
        # But the bank seize the remaining cash of the firm in bankruptcy
        # so it looses only b_lossnet
        # 5 : b_lossnet  # 13 : b_lossgross #4 : f_cash
        mainv[t, 5] = mainv[t, 13] - firms[t, is_bankrupt, 4].sum()
        # the firm is then reinitialized
        # 4 : f_cash
        firms[t, is_bankrupt, 4] = 0
        # 5 : loan
        firms[t, is_bankrupt, 5] = 0

        ## rolling: pay interest and roll the rest
        # 4 : f_cash
        firms[t, is_rolling, 4] -= interest_expense[is_rolling]
        # 10 : top_up # 5 : loan
        firms[t, is_rolling, 10] = firms[t, is_rolling, 5]

        ## pay total annuity
        # 4 : f_cash #5 : loan
        firms[t, is_paid_annuity, 4] -= annuity[is_paid_annuity]
        # 5 : loan
        firms[t, is_paid_annuity, 5] = 0

        return count_stats, is_rolling, is_no_bankrupt

    @staticmethod
    def _update_fcapr_interest_expense(
        t: int,
        mainv: np.array,
        firms: np.array,
        optv: np.array,
        is_rolling: np.array,
        is_no_bankrupt: np.array,
    ):
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
        # Firms' networth
        # 11 : f_net_worth # 4 : f_cash # 5 : loan
        firms[t, is_no_bankrupt, 11] = (
            firms[t, is_no_bankrupt, 4] - firms[t, is_no_bankrupt, 5]
        )

        # Firms' capital ratio
        # 5 : loan
        is_borrowers = (firms[t, :, 5] > 0) * is_rolling == 1
        # 12: f_capr # 11 : f_net_worth #5 : loan
        firms[t, is_borrowers, 12] = (
            firms[t, is_borrowers, 11] / firms[t, is_borrowers, 5]
        )
        # Mean of the capital ratios
        if sum(is_borrowers) > 0:
            # 26 : mean_f_capr # 12 : f_capr
            optv[t, 26] = np.nanmean(firms[t, is_borrowers, 12])
        # Sum of interest expenses
        # 12 interest_expenses # 8 :interest_expense
        mainv[t, 12] = np.nansum(firms[t, :, 8])

    @staticmethod
    def _creat_output_and_output_micro(
        t_end: int,
        n_f: int,
        mainv: np.array,
        firms: np.array,
        optv: np.array,
        varnames: dict,
    ) -> (pd.DataFrame, pd.DataFrame):
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
        ## create output by leaving micro variables----------------------------
        output = np.concatenate((mainv, optv), axis=1)
        colnames = varnames["macro_main"] + varnames["macro_optional"]
        output = pd.DataFrame(output, columns=colnames)

        ## create micro_output, a dataframe that gather the unemployment rate
        # and firms' inflation and wage inflation
        # It's Useful for Philips curve computation
        # 1 : inflation # 2 : wage_inflation
        output_micro = firms[:, :, [1, 2]]
        output_micro = np.column_stack(
            (np.repeat(np.arange(t_end), n_f), output_micro.reshape(t_end * n_f, -1))
        )
        output_micro = pd.DataFrame(
            output_micro.copy(), columns=["t", "inflation", "wage_inflation"],
        )
        output_micro = pd.merge(
            output_micro.copy(),
            (output["unemployment_rate"]).copy(),
            left_on="t",
            right_index=True,
        )
        return output, output_micro

    @staticmethod
    def _fist_steps(
        t_end: int,
        n_hh: int,
        n_f: int,
        hh_data: tuple,
        nohh_data: tuple,
        alpha: float,
        tgt_capr: float,
    ):
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
        (hh_cash, f_n_hh, hh_id_f) = hh_data
        (firms, mainv, optv) = nohh_data

        # t = 0
        is_active = firms[0, :, 0] == 1
        Gross2022._wage_payement_and_newloans(0, firms, hh_data, is_active)
        Gross2022._update_hh_demand(0, n_hh, n_f, firms, hh_cash, is_active, alpha)
        Gross2022._update_prices_and_inflations(0, mainv, firms, hh_cash, f_n_hh, alpha)
        Gross2022._update_loans_and_capr(0, mainv, firms, hh_cash)
        Gross2022._dividends_payements(0, n_hh, mainv, hh_cash, tgt_capr)
        (
            count_stats,
            is_rolling,
            is_no_bankrupt,
        ) = Gross2022._annuity_payement_and_newloans(0, t_end, mainv, firms, is_active)
        Gross2022._update_fcapr_interest_expense(
            0, mainv, firms, optv, is_rolling, is_no_bankrupt
        )
        Gross2022._update_optional_var(
            0, n_f, (hh_data, nohh_data), is_active, count_stats
        )

        # t = 1
        is_active = firms[1, :, 0] == 1
        Gross2022._update_wage_inflation(1, mainv, firms, is_active)
        Gross2022._wage_payement_and_newloans(1, firms, hh_data, is_active)
        Gross2022._update_hh_demand(1, n_hh, n_f, firms, hh_cash, is_active, alpha)

        Gross2022._update_prices_and_inflations(1, mainv, firms, hh_cash, f_n_hh, alpha)
        Gross2022._update_loans_and_capr(1, mainv, firms, hh_cash)
        Gross2022._dividends_payements(1, n_hh, mainv, hh_cash, tgt_capr)

        (
            count_stats,
            is_rolling,
            is_no_bankrupt,
        ) = Gross2022._annuity_payement_and_newloans(1, t_end, mainv, firms, is_active)
        Gross2022._update_fcapr_interest_expense(
            1, mainv, firms, optv, is_rolling, is_no_bankrupt
        )
        Gross2022._update_optional_var(
            1, n_f, (hh_data, nohh_data), is_active, count_stats
        )

    def _run_simulation(self) -> pd.DataFrame:
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

        ## functions importation ------------------------------------------------
        parameters = self.parameters
        hyper_parameters = self.hyper_parameters
        initial_values = self.initial_values
        update_wage = Gross2022._update_wage
        update_interest = Gross2022._update_interest
        update_wage_inflation = Gross2022._update_wage_inflation
        wage_payement_and_newloans = Gross2022._wage_payement_and_newloans
        update_hh_demand = Gross2022._update_hh_demand
        update_prices_and_inflations = Gross2022._update_prices_and_inflations
        update_loans_and_capr = Gross2022._update_loans_and_capr
        dividends_payements = Gross2022._dividends_payements
        annuity_payement_and_newloans = Gross2022._annuity_payement_and_newloans
        update_fcapr_interest_expense = Gross2022._update_fcapr_interest_expense
        update_optional_var = Gross2022._update_optional_var
        creat_output_and_output_micro = Gross2022._creat_output_and_output_micro

        ## initialization ------------------------------------------------------

        ((t_end, n_hh, n_f), varnames, hh_data, nohh_data,) = Gross2022._initialization(
            parameters, hyper_parameters, initial_values
        )

        firms, mainv, optv = nohh_data
        hh_cash, f_n_hh = hh_data[:2]

        ## importation of other parameters-------------------------------------------
        alpha = parameters["hh_cons_propensity"]
        tgt_capr = parameters["tgt_capital_ratio"]
        smooth_interest = int(parameters["smooth_interest"])
        beta = parameters["beta"]
        np.random.seed(hyper_parameters["seed"])

        Gross2022._fist_steps(
            t_end, n_hh, n_f, hh_data, nohh_data, alpha, tgt_capr,
        )

        ## Core of the simulation------------------------------------------------------
        for t in tqdm(range(2, t_end)):  # tqdm(range(t_end)) , range(t_end)

            ## list of active firms--------------------------------------------------
            # 0: active
            is_active = firms[t, :, 0] == 1

            update_wage(t, mainv, firms, beta)

            update_interest(t, mainv, smooth_interest)

            update_wage_inflation(t, mainv, firms, is_active)

            ## firms pay wages and contract new loans if they don't have enough cash--
            wage_payement_and_newloans(t, firms, hh_data, is_active)

            update_hh_demand(t, n_hh, n_f, firms, hh_cash, is_active, alpha)

            update_prices_and_inflations(t, mainv, firms, hh_cash, f_n_hh, alpha)

            update_loans_and_capr(t, mainv, firms, hh_cash)

            dividends_payements(t, n_hh, mainv, hh_cash, tgt_capr)

            count_stats, is_rolling, is_no_bankrupt = annuity_payement_and_newloans(
                t, t_end, mainv, firms, is_active
            )
            update_fcapr_interest_expense(
                t, mainv, firms, optv, is_rolling, is_no_bankrupt
            )

            update_optional_var(t, n_f, (hh_data, nohh_data), is_active, count_stats)
        output, self.output_micro = creat_output_and_output_micro(
            t_end, n_f, mainv, firms, optv, varnames
        )
        return output

    @staticmethod
    def _printvar(name, var, t: int = None):
        try:
            if len(var.shape) > 1:
                var = f"\n{var}"
        except AttributeError:
            pass
        if t is not None:
            print("{:20} ({}): {}".format(name, t, var))
        else:
            print("{:20}: {}".format(name, var))

    @staticmethod
    def _noprintvar(name, var, t=None):
        pass


if __name__ == "__main__":
    pass
