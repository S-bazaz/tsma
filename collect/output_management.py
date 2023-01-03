# -*- coding: utf-8 -*-
"""
functions used to encode and decode parameters into databases, to store and manage simulation data

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
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm


#############################
#  path and SQL connection  #
#############################


def get_save_path(model, prev_folder: bool = False) -> str:
    """ Construct the path to the saved outputs
    
    Parameters
    ----------
    model: Model
    folder: str (default = 'outputs')
        
    Return
    ------
    save_path : string
    """

    model_name = model.name
    path = os.getcwd()
    if prev_folder:
        path = os.path.dirname(path)
    save_path = os.sep.join([path, "Outputs", model_name])
    return save_path


def create_connection(database_name: str, model) -> sqlite3.Connection:

    """ Return a connection to the database for a certain model
    
    Parameters
    ----------
    base_name : string,
      name of the database, ex parameters 
    model : Model
    
    Return
    ------
    sqlite3.Connection
    """
    path = os.sep.join([get_save_path(model), database_name + ".db"])
    return sqlite3.connect(path)


#############################
#  SQL script constructors  #
#############################


def list_to_tab_columns(column_names: list[str], primary_key: str):
    """ Gives the list of columns in the right format for table creation from a list
    
    Parameters
    ----------
    column_names : list[str],
      name of the columns 
        
    primary_key: str,
      name of the primary key
    
    Return
    ------
    res: string,
    
    """
    res = f"{primary_key } integer PRIMARY KEY "
    res += "".join([f", {p} float" for p in column_names])
    return res


def dict_to_tab_columns(dct: dict, primary_key: str) -> str:
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
    return list_to_tab_columns(list(dct.keys()), primary_key)


def create_table(connection: sqlite3.Connection, table: str, columns: str):
    """ Create a table in a given database
    
    Parameters
    ----------
    connection : sqlite3.Connection,
        linkage with the database
    table : string, 
        name of the table
    columns : string,
      list of columns in the right format for table creation
    
    no Return
    ---------
    
    """
    create_tab = f"CREATE TABLE IF NOT EXISTS {table} ( {columns} );"
    c = connection.cursor()
    c.execute(create_tab)


def dict_to_values_command(dct) -> str:
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
    res = "VALUES (  :"
    for key in dct:
        res += key + " ,:"
    res = res[:-2]
    res += ")"
    return res


def matrix_to_values_command(ncol: int) -> str:
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
    res = " VALUES ( " + ncol * "?, "
    res = res[:-2]
    res += " )"
    return res


def dict_to_where_command(dct: str) -> str:
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
    res = "WHERE ( "
    for key in dct:
        res += "( " + key + " = :" + key + ") AND "
    res = res[:-5]
    res += ")"
    return res


##########################################
#  final functions init, save and query  #
##########################################


def initialize_outputs(parameters: dict, outputcols: list, model):
    """Initialize the files and databases for the output of a new model.
    
    This function creates the directory structure and the databases
    for storing the output of a new model. It creates the following:
    
    -directory to store the output of the model, specified by model
    -subdirectory within the output directory to store figures
    -database to store the parameters used in the simulations,
        stored as a table with columns specified by parameters
    -database to store the output of the simulations,
        stored as a table with columns specified by outputcols
    
    The function does not return anything.
    
    Parameters
    ----------
    parameters : dict
        Encoded dictionary used to initialize the columns of the parameters database.
    outputcols : list of str
        Gives the columns to initialize the simulations database.
    model : Model
        Model object.
    
    No Return
    ------
"""

    # Create pathes
    save_path = get_save_path(model)
    fig_path = os.sep.join([save_path, "figures"])
    para_path = os.sep.join([save_path, "parameters.db"])
    sim_path = os.sep.join([save_path, "simulations.db"])

    # Create directory structure
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    # Create parameters database
    if not os.path.exists(para_path):
        connection = create_connection("parameters", model)
        tab_columns = dict_to_tab_columns(
            parameters, "sim_id"
        )  # sim_id is the primary key
        create_table(connection, "parameters_table", tab_columns)
        connection.commit()
        connection.close()
    # Create simulations database
    if not os.path.exists(sim_path):
        connection = create_connection("simulations", model)
        tab_columns = list_to_tab_columns(outputcols, "t")
        create_table(connection, "sinit", tab_columns)
        connection.commit()
        connection.close()


def get_id_parameters(params: dict, model) -> int:
    """ Give the id number the first simulation with certain parameters
    
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
    connection = create_connection("parameters", model)
    cursor = connection.cursor()
    cursor.execute(
        " SELECT sim_id FROM parameters_table " + dict_to_where_command(params), params
    )
    sim_id = cursor.fetchone()
    connection.close()
    return sim_id


def get_max_id(model) -> int:
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
    connection = create_connection("parameters", model)
    cursor = connection.cursor()
    cursor.execute("SELECT MAX(sim_id) FROM parameters_table")
    res = cursor.fetchone()
    if res is None:
        return None
    return res[0]


def save_parameters(params: dict, model, sim_id: int = None) -> int:
    """ Save a set of parameters in parameters.db and return the id of this new set
    
    Parameters
    ----------
    parameters : dict,
        encoded dictionary use to initialize the columns of parameters.db
    model : Model,
        model object
    sim_id : int, 
        optional value to custome the sim_id primary key
        !!!this option can lead to integrity error if the value is already taken!!!
        
    Return
    ------
    sim_id2: int,
        id number of the simulation for the set of parameters
    """
    connection = create_connection("parameters", model)
    cursor = connection.cursor()
    if sim_id is None:
        cursor.execute("SELECT MAX(sim_id) FROM parameters_table")
        res = cursor.fetchone()
        if res is None:  # init
            sim_id2 = 0
        elif res[0] is None:
            sim_id2 = 0
        else:
            sim_id2 = res[0] + 1
    else:
        sim_id2 = int(sim_id)
    params_aux = {"sim_id": sim_id2} | params
    cursor.execute(
        " INSERT INTO parameters_table " + dict_to_values_command(params_aux),
        params_aux,
    )

    connection.commit()
    connection.close()
    return sim_id2


def query_parameters(sim_id: str, model) -> pd.DataFrame:
    """ Give the set of parameters in parameters.db for a given simulation id and model
    
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
    connection = create_connection("parameters", model)
    df = pd.read_sql_query(
        "SELECT * FROM parameters_table WHERE sim_id = " + sim_id, connection
    )
    connection.close()
    df.drop("sim_id", axis=1, inplace=True)  # delete the existing table
    return df


def save_simulation(output: pd.DataFrame, sim_id: str, model):
    """ Save a simulation in Simulations.db for a given model and id 
    
    Parameters
    ----------
    output : pd.DataFrame,
        result of the simulation 
    sim_id : string,
        the id of the simulation
    model : model,
        to get the name 
    """
    connection = create_connection("simulations", model)
    cursor = connection.cursor()
    tab_columns = list_to_tab_columns(list(output), "t")
    create_table(connection, sim_id, tab_columns)

    output_matrix = output.values
    for t in range(len(output_matrix)):
        cursor.execute(
            " INSERT INTO "
            + sim_id
            + matrix_to_values_command(len(output_matrix[t]) + 1),
            [t] + list(output_matrix[t]),
        )
    connection.commit()
    connection.close()


def query_simulation(sim_id: str, model) -> pd.DataFrame:
    """ Give the output of a simulation for a given simulation id and model
    
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
    connection = create_connection("simulations", model)
    df = pd.read_sql_query("SELECT * FROM " + sim_id, connection)
    connection.close()
    df.drop("t", axis=1, inplace=True)

    return df


def delete_simulation(sim_id: str, model):
    """ Delete the table associate to a simulation
    
    Parameters
    ----------
    sim_id : string,
        the id of the simulation
    model : model,
        to get the name 
        
    no Return
    ------
    """
    connection = create_connection("simulations", model)
    cursor = connection.cursor()
    cursor.execute("DROP TABLE " + sim_id)
    connection.close()


###############################
#  multi-simulations queries  # 
###############################


def query_nparameters(model, nsim: int, sim_id0:int=0, step:int=1, t_end:int=500) -> pd.DataFrame():
    """
    Return a dataframe of `nsim` sets of parameters
    starting from `sim_id0` with a step of `step` and `t_end` time period.
    
    Parameters
    ----------
    model : Model
        A model object.
    nsim : int
        The number of sets of parameters to return.
    sim_id0 : int, optional
        The starting id number for the parameter sets, by default 0.
    step : int, optional
        The step size for the id numbers, by default 1.
    t_end : int, optional
        The time period for the parameter sets, by default 500.
    
    Returns
    -------
    pd.DataFrame
        A dataframe of `nsim` sets of parameters
        starting from `sim_id0` with a step of `step` and `t_end` time period.
    """
    connection = create_connection("parameters", model)
    df = pd.read_sql_query(
        f"SELECT * FROM parameters_table WHERE sim_id % {step} = 0 AND p2__t_end = {t_end} LIMIT {nsim} OFFSET {sim_id0}",
        connection,
    )
    connection.close()
    return df


def query_parameters_specific(model, sim_ids:list[int], t_end:int=500) -> pd.DataFrame():
    """Retrieve specific sets of model parameters from the parameters database.
    
    Parameters
    ----------
    model : Model
        model object
    sim_ids : list[int]
        List of simulation IDs for which to retrieve the parameters.
    t_end : int, optional
        Value of the `p2__t_end` parameter to filter the results by, by default 500.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the retrieved parameters.
    """
    connection = create_connection("parameters", model)
    lst_df = []
    for sim_id in sim_ids:
        df = pd.read_sql_query(
            f"SELECT * FROM parameters_table WHERE sim_id = {int(sim_id)} AND p2__t_end = {t_end}",
            connection,
        )
        lst_df.append(df)
    connection.close()
    return pd.concat(tuple(lst_df))


def query_simulations(model, sim_ids:list[int]) -> np.array:
    """
    Retrieve the simulation outputs for the given simulation IDs and model.
    
    Parameters
    ----------
    model : Model
        The model object.
    sim_ids : list[int]
        The list of simulation IDs.
    
    Returns
    -------
    outputs : np.array
        An array containing the simulation outputs.
    """
    nsim = len(sim_ids)
    sim_ids2 = "S" + np.array(sim_ids.astype(str), dtype=object)
    output0 = query_simulation(sim_ids2[0], model)
    t_end, nvar = np.shape(output0)
    outputs = np.zeros((t_end, nvar, nsim))
    outputs[:, :, 0] = output0.values
    for i in tqdm(range(1, nsim)):
        outputs[:, :, i] = query_simulation(sim_ids2[i], model).values
    return outputs


#################
#  CSV savings  #
#################


def save_temp_outputs(model, outputs, name: str = "temp"):
    """ Save the outputs of a model in a temporary location.
    
    Parameters
    ----------
    model : Model,
        model object
    outputs : ndarray,
        outputs of the model to be saved
    name : str, optional
        name of the file where to save the outputs (default is "temp")
        
    No Return
    ----------
    """
    save_path = get_save_path(model)
    shape_file = os.sep.join([save_path, f"{name}_shape.csv"])
    sim_file = os.sep.join([save_path, f"{name}_outputs.csv"])
    np.array(np.shape(outputs)).tofile(shape_file, sep=",")
    outputs.tofile(sim_file, sep=",")


def load_temp_outputs(model, name: str = "temp"):
    """Load temporary outputs from file.
   
   Parameters
   ----------
   model : Model,
       model object
   name : str, optional
       name of the file, by default "temp"
   
   Returns
   -------
   array-like
       loaded outputs
   """
    save_path = get_save_path(model)
    shape_file = os.sep.join([save_path, f"{name}_shape.csv"])
    sim_file = os.sep.join([save_path, f"{name}_outputs.csv"])
    shape = tuple(np.genfromtxt(shape_file, delimiter=",", dtype=int))
    return np.genfromtxt(sim_file, delimiter=",").reshape(shape)


def get_csv_path(model, name: str, prev_folder: bool = False):
    """
    Get the filepath to the CSV file with the given name for the given model.
    
    Parameters
    ----------
    model : Model
        Model object.
    name : str
        Name of the CSV file.
    prev_folder : bool, optional
        If True, the function will look for the file in the parent folder
        The default is False.
        
    Returns
    -------
    str
        Filepath to the CSV file.
    """
    save_path = get_save_path(model, prev_folder)
    return os.sep.join([save_path, f"{name}.csv"])


def save_df(model, df, name: str, prev_folder: bool = False):
    """
    Save a pandas DataFrame to a CSV file with the given name for the given model.
    
    Parameters
    ----------
    model : Model
        Model object.
    df : pandas DataFrame
        DataFrame to be saved.
    name : str
        Name of the CSV file.
    prev_folder : bool, optional
        If True, the function will look for the file in the parent folder
        The default is False.
        
    No Return
    ------
    """
    path_csv = get_csv_path(model, name, prev_folder)
    df.to_csv(path_csv, sep=";")


def load_df(model, name: str, prev_folder: bool = False):
    """
    Load a pandas DataFrame from a csv file.

    Parameters
    ----------
    model : Model,
        model object
    name : str,
        name of the csv file
    prev_folder : bool, optional
        If True, the function will look for the file in the parent folder
        The default is False
        
    Returns
    -------
    pandas.DataFrame
        dataframe loaded from the csv file
    """
    path_csv = get_csv_path(model, name, prev_folder)
    return pd.read_csv(path_csv, sep=";")


def save_clusters(model, df_clusters, name: str = "temp", prev_folder: bool = False):
    """
   Save a dataframe with the clusterings of simulations to a csv file
    
    Parameters
    ----------
    model: object of the Model class
    df_clusters: dataframe with columns 'sim_id' and 'label'
    name: str, 
        name to save the dataframe, default is 'temp'
    prev_folder: bool,
        If True, the function will look for the file in the parent folder
        The default is False
        
    No Return
    ------
    """
    save_df(model, df_clusters, f"{name}_clusters", prev_folder)


def load_clusters(model, name: str = "temp", prev_folder: bool = False):
    """
    Load a dataframe with the clusterings of simulations from a csv file
    
    Parameters
    ----------
    model: object of the Model class
    name: str, name of the dataframe to load, default is 'temp'
    prev_folder: bool, default False, 
        If True, the function will look for the file in the parent folder
        The default is False
        
    Return:
    ------
    dataframe with columns 'sim_id' and 'label'
    """
    df_clusters = load_df(model, f"{name}_clusters", prev_folder)
    df_clusters.index = df_clusters["sim_id"]
    return df_clusters.drop(columns=["sim_id"])


def save_jaccard(
    model, df_jaccard, df_jaccdist, name: str = "temp", prev_folder: bool = False
):
    """
    Save two dataframes with jaccard distances to csv files
    
    Input:
    
    model: object of the Model class
    df_jaccard: dataframe with columns 'sim_id1', 'sim_id2' and 'jaccard'
    df_jaccdist: dataframe with columns 'sim_id1', 'sim_id2' and 'jaccard_distance'
    name: str,
        name to save the dataframes, default is 'temp'
    prev_folder: bool,
        If True, the function will look for the file in the parent folder
        The default is False
    No return.
    """
    save_df(model, df_jaccard, f"{name}_jaccard", prev_folder)
    save_df(model, df_jaccdist, f"{name}_jaccdist", prev_folder)


def load_jaccard(model, name: str = "temp", prev_folder: bool = False) -> tuple:
    """Load Jaccard index and distance dataframes for the given model.
    
    Parameters
    ----------
    model : Model,
        model object
    name : str, optional
        name of the files to load, by default "temp"
    prev_folder : bool, optional
        If True, the function will look for the file in the parent folder
        The default is False
    Returns
    -------
    tuple
        Jaccard index dataframe, Jaccard distance dataframe
    """
    df_jaccard = load_df(model, f"{name}_jaccard", prev_folder)
    df_jaccdist = load_df(model, f"{name}_jaccdist", prev_folder)

    if "label" in list(df_jaccard):
        df_jaccard = df_jaccard.drop(columns="label")
        df_jaccdist = df_jaccdist.drop(columns="label")
    if "Unnamed: 0" in list(df_jaccard):
        df_jaccard = df_jaccard.drop(columns="Unnamed: 0")
        df_jaccdist = df_jaccdist.drop(columns="Unnamed: 0")
    df_jaccard.index = list(df_jaccard)
    df_jaccdist.index = list(df_jaccdist)
    return (
        df_jaccard,
        df_jaccdist,
    )
