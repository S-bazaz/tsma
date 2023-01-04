# -*- coding: utf-8 -*-
"""
functions manipulate strings : name conversions, title creations ...

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
import os
import sys
from numbers import Number

#################
#  Importation  #
#################

path = os.path.dirname(os.getcwd())
sys.path.append(path)
from basics.transfers import sep, join, list_to_dict, decompose_mainname

############
#  Pathes  #
############


def sdel(string: str, lst: list[str]) -> str:
    """
    Remove elements from a string
    
    Parameters
    string : str
    input string
    lst : list[str]
    list of elements to remove from string
    
    Returns
    s : str
    modified string with elements removed
    """
    s = string
    for x in lst:
        s = s.replace(x, "")
    return s


def replace_for_path(string: str) -> str:
    """Convert a string of instances the values of a dictionary 
    into a good format for creating paths.
    
    Parameters
    ----------
    string : string,
    Returns
    -------
    string : string,
    """
    to_del = [
        " ",
        "dict_values",
        "dict_keys",
        "array",
        "_",
        "'",
        ":",
        "\n",
        "{",
        "(",
        "}",
        ")",
    ]
    res = sdel(string, to_del).replace(",", "_")
    if res[0] == "[":
        res = res[1:]
    if res[-1] == "]":
        res = res[:-1]
    return res


############
#  colors  #
############


def name_to_rgb(name: str, c: int) -> str:
    """Create a rgb color from a name,
    in order to automatically associate a color to a variable if needed

    Parameters
    ----------
    name : string,
        name of the variable
    c: int,
        adjustment parameter

    Returns
    -------
    ... : string
    """
    st_hash = str(hash(name))
    if len(st_hash) < 10:
        st_hash = 10 * st_hash
    r = (int(st_hash[-3:]) + c) % 255
    g = (int(st_hash[-6:-3]) + c) % 255
    b = (int(st_hash[-9:-6]) + c) % 255
    return "rgb(" + str(r) + "," + str(g) + "," + str(b) + ")"


################
#  parameters  #
################


def names_to_index(names: list) -> list:
    """ Convert the list of sector names into a list of sector indices ex [E,R,G,Ri] 
    
    Parameters
    ----------
    hyper_parameters : dict,
        hyper_parameters of the simulation
        
    Returns
    -------
    ... : list[char] 
        ex [E,R,G]
    """
    len_order = np.argsort(np.vectorize(len)(names))
    res = np.repeat("__", len(names))
    for i in len_order:
        name = names[i]
        index = name[0].upper()
        count = 0
        while (index in res) and (count < len(name)):
            count += 1
            index += name[count]
        res[i] = index
    return res


def is_right_tensor(shape: tuple, dim_tgt: int):
    """Check if an array is a tensor of the given size

    Parameters
    ----------
    shape : list or tuple,
        the shape of the array
    dim_tgt : int,
        the dimension of a tensor side
        
    Returns
    -------
    result : bool,
        True if the array is a tensor of the given size, False otherwise
"""
    return (np.array(shape) == dim_tgt).all()


def aux_encode(target, f_id, f_type, dim, k, v):
    if len(dim) == 0:
        target[k] = f_type(v)
    else:
        for i in range(dim[0]):
            index = f_id(i)
            aux_encode(target, f_id, f_type, dim[1:], f"{k}_{index}", v[i])


def encode_dct(origin: dict, id_dict: str, agent_ids=[], is_int=False) -> dict:
    """
    Encode dictionary content with appropriate prefixes to recreate the original: p {origin_dictionary}

    Copy code
    Parameters
    ----------
    origin : dict
        Dictionary to encode
    id_dict : str
        Id of the originating dictionary (e.g. parameters, hyper_parameters,...)
    agent_ids : list[str]
        List of sector ids
    is_int : bool (default: False)
        Whether to encode numbers as integers or floats
    
    Returns
    -------
    dict
    """

    # Initialize empty target dictionary
    target = {}

    # Select the appropriate function to encode numbers based on the is_int parameter
    if is_int:
        f_type = lambda x: int(x) if isinstance(x, Number) else x
    else:
        f_type = lambda x: float(x) if isinstance(x, Number) else x
    # Iterate through key-value pairs in the origin dictionary
    for k, v in origin.items():
        # Determine the dimension of the value
        dimension = np.shape(v)

        # If the dimension matches the length of the agent_ids list, use the agent_ids list to generate prefixes
        if is_right_tensor(dimension, len(agent_ids)):
            f_id = lambda x: agent_ids[x]
        # Otherwise, use the index in the list to generate prefixes
        else:
            f_id = lambda x: x
        # If the value has a dimension of 0 (i.e. it is a scalar), add it to the target dictionary with a modified key
        if len(dimension) == 0:
            target[f"{id_dict}__{k}"] = f_type(v)
        # If the value has a higher dimension, call the aux_encode function to handle it
        else:
            aux_encode(target, f_id, f_type, dimension, f"{id_dict}__{k}_", v)
    # Return the encoded dictionary
    return target


def encode(
    parameters: dict, hyper_parameters: dict, initial_values: dict, agent_ids=[]
) -> dict:
    """ Convert parameters, hyper_parameters, initial_values into a
    single dictionary, the original dictionaries are encoded in the prefixes of the keys.
    
    Copy code
    Parameters
    ----------
    parameters: dict,
        Dictionary of parameters.
    hyper_parameters: dict,
        Dictionary of hyperparameters.
    initial_values: dict,
        Dictionary of initial values.
    agent_ids : list string,
        List of sector ids.
    
    Returns
    -------
    res : dict,
        Resulting dictionary with keys prefixed with 
        p1, p2, and p3 for parameters, hyperparameters, and initial values, respectively.
    """
    p = encode_dct(parameters, "p1", agent_ids)
    hp = encode_dct(hyper_parameters, "p2", agent_ids, is_int=True)
    iv = encode_dct(initial_values, "p3", agent_ids)
    return p | hp | iv


def convert(lst: list, dct: dict) -> np.array:
    """Convert a list of values to a list of keys in a dictionary.

    Parameters
    ----------
    lst : list,
        list of values to convert
    dct : dict,
        dictionary to use for conversion
        
    Returns
    -------
    list : list,
        list of keys in the dictionary
    """
    aux = lambda x: dct[x]
    return np.vectorize(aux)(lst)


def lst_to_coord(lst: list):
    """Convert a list of numbers to a tuple of integers.

    Parameters
    ----------
    lst : list,
        list of integers
        
    Returns
    -------
    tuple : tuple,
        tuple of integers
    """
    return tuple(np.vectorize(int)(lst))


def aux_decode(origin: dict, target: dict, dct_agent={}):
    """Decode a dictionary encoded using the prefixes of the keys
    to recreate the original dictionary.

    Parameters
    ----------
    origin: dict,
        Encoded dictionary
    target: dict,
        Dictionary where the decoded data will be stored
    dct_agent: dict,
        Dictionary used to map agent ids to indexes (optional)
    
    Returns
    -------
    No return, the target dictionary is updated in place.
    """
    n = len(dct_agent)
    temp = {}
    # Iterate over the items in the encoded dictionary
    for k, v in origin.items():
        lst = sep(k, 2)
        # If there are only two elements in the list, it means that the value is a scalar
        if len(lst) == 2:
            target[lst[0]][lst[1]] = v
        else:
            # If there are more elements in the list, it means that the value is an array object
            dct_id, key, sindex = tuple(lst)

            # Split the sindex to get the indexes of the value
            index = sep(sindex)
            if index[0] in dct_agent:
                # agent related
                if key not in target[dct_id]:
                    dim = [n] * len(index)
                    target[dct_id][key] = np.repeat(v, n ** len(index)).reshape(
                        tuple(dim)
                    )
                index = convert(index, dct_agent)
                target[dct_id][key][lst_to_coord(index)] = v
            else:
                index = list(np.vectorize(int)(index))
                if key not in temp:
                    temp[key] = [index, {sindex: v}, dct_id]
                else:
                    temp[key][0] = max(temp[key][0], index)
                    temp[key][1][sindex] = v
    for key, lst in temp.items():
        dct_id = lst[2]
        dim = np.array(lst[0])
        dim += 1

        for sindex, v in lst[1].items():
            if key not in target[dct_id]:
                target[dct_id][key] = np.repeat(v, np.prod(dim)).reshape(tuple(dim))
            index = np.vectorize(int)(sep(sindex))
            target[dct_id][key][tuple(index)] = v


def decode(origin: dict, agent_ids=[]) -> (dict, dict, dict):
    """Decode an encoded dictionary into the original dictionaries of
    parameters, hyper_parameters, and initial_values.

    Parameters
    ----------
    origin: dict,
        encoded dictionary
    agent_ids: list, optional
        List of agent ids. The default is an empty list.
        
    Returns
    -------
    parameters: dict
        Dictionary of parameters.
    hyper_parameters: dict
        Dictionary of hyperparameters.
    initial_values: dict
        Dictionary of initial values.
    """
    # Initialize target dictionaries for each type of parameter
    target = {"p1": {}, "p2": {}, "p3": {}}
    # Convert list of agent ids to dictionary for faster indexing
    dct_agent = list_to_dict(agent_ids)

    aux_decode(origin, target, dct_agent)
    # Return decoded dictionaries
    parameters = target["p1"]
    hyper_parameters = target["p2"]
    initial_values = target["p3"]

    return parameters, hyper_parameters, initial_values


###########
#  latex  #
###########


def add_index_and_exp(core: str, index: str, exponent: str):
    """ Add to a latex formula an index and an exponent
    
    Parameters
    ----------
    core : string,
        initial latex name
    index : string, 
        index to add
    exponent : string,
        exponent to add
        
    Returns
    -------
    res : string,
        
    """
    core = core.replace("$", "").replace(" ", "")
    # Add the index and exponent to the formula and 
    # enclose it in dollar signs to make it a latex formula
    res = "$" + core + "_{" + f"{index}" + "}^{" + f"{exponent}" + "}$"
    return res


def varname_to_latex(
    varname: str,
    exponents: list[str],
    indexes: list[str],
    core_conv: dict,
    special_exp: dict,
    special_index: dict,
):
    """ Convert the name of a parameter into the right latex formula  
    !!! To adapt the conversion, please change:
        core_conv,
        special_exp,
        special_index                             !!!
    
    Parameters
    ----------
    varname : str,
        initial name
    exponents : list[str],
        list of possible exponents
    indexes : list[str],
        list of possible indexes
    core_conv : dict,
        dictionary to convert the core of the name
    special_exp : dict,
        dictionary to add to the exponent in special cases
    special_index : dict,
        dictionary to add to the index in special cases
    
    Returns
    -------
    res : str,
        string corresponding to the right latex formula
    """

    # Separate the prefix and the main part of the name
    lst_dec = sep(varname, 2)
    main = lst_dec[1]
    core, exponent, index = decompose_mainname(main, exponents, indexes)
    res = core_conv[core] if core in core_conv else core

    # Check for special cases and modify the exponent and index accordingly
    if core in special_exp:
        exponent = special_exp[core] + " " + exponent
    if core in special_index:
        index = special_index[core] + " " + index
    if len(lst_dec) > 2:
        lst_coord = sep(lst_dec[2])
        index = index + " \ " + "-".join(lst_coord)
    return add_index_and_exp(res, index, exponent)


def gross_f_latex(varname: str) -> str:
    """Convert the name of a parameter into the right latex formula for the gross formular
    !!! To adapt the conversion, please change:
    core_conv,
    special_exp,
    special_index !!!
    
    Parameters
    ----------
    varname: str,
        initial name
    
    Returns
    -------
    result: str,
        the parameter in latex format
    """
    core_conv = {
        "n": "N",
        "margin": "\ mu",
        "cons propensity": "\ alpha",
        "norisk interest": "i",
        "capital ratio": "\mathcal{CAR}",
        "wages 0": "w",
        "wages 1": "w",
        "prices 0": "p",
        "prices 1": "p",
        "t end": "\mathcal{T}",
        "beta": "\ beta",
        "smooth interest": "\mathcal{Sm}",
    }
    exponents = ["hh", "f", "b", "tgt"]
    indexes = []

    special_exp = {}
    special_index = {
        "norisk interest": "D",
        "wages 0": "0",
        "wages 1": "1",
        "prices 0": "0",
        "prices 1": "1",
        "smooth interest": "i",
    }

    return varname_to_latex(
        varname, exponents, indexes, core_conv, special_exp, special_index
    )


################
#  html title  #
################


def list_to_lines(lst: list[str], ncol: int, sep: str = " ") -> str:
    """ Convert a list of string into a multiline string, 
    with a given number of columns
    
    Parameters
    ----------
    lst : list of str,
        list to transform
    ncol : int,
        number of columns
    sep : string, 
        separator for each column
    
    Returns
    -------
    res : string,
        multiline string
    """
    nrow = len(lst) // ncol + 1
    temp = []
    for i in range(nrow - 1):
        temp.append(sep.join(lst[i * ncol : (i + 1) * ncol]))
    temp.append(sep.join(lst[(nrow - 1) * ncol :]))
    return "<br>".join(temp)


def html_dec(x, dec: str, ndig=3):
    """
    Format the string representation of a number or other object as an HTML element.
    
    Parameters
    ----------
    x : Number or object
        The value to be formatted.
    dec : str
        The HTML tag to be used for formatting.
    ndig : int
        The number of decimal places to include when formatting numbers.
    
    Returns
    -------
    formatted_string : str
        The formatted string representation of x.
    """
    if isinstance(x, Number):
        s = str(round(x, ndig))
    else:
        s = str(x)
    if dec == "":
        return s
    else:
        return f"<{dec}>{s}</{dec}>"


def dict_to_html(
    dct: dict, sep: str = "", k_dec: str = "", v_dec: str = "i", ndig: int = 3
) -> str:
    """Convert a dictionary to a list of strings with the key-value pairs formatted as HTML.

    Parameters
    ----------
    dct : dict,
        dictionary to convert
    sep : str, optional
        separator between key and value, by default ''
    k_dec : str, optional
        decoration for the keys, by default ''
    v_dec : str, optional
        decoration for the values, by default 'i'
    ndig : int, optional
        number of decimal digits for float values, by default 3
    
    Returns
    -------
    """
    res = []
    for k, v in dct.items():
        temp = [html_dec(k, k_dec, ndig), ":", html_dec(v, v_dec, ndig)]
        res.append(sep.join(temp))
    return res


######################
#  clustering codes  #
######################


def decode_clust_approach(
    clust_code: str,
) -> tuple[dict[str, dict[str, list[str]]], str, int, str]:
    """
    Decode the clustering approach code and return a tuple containing the metric selections,
    the clustering algorithm, the number of clusters, and the session id.
    
    Parameters
    ----------
    clust_code: str
        The encoding of the clustering approach
        
    Returns
    -------
    metric_selections: Dict[str, Dict[str, List[str]]]
        The metric selections for each agent
    algo: str
        The clustering algorithm
    k: int
        The number of clusters
    session: str
        The session id
    """
    lst_code = clust_code.split("_")
    m = lst_code.pop(0)
    v = lst_code.pop(0)
    e = lst_code.pop(0)
    k = lst_code.pop(0)
    session = lst_code.pop(-1)
    algo = "\n".join(lst_code)
    metric_selections = {}
    metric_selections[m] = {v: [e]}
    return metric_selections, algo, int(k[1:]), session


def get_nclust(code: str) -> int:
    """
    Extract the number of clusters from the clustering code.
    
    Parameters:
    code (str): Clustering code in the format of "m_v_e_k_session"
                where m is the name of the metric, v is the name of the value, 
                e is the name of the evaluation, k is the number of clusters,
                and session is the session id.
    
    Returns:
    int: Number of clusters.
    """
    lst_code = sep(code)
    return int(lst_code[3][1:])


# time series visualization
def para_to_string(sim_ids: list, df, name: str) -> str:
    """Convert a dataframe of parameters for simulations to a string representation

    Parameters
    ----------
    sim_ids : list
        list of simulation ids
    df : pandas.DataFrame
        dataframe of parameters for simulations
    name : str
        name of the parameter set
    
    Returns
    -------
    str
        string representation of the parameter set for simulations
    """
    lst = [f"<b> {name} </b> <br>para : " + " ".join(list(df))]
    for k in range(len(sim_ids)):
        sim = sim_ids[k]
        if isinstance(sim, str):
            block = f"{sim} | "
            sim = [sim]
        else:
            block = f"{sim[0]} nsim: {len(sim)} | "
        para = np.round(df.loc[sim, :].mean(axis=0, numeric_only=True), 2)
        block = block + " ".join(map(str, list(para)))
        lst.append(block)
    return "<br>".join(lst)


# dendrogram
def clust_family_label(label: str) -> str:
    """
    Return the family label of a cluster label
    
    Parameters
    ----------
    label: str
        cluster label
    
    Returns
    family_label: str
        family label
    """
    return label.split("^")[0] + "$"


def get_clust_families(df_clusters2) -> dict:
    """
    Returns a dictionary of clusters families where each key is a cluster family label
    and the corresponding value is a list of cluster labels belonging to that family.
    
    Parameters
    ----------
    df_clusters2: Pandas DataFrame,
        DataFrame containing the cluster labels in its 'label' column.
    
    Returns
    -------
    clust_families: dict,
        Dictionary of clusters families.
    """
    clust_families = {}
    for label in df_clusters2["label"]:
        family = clust_family_label(label)
        if family not in clust_families:
            clust_families[family] = []
        clust_families[family].append(label)
    return clust_families


def family_to_figname(family: str) -> str:
    """Convert a family label to a figure name

    Parameters
    ----------
    family : str
        Family label
    
    Returns
    -------
    str
        Figure name
    """
    return sdel(family, ["$", "{", "}"]).replace(",", "_").replace(" ", "_")


# networks
def get_cluster_codes_and_label(code: str, g: str):
    """
    Returns the cluster ID, order code, and label for a given code and group.
    
    Parameters:
    ----------
    code (str): The code for the cluster.
    g (str): The group for the cluster.
    
    Returns:
    -------
        tuple: A tuple containing the cluster ID, order code, and label.
    """

    cluster_id = join([code, g])
    lst = sep(code)
    m, v, e, k = tuple(lst[:4])
    session = lst[-1]
    algo = join(lst[4:-1])

    ordercode = join([m, algo, v, e, k, session, g])

    algo_pseudo = " ".join(s[0:2] for s in lst[4:-1])
    label = "".join(
        [
            r"$",
            algo_pseudo,
            "_{",
            f"{m[1:]},{v[1:]},{e[1:]}",
            "}^{",
            f"{g[1:]}:{k[1:]}",
            "}$",
        ]
    )
    return cluster_id, ordercode, label


def decode_label(label: str) -> tuple[str, int, int, int, int]:
    """
    Decode the label of a cluster in order to extract its parameters.
    
    Parameters
    ----------
    label : str,
        Label of the cluster.
    
    Returns
    -------
    algo : str,
        Algorithm used to create the cluster.
    m : int,
        metrics
    v : int,
        variable selection
    e : int,
        embedding
    k : int,
        number of clsuters
    """

    lst = sep(sdel(label, ["$", "{", "}"]))
    algo = lst[0]
    lst = lst[1].split("^")
    m, v, e = tuple(lst[0].split(","))
    k = lst[1].split(":")[1]

    return algo, int(m), int(v), int(e), int(k)
