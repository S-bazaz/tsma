# -*- coding: utf-8 -*-
"""
functions used to divide and join strings, lists and dictionaries

"""
__author__ = ["Samuel Bazaz"]
__credits__ = ["Samuel Bazaz"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Samuel Bazaz"]


##############
#  Packages  #
##############

from copy import deepcopy
from numbers import Number

##########
#  List  #
##########

def intersection(lst1: list, lst2: list) -> list:
    """
    Get the intersection of two lists in the order of the first one.
    
    Parameters:
        lst1 (list): The first list.
        lst2 (list): The second list.
        
    Returns:
        res (list): The intersection of the two lists.
    """
    res = []
    for x in lst1:
        if x in lst2:
            res.append(x)
    return res


def transfert_x_lst_to_lst(lst1: list, lst2: list, x) -> (list, list):
    """
    Transfer all occurrences of x from lst1 to lst2.
    
    Parameters:
        lst1 (list): The first list.
        lst2 (list): The second list.
        x: The element to transfer.
        
    Returns:
        res (tuple): A tuple containing the modified lst1 and lst2.
    """
    lst2p = deepcopy(lst2)
    lst2p.extend([x] * lst1.count(x))
    return list(filter(lambda a: a != x, lst1)), lst2p


def transfert_List_to_List(lst1: list, lst2: list, lst: list) -> (list, list):
    """
    Transfer all occurrences of elements in lst from lst1 to lst2.
    
    Parameters:
        lst1 (list): The first list.
        lst2 (list): The second list.
        lst (list): The list of elements to transfer.
        
    Returns:
        res (tuple): A tuple containing the modified lst1 and lst2.
    """
    lst1p, lst2p = deepcopy(lst1), deepcopy(lst2)
    for x in lst:
        lst1p, lst2p = transfert_x_lst_to_lst(lst1p, lst2p, x)
    return lst1p, lst2p


def inverse_and_concate(lst_lst: list[list]) -> list:
    """
    Invert and concatenate a list of lists.
    
    Parameters:
        lst_lst (list of lists): The list of lists to invert and concatenate.
        
    Returns:
        lst_res (list): The inverted and concatenated list.
    """
    lst_lst2 = deepcopy(lst_lst)
    lst_lst2.reverse()
    lst_res = []
    for lst in lst_lst2:
        lst_res += lst
    return lst_res


def add_seeds_to_list(nseed: int, lst: list) -> list:
    """
    Add seeds to each element in a list.
    
    Parameters:
        nseed (int): The number of seeds to add.
        lst (list): The list of elements to add seeds to.
        
    Returns:
        res (list): The list with seeds added to each element.
    """
    res = []
    for s in lst:
        for k in range(nseed + 1):
            res.append(s + k)
    return res


##########
#  Dict  #
##########


def sub_dict(original: dict, keys: list, inside: bool = True) -> dict:
    """
   Create a sub dictionary with the given keys.
   
   Parameters:
       original (dict): The initial dictionary.
       keys (list): The list of keys to filter the dictionary by.
       inside (bool, optional): If True, keep only the keys in `keys`.
       If False, keep all keys except those in `keys`. Defaults to True.
       
   Returns:
       res (dict): The filtered dictionary.
   """
    if inside:
        return {k: v for k, v in original.items() if k in keys}
    return {k: v for k, v in original.items() if k not in keys}


def list_to_dict(lst: list) -> dict:
    """
    Convert a list to a dictionary, with the list elements as keys and their indices as values.
    
    Parameters:
        lst (list): The list to convert.
        
    Returns:
        res (dict): The resulting dictionary.
    """
    res = {}
    for i in range(len(lst)):
        res[lst[i]] = i
    return res


def numerical_sub_dict(dct: dict) -> dict:
    """
    Create a sub dictionary from a dictionary, 
    containing only the key-value pairs with numerical values.
    
    Parameters:
        dct (dict): The dictionary to filter.
        
    Returns:
        res (dict): The resulting sub dictionary.
    """
    res = {}
    for k, v in dct.items():
        if isinstance(v, Number):

            res[k] = v
    return res


def update_dct_from_list(dct: dict, lst: list) -> dict:
    """
   Update the values in a dictionary from a list.
   
   Parameters:
       dct (dict): The dictionary to update.
       lst (list): The list to use for updating the dictionary values.
       
   Returns:
       dct (dict): The updated dictionary.
   """
    for i, k in enumerate(dct):
        dct[k] = lst[i]
    return dct


def update_dct_from_dct(dct1: dict, dct2: dict) -> dict:
    """
    Update the values in a dictionary from another dictionary.
    
    Parameters:
        dct1 (dict): The dictionary to update.
        dct2 (dict): The dictionary to use for updating the values.
        
    Returns:
        dct1 (dict): The updated dictionary.
    """
    for k, v in dct2.items():
        if k in dct1:
            dct1[k] = dct2[k]
    return dct1


############
#  string  #
############


def join(lst: list, n: int = 1) -> str:
    """
   Join the elements of a list with a delimiter.
   
   Parameters:
       lst (list): The list to join.
       n (int, optional): The number of delimiters to use. Defaults to 1.
       
   Returns:
       res (str): The resulting string.
   """
    return ("_" * n).join(lst)


def sep(s: str, n: int = 1) -> list:
    """
    Split a string into a list using a delimiter.
    
    Parameters:
        s (str): The string to split.
        n (int, optional): The number of delimiters to use. Defaults to 1.
        
    Returns:
        res (list): The resulting list.
    """
    return s.split("_" * n)


####################
#  decompositions  #
####################


def separate_suffix(varname: str, lst_suffix: list[str]) -> (list[str], list[str]):
    """
    Decompose a variable's name into two lists: prefix and suffix.
    The prefix is associated with the type of variable,
    and the suffix is composed only of elements from the list `lst_suffix`.
    
    Parameters:
        varname (str): The name of the variable.
        lst_suffix (list[str]): The list of suffixes to use for decomposition.
        
    Returns:
        prefix, suffix (list[str], list[str]): The resulting prefix and suffix lists.
    """
    lst_dec = sep(varname)
    inter = intersection(lst_dec, lst_suffix)
    prefix, suffix = transfert_List_to_List(lst_dec, [], inter)

    if prefix == []:
        return suffix, []
    return prefix, suffix


def decompose_mainname(
    mainname: str, exponents: list[str], indexes: list[str]
) -> (str, str, str):
    """
    Decompose the main name of a variable into the core name, exponent, and index.
    
    Parameters:
        mainname (str): The initial name of the variable.
        exponents (list[str]): The list of exponents to use for decomposition.
        indexes (list[str]): The list of indexes to use for decomposition.
        
    Returns:
        core (str): The core name of the variable.
        exponent (str): The exponent of the variable.
        index (str): The index of the variable.
    """
    lst = sep(mainname)
    lst, exponent = transfert_List_to_List(lst, [], exponents)
    core, indexe = transfert_List_to_List(lst, [], indexes)
    return " ".join(core), ",".join(exponent), ",".join(indexe)


#################
#  positioning  #
#################


def position_inputs(params: dict, ncol: int, ndico: int = 3) -> list[list[str]]:
    """
    Organize a list of input parameters into a grid with a given number of columns.
    
    Parameters:
        params (dict): The input parameters to organize.
        ncol (int): The number of columns in the grid.
        ndico (int): The number of dictionaries in the grid.
        
    Returns:
        grid (list[list[str]]): The grid of input parameters.
    """
    lst_sort = [[[]] for j in range(ndico)]
    for v in list(params):
        i_dict = int(v[1]) - 1
        lst_sort[i_dict][0].append(v)
    f = lambda lst: [lst[k::ncol] for k in range(ncol)]
    return [f(inverse_and_concate(lst)) for lst in lst_sort]


def add_for_groups_dict(varname: str, lst_suffix: list[str], dct_dct: dict[dict[str]]):
    """
    Add a variable to a group in a dictionary based on its name and suffix.
    
    Parameters:
        varname (str): The name of the variable to add.
        lst_suffix (list[str]): A list of possible suffixes for the variable.
        dct_dct (dict[dict[str]]): The dictionary of groups to add the variable to.
        
    Returns:
        None.
    """
    lst_dec, suffix = separate_suffix(varname, lst_suffix)
    dim = len(suffix)
    groupname = " ".join(lst_dec) if dim < 2 else " ".join(lst_dec + suffix[:-1])
    if groupname not in dct_dct[dim]:
        dct_dct[dim][groupname] = [varname]
    else:
        dct_dct[dim][groupname].append(varname)


def auto_dct_groups(varnames: list[str], lst_suffix: list[str]) -> dict:
    """ Group a list of variables into a dictionary based on their suffix and original dimensions.
    
    Parameters
    ----------
    varnames : list of str,
        list of variable names
        
    lst_suffix : list of str,
        ex: ["energy", "resources", "goods"]
        
    Returns
    -------
    dct_groups : dict,
        grouping of the variables
    """
    dct_dct = {0: {}, 1: {}, 2: {}}
    for v in varnames:
        add_for_groups_dict(v, lst_suffix, dct_dct)
    dct_groups = dct_dct[2] | dct_dct[1] | dct_dct[0]

    return dct_groups
