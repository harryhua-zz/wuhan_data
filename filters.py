import numpy as np
import pandas as pd

def trn_dev_ks(par):
    print("I'm in trn_dev_ks")
    print(par)


def filter(df, method, par):
    """
    This function acts as a universal entrance for the methods that filter rows or columns of a pandas dataframe.
    Arguments:
        df: A pandas dataframe.
        method: Name of the method to call.
        par: Parameters to be passed to the method, packed in a dictionary.
    Returns:
        A list containing the indices of the rows or columns that you want keep.
    """

    if (type(df) is not pd.core.frame.DataFrame) or (type(method) is not str) or (type(par) is not dict):
        print(__doc__)
        return []

    methodDict = {'trn_dev_ks': trn_dev_ks
                  }

    methodDict[method](par)


