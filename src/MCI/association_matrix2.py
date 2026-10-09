from numpy import asarray
from pandas import DataFrame

import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()}, inplace=True)
from .MCI_assoc import MCI_assoc

def association_matrix2(df_A, df_B, assoc_function = MCI_assoc, 
                            axis = 0, title = None,
                            name_A = None,
                            name_B = None,
                            normalize_df_A = False,
                            normalize_df_B = False,
                            normalize_df_A_clims = [-3.0, 3.0],
                            normalize_df_B_clims = [-3.0, 3.0],
                            *args, **kwargs):
    """ 
    Computes an association matrix between two dataframes using a given association function. 

        Parameters: 
            #x (series): first vector.
            #y (series): second vector.
            # n_grid (int): the size of the grid where the kernel probability 
            #     densities are computed.
            df_A (DataFrame): first input dataframe.
            df_B (DataFrame): second input dataframe.
            assoc_function (function): function to compute the association.
            normalize_df_A_rows = False,
            normalize_df_B_rows = False,
            normalize_df_A_clims=[-3.0, 3.0],
            normalize_df_B_clims=[-3.0, 3.0],
            axis (0 or 1): axis to use as internal dimension to compute association across.
            title (str): title of the plot.
            name_A (str): name of the first dataframe for the plot.
            name_B (str): name of the second dataframe for the plot.

        Returns: 
            assoc_matrix (dataframe): association matrix.
        
    """
    
    if axis == 1:

        if normalize_df_A == True:
            D = df_A.T            
            D1 = ((D - D.mean(axis=0))/D.std(axis=0)).clip(lower = normalize_df_A_clims[0], upper = normalize_df_A_clims[1])
            df_A = D1.T

        if normalize_df_B == True:
            D = df_B.T            
            D1 = ((D - D.mean(axis=0))/D.std(axis=0)).clip(lower = normalize_df_B_clims[0], upper = normalize_df_B_clims[1])
            df_B = D1.T

        if assoc_function == signal_to_noise_GSEA4:

            scores = signal_to_noise_GSEA4(df_A.iloc[0,:].values, df_B.values)
            assoc_matrix = DataFrame(scores, index = df_B.index, columns = df_A.index).T
            
        elif assoc_function == diff_means2:

            scores = diff_means2(df_A.iloc[0,:].values, df_B.values)
            assoc_matrix = DataFrame(scores, index = df_B.index, columns = df_A.index).T

        else:

            assoc_matrix = DataFrame(0, index = df_A.index, columns = df_B.index)
            for a in df_A.index:
                for b in df_B.index: 
                    assoc_matrix.loc[a, b] = assoc_function(df_A.loc[a, :], df_B.loc[b, :], *args, **kwargs)
                 
    elif axis == 0:
        assoc_matrix = DataFrame(0, index = df_A.columns, columns = df_B.columns)
        for a in df_A.columns:
            for b in df_B.columns:
                x = df_A.loc[:, a]
                y = df_B.loc[:, b]
                assoc_matrix.loc[a, b] = assoc_function(x, y, *args, **kwargs)

    return assoc_matrix
