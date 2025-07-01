# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 09:17:22 2022

@author: Sergi
"""

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
import warnings
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sn
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

cmap = cm.hot
cmap_reversed = cm.get_cmap('hot',100)

red = np.array([1,0,0,1])
yellow=np.array([1,1,0.3,1])
green=np.array([0,1,0,1])
white=np.array([1,1,1,1])

newcolors_stab = cmap_reversed(np.linspace(0, 1, 100))
newcolors_stab[:20, :] = white
# newcolors_stab[25:50, :] = red
# newcolors_stab[50:75,:] = yellow
newcolors_stab[80:,:] = green

newcmp_stab = ListedColormap(newcolors_stab)



def save_to_file(file, vals_dict):
    f = open(file, 'w')
    for v in vals_dict:
        f.write(v+'='+str(vals_dict[v])+'\n')
    f.close()


'''
meant for PFI_features.sav so it does not override info
'''
def update_file(file, vals_dict):

    lines_to_add = dict()
    for key in vals_dict.keys():
        lines_to_add[key] = '\''+key+'\': '+ str(vals_dict[key])+',\n'
        
    #read the file    
    f = open(file, 'r')
    f_content = f.read().split('\n')
    f.close()
    
    #update the file
    f = open(file, 'w')
    f.write(f_content[0]+'\n')
    
    for l in f_content[1:-1]:
        
        key = l.split(':')[0].replace('\'','')
        if key in lines_to_add.keys():
            f.write(lines_to_add.pop(key))
        else:
            f.write(l+'\n')
    
    for k in lines_to_add.keys():
        f.write(lines_to_add[k])
        
    f.write(f_content[-1])
    f.close()





'''
Plots a heatmap, intended to be used to plot combinations behaviour on the Grid of Grids project

Parameters:
    - df (pd.DataFrame()): Dataframe. 
        
    - comb_column (str): column containing the combination index
    - y (str): column with desired output
    - x (str): column with X variable
        Example:
            df: 'Stable' | 'Combination' | 'Pmmc1_pu'
                    1    |      8        |    1.23
                    0    |      2        |    0.95
                    ...
            comb_column = 'Combination'
            y = 'Stable'
            x = 'Pmmc1_pu'
    
    - n (int): number of divisions on x
    
    - combinations (list): combinations map to make the plot more readable
        Example: combinations = [[1,1,2], [1,2,1], [2,1,1], ...]
'''
def combination_heatmap(df, comb_column, y, x, n=10, combinations=None, sort=False):
    
    # generate basic heatmap
    x_list = np.linspace(df[x].min(), df[x].max(), n+1)
    
    n_comb = len(df[comb_column].unique())
    
    if combinations==None:
        combinations = np.linspace(0, n_comb-1, n_comb, dtype=int)
    
    heatmap = np.zeros((n_comb,len(x_list)-1))

    inan=[]
    for ic in range(n_comb):
        for ix in range(len(x_list)-1):
            Xmax = x_list[ix+1]
            Xmin = x_list[ix]
            indexes = (df[comb_column]==ic) & (df[x]<Xmax) & (df[x]>=Xmin)
            if np.isnan(np.average(df[y].loc[indexes])):
                inan.append([ic,ix])
                heatmap[ic][ix]= 0
            else:
                heatmap[ic][ix] = np.average(df[y].loc[indexes])
    
    # sort the heatmap (esto es un desproposito)
    if sort:
        sorted_heatmap = np.zeros((n_comb,len(x_list)-1))
        
        indices_max = []
        
        for ic, c in enumerate(heatmap):
            # index_of_max_c = np.where(c==np.amax(c))[0]
            # second_highest_val = sorted(list(set(c)))[-2]
            #                     #(combination, index of max value, number of max values, 2nd highest value)
            # indices_max.append( (ic, index_of_max_c[0], len(index_of_max_c), second_highest_val)
#             indices_c = np.where(c==np.amax(c))[0]
#             indices_max.append( (ic, indices_c[0], len(indices_c)) )          
            #c[np.where(c==-1)]=0
            indices_c=c.mean()
            indices_max.append([ic, indices_c])
            #indices_c = np.where(c==np.amax(c))[0]
            #indices_max.append( (ic, indices_c[0], len(indices_c)) )
        
#         indices_max = sorted(indices_max, key=lambda x: (x[1], -x[2], -x[3]))
#         indices_max = sorted(indices_max, key=lambda x: (x[1], -x[2]))
        indices_max = sorted(indices_max, key=lambda x: (-x[1]))
        
        for ii in range(0,len(inan)):
            heatmap[inan[ii][0],inan[ii][1]]=np.nan
        
        sorted_combinations = []
        sorted_combinations_names = []
        
        for i_tup in indices_max:
            i = i_tup[0]
            if i not in sorted_combinations:
                sorted_heatmap[len(sorted_combinations)] = heatmap[i]
                sorted_combinations.append(i)
                sorted_combinations_names.append(combinations[i])
        
        combinations = sorted_combinations_names
        heatmap = sorted_heatmap
    else:
        for ii in range(0,len(inan)):
            heatmap[inan[ii][0],inan[ii][1]]=np.nan
        
    return combinations, heatmap
        #del sorted_heatmap, sorted_combinations, sorted_combinations_names, indices_max
     
def plot_heatmap(sp,axs,heatmap,combinations,df,x,n=10):
    
    x_list = np.linspace(df[x].min(), df[x].max(), n+1)

    # plot the heatmap
    x_list_str = []
    for ix in range(len(x_list)-1):
        Xmin = x_list[ix]
        Xmax = x_list[ix+1]
        x_list_str.append('[{:.2f},{:.2f})'.format(Xmin,Xmax))
       
    #fig, axs = plt.subplots(3, 1)

    sns.heatmap(heatmap,
                xticklabels=x_list_str,
                yticklabels=combinations, 
                cmap='RdYlGn', annot=True, fmt=".2f", cbar=False,
                ax=axs[sp]).set_xlabel(x)
    plt.show()
    
# def combination_heatmap(df, comb_column, y, mmc1,mmc23, n=10, combinations=None, sort=False):
    
#     # generate basic heatmap
#     x_list = np.linspace(df[x].min(), df[x].max(), n+1)
    
#     n_comb = len(df[comb_column].unique())
    
#     if combinations==None:
#         combinations = np.linspace(0, n_comb-1, n_comb, dtype=int)
    
#     heatmap = np.zeros((n_comb,len(x_list)-1))

#     inan=[]
#     indixes_list=[]
#     for ic in range(n_comb):
#         for ix in range(len(x_list)-1):
#             Xmax = x_list[ix+1]
#             Xmin = x_list[ix]
#             indexes = (df[comb_column]==ic) & (df[mmc1]<Xmax) & (df[mmc1]>=Xmin)
#             indixes_list.append(indexes)
#             if np.isnan(np.average(df[y].loc[indexes])):
#                 inan.append([ic,ix])
#                 heatmap[ic][ix]= 0
#             else:
#                 heatmap[ic][ix] = np.average(df[y].loc[indexes])
    
# Sergi's function
# def combination_heatmap(df, comb_column, y, x, n=10, combinations=None, sort=False):
    
#     # generate basic heatmap
#     x_list = np.linspace(df[x].min(), df[x].max(), n+1)
    
#     n_comb = len(df[comb_column].unique())
    
#     if combinations==None:
#         combinations = np.linspace(0, n_comb-1, n_comb, dtype=int)
    
#     heatmap = np.zeros((n_comb,len(x_list)-1))

#     for ic in range(n_comb):
#         for ix in range(len(x_list)-1):
#             Xmax = x_list[ix+1]
#             Xmin = x_list[ix]
#             indexes = (df[comb_column]==ic) & (df[x]<Xmax) & (df[x]>=Xmin)
            
#             heatmap[ic][ix] = np.average(df[y].loc[indexes])

#     # sort the heatmap (esto es un desproposito)
#     if sort:
#         sorted_heatmap = np.zeros((n_comb,len(x_list)-1))
        
#         indices_max = []
        
#         for ic, c in enumerate(heatmap):
#             indices_c = np.where(c==np.amax(c))[0]
#             indices_max.append( (ic, indices_c[0], len(indices_c)) )          
        
#         indices_max = sorted(indices_max, key=lambda x: (x[1], -x[2]))
        
#         sorted_combinations = []
#         sorted_combinations_names = []
        
#         for i_tup in indices_max:
#             i = i_tup[0]
#             if i not in sorted_combinations:
#                 sorted_heatmap[len(sorted_combinations)] = heatmap[i]
#                 sorted_combinations.append(i)
#                 sorted_combinations_names.append(combinations[i])
        
#         combinations = sorted_combinations_names
#         heatmap = sorted_heatmap
#         del sorted_heatmap, sorted_combinations, sorted_combinations_names, indices_max
        
#     # plot the heatmap
#     x_list_str = []
#     for ix in range(len(x_list)-1):
#         Xmin = x_list[ix]
#         Xmax = x_list[ix+1]
#         x_list_str.append('[{:.2f},{:.2f})'.format(Xmin,Xmax))
       
#     plt.figure()

#     sns.heatmap(heatmap,
#                 xticklabels=x_list_str,
#                 yticklabels=combinations, 
#                 cmap='RdYlGn', annot=True, fmt=".2f", cbar=False
#                 ).set_xlabel(x)
#     plt.show()


'''
Returns a pd.Series indicating wether a data point is or is not a boundary of the dataset

Parameters:
    - df (pd.DataFrame): dataframe
    
    - ignore_columns (list): list of columns (str) to ignore when detecting boundaries
        Example: ignore_columns=['gender']

Returns:
    - is_boundary (pd.Series): pd.Series containing True/False values wether the data point is a boundary
        Example: df['is_boundary']=get_boundaries(df)
'''
def get_boundaries(df, ignore_columns=[]):
    # Create is_boundary label

    df_boundaries = (df == df.max()) | (df == df.min())

    # first check for "always boundary" cases (less than 3 different values):
    #    - only 1 value (min == max)
    #    - only 2 values (either min or max, no intermediate)

    for c in df:
        if df[c].unique().size < 3 and (c not in ignore_columns):
            ignore_columns.append(c)
        
    # Return pd.Series of [True, False] values
    # if one column is True then it is a boundary ("max" operates as an "or")
    return df_boundaries.drop(ignore_columns, axis=1).max(axis='columns') 





'''
Returns a list of correlated features in a df

Parameters:
    - df (pd.DataFrame): dataframe
    - c_threshold (float): threshold to consider correlation (if the coefficient is ge than the threshold the features are considered correlated)
    - method (str or callable): Method of correlation (see pandas.DataFrame.corr docs)

Returns:
    - correlated_features (list): list of tuples of correlated features
        Example: [('X1','X2'),('X2','X4')]
'''
def get_correlated_columns(df, c_threshold=0.999, method='pearson'):

    correlated_features = []
    correlation = df.corr(method=method)
    for i in correlation.index:
        for j in correlation:
            if i!=j and abs(correlation.loc[i,j])>=c_threshold:
                if tuple([j,i]) not in correlated_features:
                    correlated_features.append(tuple([i,j]))
                    
    return correlated_features





'''
==== DEPRECATED ====
'''

def define_boundaries(df, ignore_columns=[]):
    # Create is_boundary label
    warnings.warn('DeprecationWarning: define_boundaries(df, ignore_columns=[]) is deprecated in this version and will raise an exception in following versions. Use df[\'is_boundary\'] = get_boundaries(df) instead.')
    df_boundaries = (df == df.max()) | (df == df.min())

    # first check for "always boundary" cases (less than 3 different values):
    #    - only 1 value (min == max)
    #    - only 2 values (either min or max, no intermediate)

    for c in df:
        if df[c].unique().size < 3 and (c not in ignore_columns):
            ignore_columns.append(c)
        
    # Create the label
    # df['is_boundary'] = df_boundaries.drop(ignore_columns, axis=1).max(axis='columns') # if one column is True then it is a boundary ("max" operates as an "or")
    df['is_boundary'] = df_boundaries.drop(ignore_columns, axis=1).max(axis='columns') # if one column is True then it is a boundary ("max" operates as an "or")


def combination_heatmap_sorted(df, comb_column, y, x, n=10, combinations=None):
    
    x_list = np.linspace(df[x].min(), df[x].max(), n+1)
    warnings.warn('DeprecationWarning: combination_heatmap_sorted is deprecated in this version and will raise an exception in following versions. Use combination_heatmap(..., sort=True) instead.')
    
    n_comb = len(df[comb_column].unique())
    
    heatmap = np.zeros((n_comb,len(x_list)-1))

    for ic in range(n_comb):
        for ix in range(len(x_list)-1):
            Xmax = x_list[ix+1]
            Xmin = x_list[ix]
            indexes = (df[comb_column]==ic) & (df[x]<Xmax) & (df[x]>=Xmin)
            
            heatmap[ic][ix] = np.average(df[y].loc[indexes])


    sorted_heatmap = np.zeros((n_comb,len(x_list)-1))
    
    indices_max = []
    
    for ic, c in enumerate(heatmap):
        indices_c = np.where(c==np.amax(c))[0]
        indices_max.append( (ic, indices_c[0], len(indices_c)) )          
    
    indices_max = sorted(indices_max, key=lambda x: (x[1], -x[2]))
    
    sorted_combinations = []
    sorted_combinations_names = []
    
    for i_tup in indices_max:
        i = i_tup[0]
        if i not in sorted_combinations:
            sorted_heatmap[len(sorted_combinations)] = heatmap[i]
            sorted_combinations.append(i)
            sorted_combinations_names.append(combinations[i])

    x_list_str = []
    for ix in range(len(x_list)-1):
        Xmin = x_list[ix]
        Xmax = x_list[ix+1]
        x_list_str.append('[{:.2f},{:.2f})'.format(Xmin,Xmax))
       
    plt.figure()
    if combinations==None:
        combinations = np.linspace(0, n_comb-1, n_comb, dtype=int)

    sns.heatmap(sorted_heatmap,
                xticklabels=x_list_str,
                yticklabels=sorted_combinations_names, 
                cmap='RdYlGn', annot=True, fmt=".2f", cbar=False
                ).set_xlabel(x)
    plt.show() 






#%%

def create_bin_features(column, *df_list):
    uniquevals = set()
    for df in df_list:
        for val in sorted(df[column].unique().tolist()):
            uniquevals.update([val])
            
    for df in df_list:
        for val in sorted(uniquevals):
            df[column+'_'+str(val)] = df[column]==val     
    
# def print_df(df, consoleprint=False): # Display the df
#     if consoleprint:
#         print(final_df(df))
#     else:
#         display(final_df(df))
        
# def count_df_nan(df): # Return NaN count
#     return final_df(df).isnull().sum().sum()
    
# def print_df_nan(df, color='red'): # Display NaN values
#     display(final_df(df).loc[final_df(df).isnull().sum(axis=1)>0].style.highlight_null(null_color=color))

# def df_summary(df, w_type=True): # df summary
#     if w_type:
#         display(final_df(df).describe(include='all').append(final_df(df).dtypes, ignore_index=True).set_index(final_df(df).describe(include='all').index.append(pd.Index(['type']))))
#     else:
#         display(final_df(df).describe(include='all').set_index(final_df(df).describe(include='all').index))
        
def highlightMax(s): # df.style.apply(highlightMax).apply(highlightMin).highlight_null(null_color='red')
    isMax = s == s.max()
    return ['background-color: orange' if v else '' for v in isMax]

def highlightMin(s):
    isMin = s == s.min()
    return ['background-color: green' if v else '' for v in isMin]
