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


def group_columns():
    columns_I=['iq0_1', 'id0_1', 'iq0c_1', 'id0c_1', 'iq0_2', 'id0_2', 'iq0c_2', 'id0c_2', 'iq0_3', 'id0_3', 'iq0c_3', 'id0c_3', 'iq0_4', 'id0_4', 'iq0c_4', 'id0c_4', 'iq0_5', 'id0_5', 'iq0c_5', 'id0c_5', 'iq0_6', 'id0_6', 'iq0c_6', 'id0c_6', 'iq0_7', 'id0_7', 'iq0c_7', 'id0c_7', 'idiffq0_mmc1', 'idiffd0_mmc1', 'idiffq0_c_mmc1', 'idiffd0_c_mmc1', 'isum0_mmc1', 'idiffq0_mmc2', 'idiffd0_mmc2', 'idiffq0_c_mmc2', 'idiffd0_c_mmc2', 'isum0_mmc2', 'idiffq0_mmc3', 'idiffd0_mmc3', 'idiffq0_c_mmc3', 'idiffd0_c_mmc3', 'isum0_mmc3', 'idiffq0_mmc4', 'idiffd0_mmc4', 'idiffq0_c_mmc4', 'idiffd0_c_mmc4', 'isum0_mmc4', 'idiffq0_mmc5', 'idiffd0_mmc5', 'idiffq0_c_mmc5', 'idiffd0_c_mmc5', 'isum0_mmc5', 'idiffq0_mmc6', 'idiffd0_mmc6', 'idiffq0_c_mmc6', 'idiffd0_c_mmc6', 'isum0_mmc6', 'iq0_th1', 'id0_th1', 'iq0_th2', 'id0_th2']
    columns_PQ=['Pth1', 'Pmmc1', 'Pg1', 'Pmmc2', 'Pmmc4', 'Pmmc3', 'Pl7', 'Pth2', 'Pmmc6', 'Pmmc5', 'Pg3', 'Pg2', 'Pl2', 'Pl5', 'Pl9', 'Qth1', 'Qmmc1', 'Qg1', 'Qmmc2', 'Qmmc4', 'Qmmc3', 'Ql7', 'Qth2', 'Qmmc6', 'Qmmc5', 'Qg3', 'Qg2', 'Ql2', 'Ql5', 'Ql9', 'P1dc', 'P2dc', 'P3dc', 'P4dc', 'P5dc', 'P6dc', 'P_1', 'Q_1', 'P_2', 'Q_2', 'P_3', 'Q_3', 'P_4', 'Q_4', 'P_5', 'Q_5', 'P_6', 'Q_6', 'P_7', 'Q_7']
    columns_V=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V1dc', 'V2dc', 'V3dc', 'V4dc', 'V5dc', 'V6dc', 'vq0_1', 'vd0_1', 'vq0c_1', 'vd0c_1', 'vq0_2', 'vd0_2', 'vq0c_2', 'vd0c_2', 'vq0_3', 'vd0_3', 'vq0c_3', 'vd0c_3', 'vq0_4', 'vd0_4', 'vq0c_4', 'vd0c_4', 'vq0_5', 'vd0_5', 'vq0c_5', 'vd0c_5', 'vq0_6', 'vd0_6', 'vq0c_6', 'vd0c_6', 'vq0_7', 'vd0_7', 'vq0c_7', 'vd0c_7', 'vnq0_mmc1', 'vnd0_mmc1', 'vnq0_c_mmc1', 'vnd0_c_mmc1', 'vdiffq0_mmc1', 'vdiffd0_mmc1', 'vdiffq0_c_mmc1', 'vdiffd0_c_mmc1', 'vsum0_mmc1', 'vDC0_mmc1', 'vnq0_mmc2', 'vnd0_mmc2', 'vnq0_c_mmc2', 'vnd0_c_mmc2', 'vdiffq0_mmc2', 'vdiffd0_mmc2', 'vdiffq0_c_mmc2', 'vdiffd0_c_mmc2', 'vsum0_mmc2', 'vDC0_mmc2', 'vnq0_mmc3', 'vnd0_mmc3', 'vnq0_c_mmc3', 'vnd0_c_mmc3', 'vdiffq0_mmc3', 'vdiffd0_mmc3', 'vdiffq0_c_mmc3', 'vdiffd0_c_mmc3', 'vsum0_mmc3', 'vDC0_mmc3', 'vnq0_mmc4', 'vnd0_mmc4', 'vnq0_c_mmc4', 'vnd0_c_mmc4', 'vdiffq0_mmc4', 'vdiffd0_mmc4', 'vdiffq0_c_mmc4', 'vdiffd0_c_mmc4', 'vsum0_mmc4', 'vDC0_mmc4', 'vnq0_mmc5', 'vnd0_mmc5', 'vnq0_c_mmc5', 'vnd0_c_mmc5', 'vdiffq0_mmc5', 'vdiffd0_mmc5', 'vdiffq0_c_mmc5', 'vdiffd0_c_mmc5', 'vsum0_mmc5', 'vDC0_mmc5', 'vnq0_mmc6', 'vnd0_mmc6', 'vnq0_c_mmc6', 'vnd0_c_mmc6', 'vdiffq0_mmc6', 'vdiffd0_mmc6', 'vdiffq0_c_mmc6', 'vdiffd0_c_mmc6', 'vsum0_mmc6', 'vDC0_mmc6', 'vq0_th1', 'vd0_th1', 'vq0_th2', 'vd0_th2']
    columns_vd=['vd0_1_pu', 'vd0_2_pu', 'vd0_3_pu', 'vd0_4_pu', 'vd0_5_pu', 'vd0_6_pu', 'vnd0_mmc1_pu', 'vdiffd0_mmc1_pu', 'vnd0_mmc2_pu', 'vdiffd0_mmc2_pu', 'vnd0_mmc3_pu', 'vdiffd0_mmc3_pu', 'vnd0_mmc4_pu', 'vdiffd0_mmc4_pu', 'vnd0_mmc6_pu', 'vdiffd0_mmc6_pu']
    columns_vq=['vq0_1_pu', 'vq0_2_pu', 'vq0c_2_pu', 'vq0_3_pu', 'vq0_4_pu', 'vq0c_4_pu', 'vq0_5_pu', 'vq0c_5_pu', 'vq0_6_pu', 'vq0_7_pu', 'vq0c_7_pu', 'vnq0_mmc1_pu', 'vdiffq0_mmc1_pu', 'vnq0_mmc2_pu', 'vdiffq0_mmc2_pu', 'vnq0_mmc3_pu', 'vdiffq0_mmc3_pu', 'vnq0_mmc4_pu', 'vdiffq0_mmc4_pu', 'vnq0_mmc5_pu', 'vnq0_mmc6_pu', 'vdiffq0_mmc6_pu']
    columns_id=['id0_1_pu', 'id0_2_pu', 'id0_3_pu', 'id0_4_pu', 'id0_5_pu', 'id0c_5_pu', 'id0_6_pu', 'id0_7_pu', 'idiffd0_mmc1_pu', 'idiffd0_mmc2_pu', 'idiffd0_mmc3_pu', 'idiffd0_mmc4_pu', 'idiffd0_mmc6_pu', 'id0_th1_pu', 'id0_th2_pu']
    columns_iq=['iq0_1_pu', 'iq0_2_pu', 'iq0_3_pu', 'iq0_4_pu', 'iq0_5_pu', 'iq0_6_pu', 'idiffq0_mmc1_pu', 'idiffq0_mmc2_pu', 'idiffq0_mmc3_pu', 'idiffq0_mmc4_pu', 'idiffq0_mmc6_pu', 'iq0_th1_pu', 'iq0_th2_pu']
    columns_P=['Pth1_pu', 'Pmmc1_pu', 'Pg1_pu', 'Pmmc2_pu', 'Pmmc4_pu', 'Pmmc3_pu', 'Pl7_pu', 'Pth2_pu', 'Pmmc6_pu', 'Pmmc5_pu', 'Pg3_pu', 'Pg2_pu', 'Pl2_pu', 'Pl5_pu', 'Pl9_pu', 'P1dc_pu', 'P2dc_pu', 'P3dc_pu', 'P4dc_pu', 'P5dc_pu', 'P6dc_pu', 'P_1_pu', 'P_2_pu', 'P_3_pu', 'P_4_pu', 'P_5_pu', 'P_6_pu', 'P_7_pu']
    columns_Q=['Qth1_pu', 'Qmmc1_pu', 'Qg1_pu', 'Qmmc2_pu', 'Qmmc4_pu', 'Qmmc3_pu', 'Ql7_pu', 'Qth2_pu', 'Qmmc6_pu', 'Qmmc5_pu', 'Qg3_pu', 'Qg2_pu', 'Ql2_pu', 'Ql5_pu', 'Ql9_pu', 'Q_1_pu', 'Q_2_pu', 'Q_3_pu', 'Q_4_pu', 'Q_5_pu', 'Q_6_pu', 'Q_7_pu']

    return columns_I, columns_PQ, columns_V, columns_vd, columns_vq, columns_id, columns_iq, columns_P, columns_Q

def data_clean_results():
    columns_remove={'vnd0_mmc4', 'V4dc', 'idiffd0_mmc6_pu', 'iq0_2', 'vdiffq0_mmc2_pu', 'vq0_th1_pu', 'vd0c_4_pu', 'id0c_6_pu', 'iq0c_1_pu', 'vDC0_mmc3', 'id0c_2_pu', 'vd0c_2', 'Qmmc3', 'vdiffd0_c_mmc1_pu', 'id0_7', 'V6dc_pu', 'idiffd0_c_mmc6_pu', 'vdiffd0_mmc5_pu', 'P_6', 'vq0c_7', 'vd0_th1', 'idiffd0_mmc4_pu', 'Qmmc4', 'vdiffq0_mmc2', 'vsum0_mmc2', 'idiffq0_c_mmc5', 'V2_pu', 'idiffq0_mmc2_pu', 'id0c_6', 'vd0_6_pu', 'P1dc', 'etheta0_2', 'vnd0_mmc5_pu', 'vdiffq0_c_mmc3_pu', 'vdiffd0_c_mmc3', 'isum0_mmc6', 'vdiffq0_c_mmc1_pu', 'Ql2', 'vq0_th1', 'etheta0_1', 'vq0c_1', 'vnd0_c_mmc5', 'iq0c_5', 'V1dc_pu', 'vd0c_5_pu', 'vq0c_1_pu', 'idiffd0_c_mmc5', 'P2dc', 'theta8', 'vDC0_mmc2', 'is_boundary', 'theta1', 'idiffd0_mmc3', 'vnq0_mmc1', 'vdiffd0_mmc1', 'vq0c_5', 'etheta0_3', 'Pmmc3', 'vDC0_mmc6', 'isum0_mmc4', 'V9_pu', 'vd0c_3_pu', 'vdiffq0_mmc4_pu', 'id0c_7_pu', 'vDC0_mmc4_pu', 'vq0_3', 'vdiffd0_c_mmc5_pu', 'iq0c_4', 'vdiffq0_mmc3_pu', 'P_2', 'vdiffd0_c_mmc6_pu', 'Pg3', 'vdiffq0_c_mmc4_pu', 'idiffq0_c_mmc6', 'etheta0_mmc6', 'idiffq0_c_mmc2', 'V3dc', 'vdiffq0_c_mmc4', 'vq0_6_pu', 'iq0_5', 'Qmmc5', 'vdiffd0_c_mmc4', 'iq0_2_pu', 'vd0c_6_pu', 'idiffd0_c_mmc2', 'iq0c_6', 'vq0c_4', 'vDC0_mmc4', 'vsum0_mmc6_pu', 'vd0_2_pu', 'V5dc_pu', 'vdiffq0_mmc4', 'id0c_3', 'idiffd0_mmc2_pu', 'vdiffd0_mmc3', 'P6dc', 'iq0c_7_pu', 'idiffq0_mmc1_pu', 'idiffd0_c_mmc6', 'Q_1', 'vdiffq0_mmc5', 'vd0_1', 'idiffq0_mmc1', 'iq0_6', 'idiffd0_mmc5', 'Pl5', 'vdiffd0_c_mmc2', 'vDC0_mmc3_pu', 'vnd0_mmc3', 'vnq0_c_mmc4', 'vdiffd0_mmc2', 'vdiffq0_c_mmc6_pu', 'iq0_3_pu', 'vdiffd0_c_mmc3_pu', 'vsum0_mmc4_pu', 'idiffd0_c_mmc2_pu', 'idiffq0_mmc5_pu', 'vd0c_1_pu', 'Qmmc6', 'vdiffq0_mmc5_pu', 'vsum0_mmc3_pu', 'id0_th2', 'V5', 'V4dc_pu', 'idiffq0_c_mmc3_pu', 'vsum0_mmc6', 'idiffq0_mmc4', 'vd0_3_pu', 'vnq0_c_mmc6_pu', 'vnq0_c_mmc2_pu', 'vd0c_7', 'vsum0_mmc2_pu', 'vnq0_mmc4', 'vd0c_3', 'vnd0_c_mmc6_pu', 'Qg3', 'vq0c_6', 'iq0_th1_pu', 'vd0_5_pu', 'vnd0_c_mmc3_pu', 'idiffd0_c_mmc3', 'vnq0_c_mmc4_pu', 'etheta0_mmc4', 'vdiffd0_c_mmc2_pu', 'vnq0_mmc3', 'idiffd0_mmc1_pu', 'etheta0_4', 'vd0_5', 'id0_7_pu', 'V2dc_pu', 'vdiffq0_c_mmc5', 'vnq0_mmc6_pu', 'vd0_1_pu', 'vdiffq0_c_mmc2', 'iq0c_7', 'vnd0_c_mmc2', 'vq0_3_pu', 'vDC0_mmc6_pu', 'vdiffq0_c_mmc2_pu', 'vDC0_mmc1', 'Q_3', 'etheta0_mmc1', 'idiffd0_mmc1', 'id0c_7', 'V3dc_pu', 'vsum0_mmc4', 'iq0c_3_pu', 'vnd0_mmc3_pu', 'vdiffd0_c_mmc6', 'iq0_4', 'iq0c_2', 'vnq0_mmc5', 'id0_6', 'iq0_th2_pu', 'V2', 'vnd0_c_mmc1', 'vd0_4_pu', 'id0_th1', 'iq0_5_pu', 'vd0c_6', 'P4dc', 'vsum0_mmc5_pu', 'vq0_6', 'vq0c_7_pu', 'vnd0_mmc1', 'idiffd0_mmc2', 'vq0c_2', 'vdiffq0_mmc6_pu', 'iq0_3', 'V7', 'V1dc', 'vq0_2', 'vd0_7_pu', 'etheta0_6', 'id0c_5', 'vdiffd0_mmc6', 'vdiffd0_mmc3_pu', 'idiffd0_c_mmc5_pu', 'vd0_6', 'iq0_7', 'idiffq0_mmc6', 'idiffq0_mmc5', 'idiffq0_mmc6_pu', 'Q_5', 'vq0c_2_pu', 'id0_4_pu', 'vdiffd0_c_mmc5', 'isum0_mmc5', 'vnd0_c_mmc3', 'vnq0_mmc6', 'vq0_2_pu', 'id0_1_pu', 'vsum0_mmc1_pu', 'idiffq0_mmc3_pu', 'iq0_1_pu', 'vnq0_c_mmc1_pu', 'vdiffd0_mmc2_pu', 'vnq0_c_mmc2', 'V5_pu', 'vnd0_mmc2_pu', 'Pl2', 'vnd0_mmc6', 'idiffq0_c_mmc3', 'id0_2', 'iq0c_1', 'vdiffd0_c_mmc1', 'Combination', 'V4', 'idiffd0_c_mmc4', 'vnq0_mmc2_pu', 'vnq0_mmc3_pu', 'vd0_th2', 'id0c_5_pu', 'vdiffq0_c_mmc6', 'vq0c_5_pu', 'vnd0_mmc6_pu', 'id0c_4_pu', 'Pth2', 'vq0_4_pu', 'etheta0_mmc2', 'vq0_th2_pu', 'V1', 'isum0_mmc1', 'vnd0_c_mmc1_pu', 'V8', 'vq0_th2', 'vdiffq0_c_mmc1', 'vnq0_c_mmc3_pu', 'vnq0_mmc1_pu', 'id0_5_pu', 'id0_th2_pu', 'id0_5', 'id0c_1', 'iq0_1', 'idiffq0_c_mmc6_pu', 'id0c_2', 'iq0_th2', 'vd0_7', 'vd0c_4', 'vnd0_c_mmc2_pu', 'vnq0_c_mmc1', 'vdiffq0_mmc3', 'idiffd0_c_mmc3_pu', 'id0_th1_pu', 'vnq0_c_mmc5', 'vd0_4', 'vd0c_1', 'V4_pu', 'id0c_3_pu', 'vnd0_c_mmc4_pu', 'Qmmc1', 'V2dc', 'idiffq0_c_mmc5_pu', 'idiffq0_mmc4_pu', 'Pl7', 'Pmmc6', 'P5dc', 'vq0c_3_pu', 'iq0c_6_pu', 'vdiffq0_c_mmc5_pu', 'Ql9', 'vsum0_mmc3', 'Pmmc5', 'vnd0_mmc2', 'id0_4', 'V11', 'idiffd0_c_mmc4_pu', 'vnd0_c_mmc5_pu', 'vq0_1', 'idiffd0_mmc3_pu', 'vnd0_c_mmc4', 'idiffq0_c_mmc4_pu', 'vdiffq0_mmc6', 'Pmmc4', 'etheta0_5', 'P_4', 'vq0_1_pu', 'vnq0_mmc5_pu', 'V10', 'Qg1', 'iq0c_5_pu', 'vd0c_7_pu', 'vnq0_c_mmc5_pu', 'Pg2', 'idiffq0_c_mmc1', 'idiffd0_c_mmc1', 'idiffq0_mmc2', 'Q_6', 'Pmmc2', 'vd0_2', 'vDC0_mmc5', 'vd0_th1_pu', 'vdiffq0_c_mmc3', 'id0_3_pu', 'P_3', 'vq0_5', 'idiffq0_c_mmc1_pu', 'vdiffq0_mmc1', 'vdiffd0_mmc4_pu', 'vnd0_mmc5', 'idiffd0_mmc6', 'idiffd0_c_mmc1_pu', 'vnq0_c_mmc3', 'id0_2_pu', 'iq0_7_pu', 'P_5', 'vdiffd0_mmc5', 'IPCF', 'idiffq0_c_mmc2_pu', 'vDC0_mmc2_pu', 'vq0_7', 'Powerflow', 'vd0c_5', 'vdiffd0_mmc4', 'V6dc', 'vnd0_mmc1_pu', 'vsum0_mmc1', 'Ql5', 'vd0_3', 'Qmmc2', 'vnq0_mmc4_pu', 'vdiffd0_mmc1_pu', 'theta11', 'vq0_7_pu', 'vDC0_mmc1_pu', 'P_1', 'V6', 'vdiffq0_mmc1_pu', 'vDC0_mmc5_pu', 'P3dc', 'id0_1', 'vsum0_mmc5', 'id0c_1_pu', 'iq0c_2_pu', 'vnd0_mmc4_pu', 'Q_2', 'etheta0_mmc3', 'idiffd0_mmc5_pu', 'vnd0_c_mmc6', 'vq0_5_pu', 'vdiffd0_mmc6_pu', 'Qth1', 'id0c_4', 'etheta0_7', 'iq0c_3', 'idiffq0_c_mmc4', 'vq0c_6_pu', 'vq0c_3', 'V9', 'Pth1', 'Qg2', 'Ql7', 'iq0_th1', 'vnq0_mmc2', 'id0_3', 'Qth2', 'iq0_6_pu', 'V5dc', 'Pmmc1', 'idiffd0_mmc4', 'vdiffd0_c_mmc4_pu', 'id0_6_pu', 'iq0c_4_pu', 'P_7', 'Pg1', 'etheta0_mmc5', 'vd0_th2_pu', 'Q_7', 'Q_4', 'Pl9', 'idiffq0_mmc3', 'V3', 'vnq0_c_mmc6', 'vq0_4', 'V10_pu', 'vq0c_4_pu', 'isum0_mmc2', 'isum0_mmc3', 'vd0c_2_pu', 'iq0_4_pu'}
    rows_remove=set()
    return columns_remove, rows_remove