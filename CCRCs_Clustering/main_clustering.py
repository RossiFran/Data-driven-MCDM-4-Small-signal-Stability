#%% IMPORTS
import numpy as np # matrix operations
import pandas as pd # data management

import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting
import matplotlib
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import axes3d


# machine learning
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.feature_selection import RFECV #RFE-CV
from sklearn.inspection import permutation_importance #PFI

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold, KFold
from sklearn.model_selection import cross_val_score

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, fbeta_score

from sklearn import tree

##
from metrics_n import *
from utils import *

from create_df import *
from clustering_functions import *
from data_preprocessing import *
#%%

plt.rcParams.update({"figure.figsize" : [8, 6],
                     "text.usetex": True,
                     "font.family": "serif",
                     "font.serif": "Computer Modern",
                     "axes.labelsize": 20,
                     "axes.titlesize": 20,
                     "legend.fontsize": 20,
                     "xtick.labelsize": 16,
                     "ytick.labelsize": 16,
                     "savefig.dpi": 130,
                    'legend.fontsize': 20,
                    'legend.handlelength': 2,
                    'legend.loc': 'upper right'})


#%%
#Read the list of CCRCs: 1: AC-GFM ; 2: DC-GFM; 3: GFL
exec(open('../Settings/combinations.sav').read())

indicators_list=['H2_vdc','DCgain_vdc','H2_freq','DCgain_freq']

indicators_plot_labels=dict()
indicators_plot_labels['H2_freq']=("$\mathcal{H}_{2,f}$")
indicators_plot_labels['H2_vdc']=("$\mathcal{H}_{2,V_{DC}}$")
indicators_plot_labels['DCgain_freq']=("$\mathcal{K}_{f}$")
indicators_plot_labels['DCgain_vdc']=("$\mathcal{K}_{V_{DC}}$")

#%%
path='./Results/'
path_data='./Data/'

#%%  ---- CREATE dataset with indicator values PFs x Combinations  ----
df_X_PF = pd.read_excel(path_data+'X_PF.xlsx')
df_X_IPC = pd.read_excel(path_data+'X_IPC.xlsx')

df_X_PF_IPC=pd.concat([df_X_PF,df_X_IPC],axis=1).reset_index(drop=True)

n_powerflows = len(df_X_PF)

#%% ---- DATA PREPROCESSING AND PCA ----
X = preprocess_data(df_X_PF_IPC)

n_intervals=5
df_X_PCA, x_list, y_list, z_list = PCA_cluster(X, n_intervals, plot=True)

#%%
for indicator in indicators_list:
    
    df_pf_ind = PFs_x_CCRCs(n_powerflows, indicator, combinations, path_data)
            
    #%%
    df_CCRs_PF_ind = df_1st_exploration(df_X_PF_IPC, np.arange(1,len(combinations)+1), combinations, n_powerflows, path_data, True)
    
    heatmap, CCRCs2clustering, df_pf_ind_levels = heatmap_function(n_intervals, x_list, y_list, z_list, indicator, 
                                                df_X_PCA, df_CCRs_PF_ind, df_pf_ind, 
                                                combinations, indicators_plot_labels)#, plot=True)
                        
    
    #%% ---- CCRCs CLUSTERING ACCORDING TO SINGLE INDICATOR ----
    if indicator == 'DCgain_vdc':
        seed=2
    elif indicator == 'H2_vdc':
        seed=3
    else:
        seed=42
    #seed=42
    labels, n_clusters = CCRCs_clustering(CCRCs2clustering,heatmap,seed)

    #%% ---- CCRCs CLUSTERS KNOWLEDGE EXTRACTION ----
    sorted_clusters_rules, df_CCRs_PF_ind_stable = knowledge_extraction(df_CCRs_PF_ind, labels, combinations, indicator)
   
    #%% ---- CCRCs CLUSTERS SELECTION ----
    selected_clusters, selected_clusters_unique = clusters_selection(df_CCRs_PF_ind_stable,sorted_clusters_rules,labels)

    #%% ---- WRITE RESULTS----
        
    writer = pd.ExcelWriter(path+"Clustering_results_"+indicator+".xlsx",
                            engine='xlsxwriter')
    
    if indicator=='Stab':
        labels.to_excel(writer, sheet_name='comb_clusters')
    else:
    
        sorted_clusters_rules.to_excel(writer, sheet_name='sorted_clusters_rules')
        selected_clusters.to_excel(writer, sheet_name='selected_clusters')
        labels.to_excel(writer, sheet_name='comb_clusters')
    
    writer.close()
    
    #%% ---- #REARRANGE ATTRIBUTE MATRIX ----
    
    complete_heat_map_rearranged = Rearranged_Attribute_Matrix(df_pf_ind_levels, heatmap, np.arange(0, n_clusters), labels, plot=True ,indicators_plot_labels=indicators_plot_labels, indicator=indicator, save_plot=True)
    reduced_heat_map_rearranged = Rearranged_Attribute_Matrix(df_pf_ind_levels, heatmap, selected_clusters_unique, labels, plot=True ,indicators_plot_labels=indicators_plot_labels, indicator=indicator, type_matrix='_reduced', save_plot=True)

#%% ---- APPLY SET INTERSECTION ----

CCRC_dict = set_intersection(n_powerflows, indicators_list, path, plot=True, save_plot = True)

# WRITE RESULTS    
f = open(path+"CCRC_dict.sav","w")
f.write( 'CCRC_dict='+str(CCRC_dict) )
f.close()

