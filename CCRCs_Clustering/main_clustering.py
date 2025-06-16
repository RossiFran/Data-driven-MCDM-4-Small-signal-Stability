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
#TODO complete:
#Read the list of CCRCs: 1:  ; 2:   ; 3:
exec(open('../Settings/combinations.sav').read())

indicators_list=['H2_freq', 'H2_vdc','DCgain_freq','DCgain_vdc']

indicators_plot_labels=dict()
indicators_plot_labels['H2_freq']=("$\mathcal{H}_{2,f}$")
indicators_plot_labels['H2_vdc']=("$\mathcal{H}_{2,V_{DC}}$")
indicators_plot_labels['DCgain_freq']=("$\mathcal{K}_{f}$")
indicators_plot_labels['DCgain_vdc']=("$\mathcal{K}_{V_{DC}}$")

#%%
path='./Results/'
path_data='./Data_Generation/Data/'

#%%  ---- CREATE dataset with indicator values PFs x Combinations  ----
df_X_PF = pd.read_excel(path_data+'X_PF.xlsx')
df_X_IPC = pd.read_excel(path_data+'X_IPC.xlsx')

df_X_PF_IPC=pd.concat([df_X_PF,df_X_IPC],axis=1).reset_index(drop=True)

n_powerflows = len(df_X_PF)

for indicator in indicators_list:
    
    df_pf_ind = PFs_x_CCRCs(n_powerflows, indicator, combinations, path_data)
    
    #%% ---- DATA PREPROCESSING AND PCA ----
    X = preprocess_data(df_X_PF_IPC)
    
    n_intervals=5
    df_X_PCA, x_list, y_list, z_list = PCA_cluster(X, n_intervals, plot=True)
    
       
    #%%
    df_CCRs_PF_ind = df_1st_exploration(df_X_PF_IPC, np.arange(1,len(combinations)+1), combinations, n_powerflows, path_data, True)
    
    heatmap, CCRCs2clustering, df_pf_ind_levels = heatmap_function(n_intervals, x_list, y_list, z_list, indicator, 
                                                df_X_PCA, df_CCRs_PF_ind, df_pf_ind, 
                                                combinations, indicators_plot_labels, plot=True)
                        
    
    #%% ---- CCRCs CLUSTERING ACCORDING TO SINGLE INDICATOR ----
    labels, n_clusters = CCRCs_clustering(CCRCs2clustering,heatmap)

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
    
    #%% ---- Rearranged Attribute Matrix----
    
    complete_heat_map_rearranged = Rearranged_Attribute_Matrix(df_pf_ind_levels, heatmap, np.arange(0, n_clusters), labels, plot=True ,indicators_plot_labels=indicators_plot_labels, indicator=indicator, save_plot=True)
    reduced_heat_map_rearranged = Rearranged_Attribute_Matrix(df_pf_ind_levels, heatmap, selected_clusters_unique, labels, plot=True ,indicators_plot_labels=indicators_plot_labels, indicator=indicator, type_matrix='_reduced', save_plot=True)

#%%

 # exec(open('venn_diagram.py').read())
 



# # #%%

# # # Colors
# # BG_WHITE = "#fbf9f4"
# # GREY_LIGHT = "#b4aea9"
# # GREY50 = "#7F7F7F"
# # BLUE_DARK = "#1B2838"
# # BLUE = "#2a475e"
# # BLACK = "#282724"
# # GREY_DARK = "#747473"
# # RED_DARK = "#850e00"

# # # Colors taken from Dark2 palette in RColorBrewer R library
# # COLOR_SCALE = ["#1B9E77", "#D95F02", "#7570B3"]

# # # Horizontal positions for the violins. 
# # # They are arbitrary numbers. They could have been [-1, 0, 1] for example.
# # POSITIONS = [0]#, 1]

# # # # Horizontal lines
# # # HLINES = [0, 15, 30]

# # #%%

# # df=pd.read_csv(path_data+'df_BigModel_PFcomparison_1exp_H2_Hinf_DCgain.csv').query('Stable==1')

# # scu=list(selected_clusters_unique)
# # combinations_gen=list(labels.query('lab==@scu').index)

# # df_comb_gen=df.query('Combination == @combinations_gen')

# # indicators=['H2_en', 'H2_freq', 'H2_vdc','DCgain_vdc','DCgain_freq']

# # df_ind_descr=df[indicators].describe()
# # df_comb_gen_ind_descr=df_comb_gen[indicators].describe()

# # #%%
# # ylabels=[r'$\mathcal{H}_{2,En}$',r'$\mathcal{H}_{2,f}$',r'$\mathcal{H}_{2,V_{DC}}$',r'$\mathcal{K}_{f}$',r'$\mathcal{K}_{V_{DC}}$']

# # ii=0
# # for indicator in indicators:
    
# #     y_data=[]
# #     y_data.append(df[indicator].values)
# #     y_data.append(df_comb_gen[indicator].values)
    
    
# #     fig, ax = plt.subplots(figsize= (10, 3))
    
# #     # Some layout stuff ----------------------------------------------
# #     # Background color
# #     fig.patch.set_facecolor(BG_WHITE)
# #     ax.set_facecolor(BG_WHITE)
    
    
    
# #     # # Horizontal lines that are used as scale reference
# #     # for h in HLINES:
# #     #     ax.axhline(h, color=GREY50, ls=(0, (5, 5)), alpha=0.8, zorder=0)
    
# #     # Add violins ----------------------------------------------------
# #     # bw_method="silverman" means the bandwidth of the kernel density
# #     # estimator is computed via Silverman's rule of thumb. 
# #     # More on this in the bonus track ;)
    
# #     # The output is stored in 'violins', used to customize their appearence
# #     # violins = ax.violinplot(
# #     #     y_data, 
# #     #     positions=POSITIONS,
# #     #     widths=0.45,
# #     #     bw_method="silverman",
# #     #     showmeans=False, 
# #     #     showmedians=False,
# #     #     showextrema=False
# #     # )
    
# #     # # Customize violins (remove fill, customize line, etc.)
# #     # for pc in violins["bodies"]:
# #     #     pc.set_facecolor("none")
# #     #     pc.set_edgecolor(BLACK)
# #     #     pc.set_linewidth(1.4)
# #     #     pc.set_alpha(1)
        
    
# #     # Add boxplots ---------------------------------------------------
# #     # Note that properties about the median and the box are passed
# #     # as dictionaries.
    
# #     medianprops = dict(
# #         linewidth=4, 
# #         color=GREY_DARK,
# #         solid_capstyle="butt"
# #     )
# #     boxprops = dict(
# #         linewidth=2, 
# #         color=GREY_DARK
# #     )
    
# #     ax.boxplot(
# #         y_data,
# #         positions=POSITIONS, 
# #         showfliers = True, # Do not show the outliers beyond the caps.
# #         showcaps = False,   # Do not show the caps
# #         medianprops = medianprops,
# #         whiskerprops = boxprops,
# #         boxprops = boxprops,
# #         vert=False
# #     )
# #     ax.set_yticklabels(['Selected \n CCRCs'],rotation=90)#, 'Selected \n CCRCs'], r"CCRCs $\in \mathcal{R}$"])
    
# #     ax.set_xlabel(r'$\mathcal{H}_{2,V_{DC}}$')#ylabels[ii])
# #     fig.tight_layout()
# #     ax.grid()
# #     ii=ii+1
# #     # # Add jittered dots ----------------------------------------------
# #     # for x, y, color in zip(x_jittered, y_data, COLOR_SCALE):
# #     #     ax.scatter(x, y, s = 100, color=color, alpha=0.4)


# # # #%% Cluster analysis

# # # exec(open('scripts/BigModel/Clusters_analyis_h2.py').read())

# # # #%%

# # # l=16
# # # ind_cl=list(labels.loc[labels['lab']==l].index)
    
# # # comb_cluster=np.array(combinations)[ind_cl]

# # # #%%

# # # labels=pd.read_excel(path+'Clustering_results_'+indicator+'.xlsx', sheet_name='comb_clusters')#['lab']
# # # labels_index=list(labels['Unnamed: 0'])

# # # X3=X[labels_index,:]

# # # cmap = matplotlib.cm.hot
# # # cmap_reversed = matplotlib.cm.get_cmap('hot',100)

# # # red = np.array([ 1, 0, 0, 1])
# # # yellow=np.array([1,1,0.3,1])
# # # green=np.array([0,1,0,1])
# # # orange=np.array([1,165/255,0,1])

# # # newcolors_stab = cmap_reversed(np.linspace(0, 1, 100))
# # # newcolors_stab[:25, :] = green
# # # newcolors_stab[25:50, :] = yellow
# # # newcolors_stab[50:75,:] = orange
# # # newcolors_stab[75:,:] = red

# # # newcmp_stab = ListedColormap(newcolors_stab)

# # # xgrid = np.arange(X.shape[1])+1
# # # ygrid = np.arange(X.shape[0]+1)+1

# # # # fig = plt.figure(figsize=(13,9))
# # # # gs = GridSpec(2, 1,left=0.15,right=0.95, top=0.93,bottom=0.1,hspace=0.25)

# # # # ax= fig.add_subplot(gs[0])
# # # # bx= fig.add_subplot(gs[1])

# # # fig = plt.figure(figsize=(8,7))
# # # ax= fig.add_subplot()

# # # aa=ax.pcolormesh(xgrid, ygrid, pd.DataFrame(X),cmap=newcmp_stab)#,edgecolors='w', linewidths=0.01)
# # # ax.set_frame_on(False) # remove all spines

# # # plt.setp(ax.get_xticklabels(), visible=False)
# # # ax.tick_params(axis='x', which='both', length=0)
# # # ax.set_yticks([1,95])
# # # ax.tick_params(axis='y', which='both', labelsize=25)
# # # ax.set_ylabel('CCRCs', fontsize=30, labelpad=-15)
# # # ax.set_xlabel('Stability Maps Sub-regions', fontsize=30) #[$X_A$,$X_B$,$X_C$] Ranges
# # # ax.set_title('Attributes Matrix', fontsize=30)

# # # cb=fig.colorbar(aa, ax=ax,ticks=[1,2,3,4,5])

# # # cb.set_label("$\mathcal{I}_{V_{DC}}$", size=25, rotation=270, labelpad=25)
# # # cb.ax.tick_params(labelsize=25)

# # # fig.tight_layout()

# # # fig = plt.figure(figsize=(8,8))
# # # bx= fig.add_subplot()

# # # bb=bx.pcolormesh(xgrid, ygrid, pd.DataFrame(X3),cmap=newcmp_stab,edgecolors='w', linewidths=0.05)
# # # bx.set_frame_on(False) # remove all spines

# # # plt.setp(bx.get_xticklabels(), visible=False)
# # # plt.setp(bx.get_yticklabels(), visible=False)
# # # bx.tick_params(axis='both', which='both', length=0)
# # # bx.set_ylabel('CCRCs', fontsize=30, labelpad=32)
# # # bx.set_xlabel('Stability Maps Sub-regions', fontsize=30) #[$X_A$,$X_B$,$X_C$] Ranges
# # # bx.set_title('Attributes Matrix \n Rearranged according to Clusters', fontsize=30)

# # # cb=fig.colorbar(bb, ax=bx,ticks=[1,2,3,4,5])

# # # cb.set_label("$\mathcal{I}_{V_{DC}}$", size=25, rotation=270, labelpad=25)
# # # cb.ax.tick_params(labelsize=25)

# # # fig.tight_layout()

# # # #gs.update(left=0.1,right=0.97, top=0.95,bottom=0.1,hspace=0.47)

# # # #%%
# # # indicators=['H2_vdc','H2_freq','H2_en','DCgain_vdc']
# # # selected_combinations0=np.empty([1])
# # # for i in indicators:
# # #     selected_clusters=pd.read_excel("Clustering_results_"+i+".xlsx",sheet_name='selected_clusters')['Combinations'].drop_duplicates()
# # #     for j in selected_clusters.index:
# # #         comb=np.array(selected_clusters[j][1:-1].split(','))
# # #         selected_combinations0=np.concatenate((selected_combinations,comb),axis=0)             

# # # selected_combinations=np.zeros([len(selected_combinations0)-1,1])
# # # for i in range(0,len(selected_combinations)-1):
# # #     selected_combinations[i,0]=int(float(selected_combinations0[i+1]))
    
# # # selected_combinations=np.unique(selected_combinations)


# # # # #%%  ---- CREATE INPUT DATA FOR SUCCESSIVE ITERATIONS ----
# # # # df_pf = pd.read_excel(path+'PFtab_red_1exp_hnorm_dcgain.xlsx')

# # # # exec(open('scripts/BigModel/input_for_successive_iterations.py').read())

