#%% IMPORTS
import numpy as np # matrix operations
import pandas as pd # data management
import statsmodels.api as sm # statistical analysis
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting

# machine learning
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.feature_selection import RFECV #RFE-CV
from sklearn.inspection import permutation_importance #PFI

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold, KFold
from sklearn.model_selection import cross_val_score

from data_preprocessing import *
from create_dataset import *

#%%
plt.rcParams.update({"figure.figsize" : [15 , 6],
                     "text.usetex": True,
                     "font.family": "serif",
                     "font.serif": "Palatino",
                     "axes.labelsize": 20,
                     "axes.titlesize": 20,
                     "axes.spines.right":False,
                     "axes.spines.top":False,
                     "legend.fontsize": 20,
                     "xtick.labelsize": 16,
                     "ytick.labelsize": 16,
                     "savefig.dpi": 130,
                    'legend.fontsize': 20,
                    'legend.handlelength': 2,
                    'legend.loc': 'upper right'})

# pandas display options
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width', 84)
pd.set_option('display.max_columns', None)

#%%
''' to use the same list of CCRCs selected in the paper set reproduce_paper='_paper', else reproduce_paper=str()'''

reproduce_paper=str()#'_paper'

#%%
indicators_list=['H2_freq', 'H2_vdc','DCgain_vdc','DCgain_freq']

#%%
exec(open('../Settings/combinations.sav').read())
exec(open('../CCRCs_clustering/Results/CCRC_dict'+reproduce_paper+'.sav').read())

combinations_selected=CCRC_dict['selected_CCRC']

#%% ---- CREATE COMPLETE DB / LOAD CREATED DB ----
path_data='./'
path_source=path_data+'Stability_analysis_results_all_CCRCs/'

# Create DB

df = create_dataset_with_selected_CCRCs(path_data,path_source, combinations_selected,combinations, save_dataset=True, filename='df_selected_combinations.csv')
df = df.drop('Unnamed: 0',axis=1)

# #Load DB
# filename='df_selected_combinations'+reproduce_paper+'.csv'
# df=pd.read_csv(path_data+filename).drop('Unnamed: 0',axis=1)


#%% DATA PREPROCESSING

df = df.drop(indicators_list, axis=1)

df = preprocess_data(df, True)
