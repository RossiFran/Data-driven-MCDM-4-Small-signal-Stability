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

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline


from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.model_selection import cross_val_score

from sklearn import tree
from metrics_n import *
from utils import *
from data_preprocessing import *
from models_comparison import *
from training_functions import *

#SAVE THE MODEL
import joblib

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
indicators_list=['H2_freq', 'H2_vdc','DCgain_vdc','DCgain_freq']

#%%
exec(open('../Settings/combinations.sav').read())
exec(open('../CCRCs_clustering/Results/CCRC_dict_paper.sav').read())

combinations_selected=CCRCs_dict['selected_CCRC']

#%% ---- CREATE COMPLETE DB ----
path='./Results/'
path_data='./Data/'

filename='df_selected_combinations.csv'

df=pd.read_csv(path_data+filename).drop('Unnamed: 0',axis=1)

#%% SET SCORER METRICS

print('Stable cases = ', df[['Stable']].mean()[0]*100)
scorer = set_scorer(df)

#%% DATA PREPROCESSING

#TODO: fix this in the files
df.rename(columns = {'P2l':'Pl2', 'P5l':'Pl5', 'P9l':'Pl9','Q2l':'Ql2', 'Q5l':'Ql5', 'Q9l':'Ql9'}, inplace = True)
df = df.drop(indicators_list, axis=1)

df = pu_conversion(df)

df = features_creation(df)

columns_I, columns_PQ, columns_V, columns_vd, columns_vq, columns_id, columns_iq, columns_P, columns_Q = group_columns()

columns_remove, rows_remove = data_clean_results()

df = final_df(df,columns_remove,rows_remove)


#%% TRAIN-TEST SPLIT

train_size = 0.8 
test_size = 0.2 

Y = df[['Stable']] 
X = df.drop(['Stable'], axis=1)

seed=23

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, shuffle=True, random_state=seed)


#%% STANDARDIZATION
scaler = StandardScaler()
scaler.fit(X_train) # Calculate the scaler

# print('Mean =', scaler.mean_)
# print('Std.Dev =', scaler.scale_)

X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
# X_eval = pd.DataFrame(scaler.transform(X_eval), columns=X_eval.columns, index=X_eval.index)

#%% Comparison

models_list = []
models_list.append(('Dummy', DummyClassifier(strategy='constant',constant=1)))
models_list.append(('LogReg', LogisticRegression(max_iter=5000)))
models_list.append(('DTC', DecisionTreeClassifier()))
models_list.append(('MLP', MLPClassifier(max_iter=5000)))
models_list.append(('XGB-DTC', XGBClassifier()))

n_folds = 6

model_results = compare_models(models_list, X_train, Y_train['Stable'], scoring=scorer,n_folds=n_folds, plot=True, save_plot=True, path=path)

#%% Setup cross validation results table
#filename='models_CV_res.csv'
CrossVal_res_training=pd.DataFrame()
CrossVal_res_training['metrics']=['f_beta']
CrossVal_res_training=CrossVal_res_training.set_index('metrics')

CrossVal_res_test=pd.DataFrame()
CrossVal_res_test['metrics']=['f_beta']
CrossVal_res_test=CrossVal_res_test.set_index('metrics')

PFI_dict={'MLP':[],'XGB':[]}

#%% --- MLP CLASSIFIER ---

#Feature selection by Permutation Feature Importance (PFI)

estimator = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier())])

PFI_dict['MLP'] = PFI_fun(estimator, X_train, Y_train, X_test, Y_test, scorer)

f = open(path+"PFI_features_MLP.sav","w")

# write file
f.write( 'PFI_dict='+str(PFI_dict['MLP']) )

# close file
f.close()

# Hyperparameters Tuning
n_feat = len(PFI_dict['MLP'])
param_grid = {
    'mlp__max_iter': [5000],
    'mlp__hidden_layer_sizes': [(n_feat,), (2*n_feat,), (3*n_feat,),
                           (n_feat,n_feat,), (2*n_feat,2*n_feat,),
                           (2*n_feat,n_feat), (n_feat,2*n_feat,)],
                           #(100,), (100,100,), (50,50,), (50,100,)],
    'mlp__activation': ['logistic', 'tanh', 'relu'],
    }


best_model, best_params, means, stds, params = GSkFCV(param_grid, X_train[PFI_dict['MLP']], Y_train, estimator, scorer)

model = best_model

CrossVal_res_training.loc['f_beta','MLP']= str(sorted(zip(means, stds, params), key=lambda x: x[0], reverse=True)[0][0])+'\pm'+str(sorted(zip(means, stds, params), key=lambda x: x[0], reverse=True)[0][1])

CV_scores = cross_val_score(model, X_test[PFI_dict['MLP']], Y_test, cv=6)
CrossVal_res_test.loc['f_beta','MLP']= str(CV_scores.mean())+'\pm'+str(CV_scores.std())

#%% --- FINAL TRAINING ---
best_params_grid={'activation': best_params['mlp__activation'],
              'hidden_layer_sizes': best_params['mlp__hidden_layer_sizes'],
              'max_iter': best_params['mlp__max_iter']}

# f=open(path+'MLP'+filename+'_best_params_and_GSCV_metrics.sav','w')
# f.write(str(best_params_2)+'\n'+str(print_metrics(test_results, names=['Accuracy', 'Precision', 'Prec. Unstable', 'Recall', 'F1 score'])))
# f.close()              

estimator = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(**best_params_grid))])

estimator.fit(df[PFI_dict['MLP']],df['Stable'])

filename=path+'MLP.sav'

joblib.dump(estimator, filename)

#%% XGB CLASSIFIER

estimator = Pipeline([('scaler', StandardScaler()), ('xgb', XGBClassifier())])

PFI_dict['XGB']  = PFI_fun(estimator, X_train, Y_train, X_test, Y_test, scorer)

f = open(path+"PFI_features_XGB.sav","w")

# write file
f.write( 'PFI_dict='+str(PFI_dict['XGB']  ) )

# close file
f.close()

param_grid = {'xgb__eta':np.arange(0.1,0.5,0.2),
              'xgb__max_depth':[5,6],#np.arange(3,7,2),
              'xgb__subsample':[0.5,1]
    }

best_model, best_params, means, stds, params = GSkFCV(param_grid, X_train[PFI_dict['XGB']], Y_train, estimator, scorer)

model = best_model

CrossVal_res_training.loc['f_beta','XGB-DTC']= str(sorted(zip(means, stds, params), key=lambda x: x[0], reverse=True)[0][0])+'\pm'+str(sorted(zip(means, stds, params), key=lambda x: x[0], reverse=True)[0][1])

CV_scores = cross_val_score(model, X_test[PFI_dict['XGB']], Y_test, cv=6)

CrossVal_res_test.loc['f_beta','XGB-DTC']= str(CV_scores.mean())+'\pm'+str(CV_scores.std())

#%%
best_params_grid={'xgb__eta': best_params['xgb__eta'],
              'xgb__max_depth': best_params['xgb__max_depth'],
              'xgb__subsample': best_params['xgb__subsample']}

# filename='BigModel_22marz_19comb.sav'
              
# f=open(path+'XGB'+filename+'_best_params_and_GSCV_metrics.sav','w')
# f.write(str(best_params_2)+'\n'+str(print_metrics(test_results, names=['Accuracy', 'Precision', 'Prec. Unstable', 'Recall', 'F1 score'])))
# f.close()              


estimator = Pipeline([('scaler', StandardScaler()), ('xgb', XGBClassifier(**best_params_grid))])

estimator.fit(df[PFI_dict['XGB'] ],df['Stable'])

filename=path+'XGB.sav'

# pickle.dump(hybrid_model, open(filename,'wb'))
joblib.dump(estimator, filename)

#%%

pd.DataFrame.to_excel(CrossVal_res_training,path+'CV_Results_train.xlsx')
pd.DataFrame.to_excel(CrossVal_res_test,path+'CV_Results_test.xlsx')

# #%% --- CREATE EVAL DB ---

# df_pf= pd.read_excel(path_data+'PFtab_red_1exp_hnorm_dcgain_2.xlsx')
# indicator='Stab'
# n_powerflows = len(df_pf)
# df_pf_h2=pd.DataFrame()
# for c in combinations_gen:#range(1,len(combinations)+1):
#     df_stab_hinf_h2 = pd.read_excel(path_data+'Stab_Hinf_H2_DCgain_comb'+str(c)+'_1exp_2.xlsx')#_longline_pdred.xlsx')#.iloc[selected_combinations_py]

#     df_pf_h2[['Combination_{0:.0f}'.format(c)]]= df_stab_hinf_h2[[indicator]]

# df_pq_th_mmc = pd.read_excel(path_data+'PQ_MMC_TH_hnorm_dcgain_comb1_1exp_2.xlsx')#'_longline_pdred.xlsx')
# df_pf=pd.concat([df_pf,df_pq_th_mmc],axis=1)

# #all_comb=np.arange(1,len(combinations)+1)
# name_file='df_BigModel_PFcomparison_1exp_H2_Hinf_DCgain_2'

# df=pd.DataFrame()
  
# for i in combinations_gen:#all_comb:
#     df_pf['Combination']=i#selected_combinations[i]
#     df_pf['IPCA'] = combinations[i-1][0]
#     df_pf['IPCB'] = combinations[i-1][1]
#     df_pf['IPCC'] = combinations[i-1][2]
#     df_pf['IPCD'] = combinations[i-1][3]
#     df_pf['IPCE'] = combinations[i-1][4]
#     df_pf['IPCF'] = combinations[i-1][5]
#     df_pf['Powerflow']=np.arange(1,len(df_pf)+1)
#     df_stab_hinf_h2 = pd.read_excel(path_data+'Stab_Hinf_H2_DCgain_comb'+str(i)+'_1exp_2.xlsx')
#     df_pf['Stable']=df_stab_hinf_h2['Stab']
#     df_pf['Hinf_en']=df_stab_hinf_h2['Hinf_en']
#     df_pf['Hinf_freq']=df_stab_hinf_h2['Hinf_freq']
#     df_pf['Hinf_vdc']=df_stab_hinf_h2['Hinf_vdc']
#     df_pf['H2_en']=df_stab_hinf_h2['H2_en']
#     df_pf['H2_freq']=df_stab_hinf_h2['H2_freq']
#     df_pf['H2_vdc']=df_stab_hinf_h2['H2_vdc']
#     df_pf['DCgain_vdc']=df_stab_hinf_h2['DCgain_vdc']
#     df_pf['DCgain_freq']=df_stab_hinf_h2['DCgain_freq']
    
#     df=pd.concat([df,df_pf],axis=0)

# df=df.reset_index(drop=True)
        
    
# df.to_csv(path+name_file+'.csv', index=True, index_label='Case')    

# #%% --- LOAD EVAL DB ---
# name_file='df_BigModel_PFcomparison_1exp_H2_Hinf_DCgain_2'

# df = pd.read_csv(path+name_file+'.csv').set_index('Case')
# df.rename(columns = {'P2l':'Pl2', 'P5l':'Pl5', 'P9l':'Pl9','Q2l':'Ql2', 'Q5l':'Ql5', 'Q9l':'Ql9'}, inplace = True)

# def final_df(df): # Return final df (w/o removed columns and rows)
#     return df.drop(columns_remove, axis=1, errors='ignore').drop(rows_remove, axis=0, errors='ignore')

# exec(open('settings/pu_params.sav').read())


# exec(open('settings/columns.sav').read())


# exec(open('settings/featurecreation_complete_tool_30marzo.sav').read())


# exec(open('settings/dataclean.sav').read())

# #%%

# model_name='_XGB'#'_MLP'#

# combinations_gen=dict_gen_and_test['gen_comb']
# df_eval=df.query('Combination == @combinations_gen')#'@c_gen')

# Y = final_df(df_eval)[['Stable']] # [['Stable', 'DIglob', 'DIloc1', 'DIloc2', 'DIloc3']]
# X = final_df(df_eval).drop(Y.columns, axis=1)
# # X=final_df(df)[PFI_features]

# exec(open('settings/PFI_features'+model_name+'.sav').read())
# PFI_features=PFI_dict[model_name[1:]]
# import joblib
# # filename='models/'+model_name[1:]+'_BigModel_22marz_19comb.sav'
# filename='models/'+model_name[1:]+'_14comb.sav'
# estimator=joblib.load(filename)


# n_splits=6
# metrics_list = [accuracy_score, precision_score, recall_score, f1_score, scorer]
# test_results = get_metrics(estimator, X[PFI_features], Y, metrics_list, scorer, n=n_splits)
# print('FINAL MODEL METRICS ON EVAL SET')
# print_metrics(test_results, names=['Accuracy', 'Precision', 'Recall', 'F1 score', 'f_beta'])


# #%%
# # exec(open('settings/BigModel/PFI_features_XGB.sav').read())
# # PFI_features=PFI_dict['XGB']
# # import joblib
# # filename='models/'+model_name+'_BigModel_22marz_19comb.sav'

# # estimator=joblib.load(filename)

# # f = open(path+"results_19_comb_XGB.sav","a")

# # combinations_test=dict_gen_and_test ['test_comb']
# # c=0
# # for c_test in combinations_test:
# #     if len(c_test)>0:
# #         df_eval=df.query('Combination == @c_test')
# #         c_gen=combinations_gen[c]
        
# #         df_eval['IPCA']=combinations[c_gen-1][0]
# #         df_eval['IPCB']=combinations[c_gen-1][1]
# #         df_eval['IPCC']=combinations[c_gen-1][2]
# #         df_eval['IPCD']=combinations[c_gen-1][3]
# #         df_eval['IPCE']=combinations[c_gen-1][4]
# #         df_eval['IPCF']=combinations[c_gen-1][5]

# #         Y = final_df(df_eval)[['Stable']] # [['Stable', 'DIglob', 'DIloc1', 'DIloc2', 'DIloc3']]
# #         X = final_df(df_eval)[PFI_features]#.drop(Y.columns, axis=1)
# #     # X=final_df(df)[PFI_features]

     
# #         n_splits=6
# #         metrics_list = [accuracy_score, precision_score, recall_score, f1_score]
# #         test_results = get_metrics(estimator, X[PFI_features], Y, metrics_list, n=n_splits)
# #         #print('FINAL MODEL METRICS ON EVAL SET - Comb_gen',c_gen)
# #         #print_metrics(test_results, names=['Accuracy', 'Precision', 'Recall', 'F1 score'])
        
# #         # write file
# #         f.write( 'FINAL MODEL METRICS ON EVAL SET - Comb_gen'+str(c_gen)+' - Comb_test: ' +str(c_test)+ ' \n')
# #         print_metrics_on_file(test_results, f, names=['Accuracy', 'Precision', 'Recall', 'F1 score'])       

# #     else:
# #         c_gen=combinations_gen[c]
# #         df_eval=df.query('Combination == @c_gen')
        
# #         Y = final_df(df_eval)[['Stable']] # [['Stable', 'DIglob', 'DIloc1', 'DIloc2', 'DIloc3']]
# #         X = final_df(df_eval)[PFI_features]#.drop(Y.columns, axis=1)
# #     # X=final_df(df)[PFI_features]

     
# #         n_splits=6
# #         metrics_list = [accuracy_score, precision_score, recall_score, f1_score]
# #         test_results = get_metrics(estimator, X[PFI_features], Y, metrics_list, n=n_splits)
# #         #print('FINAL MODEL METRICS ON EVAL SET - Comb_gen',c_gen)
# #         #print_metrics(test_results, names=['Accuracy', 'Precision', 'Recall', 'F1 score'])
        
# #         # write file
# #         f.write( 'FINAL MODEL METRICS ON EVAL SET - Comb_gen'+str(c_gen) + ' \n')
# #         print_metrics_on_file(test_results, f, names=['Accuracy', 'Precision', 'Recall', 'F1 score'])       


# #     c=c+1

# # # close file
# # f.close()

# # #%%
