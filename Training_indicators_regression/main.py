#%% IMPORTS
import numpy as np # matrix operations
import pandas as pd # data management
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt # plotting

# machine learning
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from sklearn.feature_selection import RFECV #RFE-CV
from sklearn.inspection import permutation_importance #PFI

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold, KFold
from sklearn.model_selection import cross_val_score

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, fbeta_score, r2_score

from sklearn import tree

##
from metrics_n import *
from utils import *
from data_preprocessing import *
from training_functions import *
from scipy.stats.mstats import winsorize

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline

import joblib

#%%

plt.rcParams.update({"figure.figsize" : [8, 10],
                     "text.usetex": True,
                     "font.family": "serif",
                     "font.serif": "Computer Modern",
                     "axes.labelsize": 20,
                     "axes.titlesize": 20,
                     "legend.fontsize": 20,
                     "xtick.labelsize": 20,
                     "ytick.labelsize": 20,
                     "savefig.dpi": 130,
                    'legend.fontsize': 20,
                    'legend.handlelength': 2,
                    'legend.loc': 'upper right'})

#%%
# exclude_indicators=['Stable','Hinf_en', 'Hinf_freq', 'Hinf_vdc', 'H2_en', 'H2_freq', 'H2_vdc','DCgain_vdc','DCgain_freq']

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

#%% DATA PREPROCESSING

##TODO: fix this in the files
# df.rename(columns = {'P2l':'Pl2', 'P5l':'Pl5', 'P9l':'Pl9','Q2l':'Ql2', 'Q5l':'Ql5', 'Q9l':'Ql9'}, inplace = True)
df_inputs = df.drop(indicators_list+['Stable'], axis=1)

df_inputs= preprocess_data(df_inputs)

df=pd.concat([df_inputs,df[indicators_list+['Stable']]],axis=1)

#%% Take only stable cases
df_stab=df.query('Stable==1')

#%% SET SCORER METRICS

scorer=make_scorer(r2_score)

#%% TRAIN-TEST SPLIT

train_size = 0.8 #60%
test_size = 0.2 #20%

X_train=pd.DataFrame()
Y_train=pd.DataFrame()
X_test=pd.DataFrame()
Y_test=pd.DataFrame()

for comb in combinations_selected:

    Y = df_stab.query('Combination == @comb')[['Combination','H2_freq','H2_vdc','DCgain_freq','DCgain_vdc']]
    X = df_stab.query('Combination == @comb').drop(indicators_list, axis=1)

    seed=23

    X_train0, X_test0, Y_train0, Y_test0 = train_test_split(X, Y, test_size=test_size, shuffle=True, random_state=seed)

    X_train=pd.concat([X_train,X_train0],axis=0)
    X_test=pd.concat([X_test,X_test0],axis=0)
    Y_train=pd.concat([Y_train,Y_train0],axis=0)
    Y_test=pd.concat([Y_test,Y_test0],axis=0)



#%%

models_list = []
models_list.append(('Dummy', DummyRegressor(strategy='mean')))
models_list.append(('LR', LinearRegression()))
models_list.append(('Ridge', Ridge(alpha=0.5)))
models_list.append(('DTR', DecisionTreeRegressor()))
models_list.append(('MLP', MLPRegressor(max_iter=5000)))
models_list.append(('XGB', XGBRegressor(eta=0.5,max_depth=1, alpha=0)))

cv = KFold(n_splits=6, shuffle=True, random_state=1)

scores_indicators={}

for indicator in indicators_list:
    scores_indicators[indicator]=[]
    scores_CCRC={}
    for c in combinations_selected:
        scores_CCRC['CCRC_{}'.format(c)]=[]
        X_train_CCRC=X_train.query('Combination == @c').drop('Combination',axis=1)
        X_train_CCRC=X_train_CCRC.drop(['IPCA','IPCB','IPCC','IPCD','IPCE'],axis=1)
        y_train_CCRC=winsorize(Y_train.query('Combination == @c ')[indicator],limits=[0,0.05])

        scores={}
        for name,alg in models_list:
            pipeline = Pipeline([('scaler', StandardScaler()), (name, alg)])
            model=TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())

            CV_score = cross_val_score(model, X_train_CCRC, y_train_CCRC, scoring=scorer, cv=cv)
            res_cv=np.mean(CV_score),np.std(CV_score)
            scores[name]=[]
            scores[name].append(res_cv[0])

        scores_CCRC['CCRC_{}'.format(c)].append(scores)
    scores_indicators[indicator].append(scores_CCRC)

#%%
model_comparison=pd.DataFrame()
model_comparison_r2=pd.DataFrame()
for c in range(0,len(combinations_selected)):
   for indicator in indicators_list:

       model_comparison.loc[c,'CCRC']=combinations_selected[c]
       model_comparison_r2.loc[c,'CCRC']=combinations_selected[c]
       max_index=np.argmax(np.array(pd.DataFrame(scores_indicators[indicator][0]['CCRC_{}'.format(combinations_selected[c])]).T))
       model_comparison.loc[c,indicator]= models_list[max_index][0]
       model_comparison_r2.loc[c,indicator]=scores_indicators[indicator][0]['CCRC_{}'.format(combinations_selected[c])][0][models_list[max_index][0]][0]

#%%
# #model_comparison2=model_comparison.copy(deep=True).set_index('Combination')

models_dict = {'Dummy': DummyRegressor(strategy='mean'),
               'LR': LinearRegression(),
               'Ridge': Ridge(),
               'DTR': DecisionTreeRegressor(),
               'MLP': MLPRegressor(max_iter=5000),
               'XGB': XGBRegressor()}

scores_PFI=pd.DataFrame()

for c in range(len(combinations_selected)):

    CCRC = combinations_selected[c]
    PFI_dict={'H2_freq':[], 'H2_vdc':[], 'DCgain_freq':[], 'DCgain_vdc':[]}

    for indicator in indicators_list:
        name=model_comparison.loc[c,indicator]
        alg=models_dict[name]
        pipeline = Pipeline([('scaler', StandardScaler()), (name, alg)])
        estimator=TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())

        X_train_CCRC=X_train.query('Combination == @CCRC').drop('Combination',axis=1)
        X_train_CCRC=X_train_CCRC.drop(['IPCA','IPCB','IPCC','IPCD','IPCE'],axis=1)
        y_train_CCRC=np.array(winsorize(Y_train.query('Combination == @CCRC ')[indicator],limits=[0,0.05]))

        X_test_CCRC=X_test.query('Combination == @CCRC').drop('Combination',axis=1)
        X_test_CCRC=X_test_CCRC.drop(['IPCA','IPCB','IPCC','IPCD','IPCE'],axis=1)
        y_test_CCRC=Y_test.query('Combination == @CCRC ')[indicator]


        if str(name) == 'Dummy_r2' or comb==6:
            PFI_features=list(X_train_comb.columns)
        else:
            PFI_features = PFI_fun(estimator, X_train_CCRC, y_train_CCRC, X_test_CCRC, y_test_CCRC, scorer)
            
        if PFI_features == []:
            PFI_features=list(X_train_CCRC.columns)

        PFI_dict[indicator] = PFI_features

        estimator.fit(X_train_CCRC[PFI_features], y_train_CCRC)

        scores_PFI.loc[c,'CCRC']=combinations_selected[c]
        scores_PFI.loc[c,indicator]=r2_score(y_test_CCRC,estimator.predict(X_test_CCRC[PFI_features]))
                
        Y = winsorize(df_stab.query('Combination == @CCRC')[indicator],limits=[0,0.05])
        X = df_stab.query('Combination == @CCRC')[PFI_features]

        estimator.fit(X, Y)

        filename=path+"Regr_"+indicator+"_CCRC"+str(comb)+".sav"

        joblib.dump(estimator,filename)



    f = open(path+"PFI_features_CCRC"+str(comb)+".sav","w")

    # write file
    f.write( 'PFI_dict='+str(PFI_dict) )

    # close file
    f.close()

#%%
use_PFI=pd.DataFrame(columns=['CCRC']+indicators_list)
use_PFI['CCRC']=model_comparison_r2['CCRC']
for i in range(len(model_comparison_r2)):
    for indicator in indicators_list:
        if model_comparison_r2.loc[i,indicator] > scores_PFI.loc[i,indicator]:
            use_PFI.loc[i,indicator]= 0 
        else:
            use_PFI.loc[i,indicator]= 1
            

#%% Write results
import xlsxwriter

writer = pd.ExcelWriter(path+"model_comparison_indicators_regressions.xlsx",
                        engine='xlsxwriter')

model_comparison.to_excel(writer, sheet_name='model_comparison')
model_comparison_r2.to_excel(writer, sheet_name='model_comparison_r2')
scores_PFI.to_excel(writer, sheet_name='scores_PFI')
use_PFI.to_excel(writer, sheet_name='use_PFI')
# writer.save()
writer.close()

#%%

use_PFI=use_PFI.set_index('CCRC')

#%%
models_dict = {'Dummy': DummyRegressor(strategy='mean'),
               'LR': LinearRegression(),
               'Ridge': Ridge(),
               'DTR': DecisionTreeRegressor(),
               'MLP': MLPRegressor(max_iter=5000),
               'XGB': XGBRegressor()}

indicators_list=['H2_freq','H2_vdc','DCgain_freq','DCgain_vdc']


for c in range(len(combinations_selected)):

    CCRC = combinations_selected[c]
    PFI_dict={'H2_freq':[], 'H2_vdc':[], 'DCgain_freq':[], 'DCgain_vdc':[]}

    for indicator in indicators_list:
        name=model_comparison.loc[c,indicator]
        use_PFI_bool=use_PFI.loc[CCRC,indicator]
        
        if use_PFI_bool == 0:
        
            alg=models_dict[name]
            pipeline = Pipeline([('scaler', StandardScaler()), (name, alg)])
            estimator=TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())

            X = df_stab.query('Combination == @CCRC').drop('Combination',axis=1)                   
            Y = winsorize(df_stab.query('Combination == @CCRC')[indicator],limits=[0,0.05])
    
            estimator.fit(X, Y)
    
            filename=path+"Regr_"+indicator+"_CCRC"+str(CCRC)+"_NOPFI.sav"
    
            joblib.dump(estimator,filename)
    
