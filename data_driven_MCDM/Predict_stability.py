#%% IMPORTS
import numpy as np # matrix operations
import pandas as pd # data management
import joblib
from xgboost import XGBClassifier
#import sklearn.neural_network #import MLPClassifier

import warnings
warnings.filterwarnings("ignore")

#%%
def system_bases():
	Sb=500e6
	Vb=280e3
	Fb=50
	Ib=Sb/(np.sqrt(3)*Vb)
	Zb=Vb**2/Sb
	Wb=2*np.pi*Fb
	
	return Sb, Vb, Fb, Ib, Zb, Wb 

#%%
if use_paper_model==True:
    reproduce_paper='_paper'
else:
    reproduce_paper=str()

#%%

model_name='XGB'
path='../Trainig_stability_assessment/Results/'

filename=path+model_name+reproduce_paper+'.sav'

model = joblib.load(filename)
    
exec(open('../Settings/combinations.sav').read())

CCRC=combinations[int(c)-1]

X=pd.DataFrame(X)
X.columns=['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','theta1','theta2','theta3','theta4','theta5','theta6','theta7','theta8','theta9','theta10','theta11',
    'Pth1','Pmmc1','Pg1','Pmmc2','Pmmc4','Pmmc3','Pl7','Pth2','Pmmc6','Pmmc5','Pg3','Pg2','Pl2','Pl5','Pl9',   
    'Qth1','Qmmc1','Qg1','Qmmc2','Qmmc4','Qmmc3','Ql7','Qth2','Qmmc6','Qmmc5','Qg3','Qg2','Ql2','Ql5','Ql9',   
    'V1dc','V2dc','V3dc','V4dc','V5dc','V6dc','P1dc','P2dc','P3dc','P4dc','P5dc','P6dc',
    'vq0_1', 'vd0_1', 'vq0c_1', 'vd0c_1', 'iq0_1', 'id0_1', 'iq0c_1', 'id0c_1', 'etheta0_1', 'P_1', 'Q_1',
    'vq0_2', 'vd0_2', 'vq0c_2', 'vd0c_2', 'iq0_2', 'id0_2', 'iq0c_2', 'id0c_2', 'etheta0_2', 'P_2', 'Q_2',
    'vq0_3', 'vd0_3', 'vq0c_3', 'vd0c_3', 'iq0_3', 'id0_3', 'iq0c_3', 'id0c_3', 'etheta0_3', 'P_3', 'Q_3',
    'vq0_4', 'vd0_4', 'vq0c_4', 'vd0c_4', 'iq0_4', 'id0_4', 'iq0c_4', 'id0c_4', 'etheta0_4', 'P_4', 'Q_4',
    'vq0_5', 'vd0_5', 'vq0c_5', 'vd0c_5', 'iq0_5', 'id0_5', 'iq0c_5', 'id0c_5', 'etheta0_5', 'P_5', 'Q_5',
    'vq0_6', 'vd0_6', 'vq0c_6', 'vd0c_6', 'iq0_6', 'id0_6', 'iq0c_6', 'id0c_6', 'etheta0_6', 'P_6', 'Q_6',
    'vq0_7', 'vd0_7', 'vq0c_7', 'vd0c_7', 'iq0_7', 'id0_7', 'iq0c_7', 'id0c_7', 'etheta0_7', 'P_7', 'Q_7',
    'vnq0_mmc1', 'vnd0_mmc1', 'vnq0_c_mmc1', 'vnd0_c_mmc1', 'idiffq0_mmc1', 'idiffd0_mmc1', 'idiffq0_c_mmc1', 'idiffd0_c_mmc1', 'vdiffq0_mmc1', 'vdiffd0_mmc1', 'vdiffq0_c_mmc1', 'vdiffd0_c_mmc1', 'isum0_mmc1', 'vsum0_mmc1', 'vDC0_mmc1', 'etheta0_mmc1','vnq0_mmc2', 'vnd0_mmc2', 'vnq0_c_mmc2', 'vnd0_c_mmc2', 'idiffq0_mmc2', 'idiffd0_mmc2', 'idiffq0_c_mmc2', 'idiffd0_c_mmc2', 'vdiffq0_mmc2', 'vdiffd0_mmc2', 'vdiffq0_c_mmc2', 'vdiffd0_c_mmc2', 'isum0_mmc2', 'vsum0_mmc2', 'vDC0_mmc2', 'etheta0_mmc2','vnq0_mmc3', 'vnd0_mmc3', 'vnq0_c_mmc3', 'vnd0_c_mmc3', 'idiffq0_mmc3', 'idiffd0_mmc3', 'idiffq0_c_mmc3', 'idiffd0_c_mmc3', 'vdiffq0_mmc3', 'vdiffd0_mmc3', 'vdiffq0_c_mmc3', 'vdiffd0_c_mmc3', 'isum0_mmc3', 'vsum0_mmc3', 'vDC0_mmc3', 'etheta0_mmc3','vnq0_mmc4', 'vnd0_mmc4', 'vnq0_c_mmc4', 'vnd0_c_mmc4', 'idiffq0_mmc4', 'idiffd0_mmc4', 'idiffq0_c_mmc4', 'idiffd0_c_mmc4', 'vdiffq0_mmc4', 'vdiffd0_mmc4', 'vdiffq0_c_mmc4', 'vdiffd0_c_mmc4', 'isum0_mmc4', 'vsum0_mmc4', 'vDC0_mmc4', 'etheta0_mmc4','vnq0_mmc5', 'vnd0_mmc5', 'vnq0_c_mmc5', 'vnd0_c_mmc5', 'idiffq0_mmc5', 'idiffd0_mmc5', 'idiffq0_c_mmc5', 'idiffd0_c_mmc5', 'vdiffq0_mmc5', 'vdiffd0_mmc5', 'vdiffq0_c_mmc5', 'vdiffd0_c_mmc5', 'isum0_mmc5', 'vsum0_mmc5', 'vDC0_mmc5', 'etheta0_mmc5','vnq0_mmc6', 'vnd0_mmc6', 'vnq0_c_mmc6', 'vnd0_c_mmc6', 'idiffq0_mmc6', 'idiffd0_mmc6', 'idiffq0_c_mmc6', 'idiffd0_c_mmc6', 'vdiffq0_mmc6', 'vdiffd0_mmc6', 'vdiffq0_c_mmc6', 'vdiffd0_c_mmc6', 'isum0_mmc6', 'vsum0_mmc6', 'vDC0_mmc6', 'etheta0_mmc6','vq0_th1', 'vd0_th1', 'iq0_th1', 'id0_th1','vq0_th2', 'vd0_th2', 'iq0_th2', 'id0_th2']

X['IPCA']=CCRC[0]
X['IPCB']=CCRC[1]
X['IPCC']=CCRC[2]
X['IPCD']=CCRC[3]
X['IPCE']=CCRC[4] 
X['IPCF']=CCRC[5] # F is E

#%% DATA CLEAN

Sb, Vb, Fb, Ib, Zb, Wb = system_bases()


exec(open('../Datasets/columns_groups'+reproduce_paper+'.sav').read().replace('df','X'))


exec(open('../Datasets/feature_creation'+reproduce_paper+'.sav').read().replace('df','X'))


exec(open(path+'PFI_features_'+model_name+reproduce_paper+'.sav').read().replace('df','X'))#_fbeta40perc

if reproduce_paper=='_paper':

    PFI_features = PFI_dict[model_name]
else:
    PFI_features = PFI_dict

stab=model.predict(X[PFI_features])

    