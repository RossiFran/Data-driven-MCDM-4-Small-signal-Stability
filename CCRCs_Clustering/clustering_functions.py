from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

import xlsxwriter 

#from utils import *
from create_df import *
from pu_conversion import *
from knowledge_extraction_from_cluster import *
from CCRCs_clusters_selection import *

def final_df(df,columns_remove,rows_remove): # Return final df (w/o removed columns and rows)
    return df.drop(columns_remove, axis=1, errors='ignore').drop(rows_remove, axis=0, errors='ignore')

#%%

def CCRCs_clustering(CCRCs2clustering,heatmap,seed):
        #df_X_PF_IPC, df_pf_ind,indicator, combinations,n_powerflows,path_data,indicators_plot_labels):
    sil_score=np.zeros([len(CCRCs2clustering),1])
    
    for i in range(2,len(CCRCs2clustering)+1):
        est=KMeans(n_clusters=i, n_init=15, max_iter=500, init='k-means++',random_state=seed)#'random')#
        est.fit(heatmap)
    
        labels = est.labels_
    
        sil_score[i-2,0]=silhouette_score(heatmap, labels, metric='euclidean')
   
    
    n_clusters=np.argmax(sil_score)+2
    
    #%% Final clustering
    
    est=KMeans(n_clusters, n_init=15, max_iter=500, init='k-means++',random_state=seed)#'random')#
    
    est.fit(heatmap)
    
    labels = est.labels_
    labels=pd.DataFrame(labels)
    labels.columns=['lab']
    labels['Combination']=CCRCs2clustering 
    labels=labels.sort_values(by='lab')
    
    return labels, n_clusters

def preprocess_data(df_X_PF_IPC):
    #TODO: fix this in the files
    df_X_PF_IPC.rename(columns = {'P2l':'Pl2', 'P5l':'Pl5', 'P9l':'Pl9','Q2l':'Ql2', 'Q5l':'Ql5', 'Q9l':'Ql9'}, inplace = True)
    
    df_X_PF_IPC = pu_conversion(df_X_PF_IPC)
    
    columns_I, columns_PQ, columns_V, columns_vd, columns_vq, columns_id, columns_iq, columns_P, columns_Q = group_columns()
    columns_remove, rows_remove = data_clean_results()
    
    X = final_df(df_X_PF_IPC,columns_remove,rows_remove)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X

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
    
    
def PCA_cluster(X, n_intervals, x_column = 'X1_PCA', y_column = 'X2_PCA', z_column = 'X3_PCA', plot=False):
    
    pca= PCA()
    X_PCA = pca.fit_transform(X)
    exp_var_pca = pca.explained_variance_ratio_
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.bar(range(1,len(exp_var_pca)+1), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.tick_params(axis='x')
        plt.tick_params(axis='y')
        ax.set_xlim(0,20)
        ax.set_xticks([1,2,3])
        ax.set_xticklabels(['$X_A$','$X_B$','$X_C$'])
        plt.tight_layout()
                
    df_X_PCA=pd.DataFrame(X_PCA[:,0:3], columns=[x_column,y_column,z_column])
    
    x_list = np.linspace(df_X_PCA[x_column].min(), df_X_PCA[x_column].max(), n_intervals+1)
    y_list = np.linspace(df_X_PCA[y_column].min(), df_X_PCA[y_column].max(), n_intervals+1)
    z_list = np.linspace(df_X_PCA[z_column].min(), df_X_PCA[z_column].max(), n_intervals+1)
 
    return df_X_PCA, x_list, y_list, z_list

def heatmap_function(n_intervals, x_list, y_list, z_list, indicator, 
                     df_X_PCA, df_CCRs_PF_ind, df_pf_ind, 
                     combinations,indicators_plot_labels, 
                     excluding_totally_unst_comb=True, plot=False):
    
    x_column = df_X_PCA.columns[0]
    y_column = df_X_PCA.columns[1]
    z_column = df_X_PCA.columns[2]
    intervals=[]
    for iy in range(n_intervals):
        Ymax = y_list[iy+1]
        Ymin = y_list[iy]
        for ix in range(n_intervals):
            Xmax = x_list[ix+1]
            Xmin = x_list[ix]
            for iz in range(n_intervals):
                Zmax = z_list[iz+1]
                Zmin = z_list[iz]
                
                n_pf=len(df_X_PCA.query('(@Ymin <= {} < @Ymax) & (@Xmin <= {} < @Xmax) & (@Zmin <= {} < @Zmax)'.format(y_column, x_column, z_column)))
                if n_pf>0:
                    intervals.append([Ymin,Ymax,Xmin,Xmax,Zmin,Zmax,n_pf])
                    
    intervals=np.array(intervals)
    
    if not excluding_totally_unst_comb:
        CCRCs2clustering=list(np.arange(1,len(combinations)+1))
        comb_totally_unstab=[]
    
    else:
        CCRCs2clustering=list(np.unique(df_CCRs_PF_ind.query('Stable == 1')['Combination']))
        CCRCs_totally_unstab=list(set(list(np.arange(1,len(combinations)+1)))-set(CCRCs2clustering))
    
    df_pf_ind=df_pf_ind.drop(['Combination_'+str(c) for c in CCRCs_totally_unstab], axis=1)
       
    df_pf_ind_levels=df_pf_ind.copy(deep=True)
    
        
    if indicator!='Stab':
        
        indicator_stable=np.array(df_pf_ind.replace(65535,np.nan)).ravel()
        mask = np.isnan(indicator_stable)
        
        indicator_describe=pd.DataFrame(indicator_stable[~mask]).describe()

        if plot:
            fig=plt.figure()
            ax=fig.add_subplot()
            ax = sns.boxplot(data=pd.DataFrame(indicator_stable[~mask]), orient="h", palette="Set2")
    
            plt.ylabel(indicators_plot_labels[indicator])
            ax.set_xticks([indicator_describe.iloc[3][0],indicator_describe.iloc[7][0]])#,n_clusters,
            ax.set_yticks([])#,n_clusters,
            #ax.set_xticklabels(['0.12','75'], fontsize=30)
            plt.tight_layout()
    
    
        for i in [7,6,5,4]:
            imin=indicator_describe.iloc[i-1][0]
            imax=indicator_describe.iloc[i][0]
            for c in CCRCs2clustering:
                ind=df_pf_ind.query('@imin < Combination_{} <= @imax'.format(c)).index
                df_pf_ind_levels.loc[ind,'Combination_{0:.0f}'.format(c)]=i-3
        
        for c in CCRCs2clustering:#range(1,96):
            ind=df_pf_ind.query('Combination_{} == 65535'.format(c)).index
            df_pf_ind_levels.loc[ind,'Combination_{0:.0f}'.format(c)]=5
    
        for c in CCRCs2clustering:#range(1,96):
            ind=df_pf_ind.query('Combination_{} == @imin'.format(c)).index
            df_pf_ind_levels.loc[ind,'Combination_{0:.0f}'.format(c)]=1
    
    heatmap=np.zeros([len(intervals),len(CCRCs2clustering)])
    
    for i in range(0,len(intervals)):
        Ymin=intervals[i,0]
        Ymax=intervals[i,1]
        Xmin=intervals[i,2]
        Xmax=intervals[i,3]
        Zmin=intervals[i,4]
        Zmax=intervals[i,5]
            
        dfi_range = df_X_PCA.query('(@Ymin <= {} < @Ymax) & (@Xmin <= {} < @Xmax) & (@Zmin <= {} < @Zmax)'.format(y_column, x_column, z_column))
        ind_dfi_range=dfi_range.index        
        
        if len(dfi_range)==0:
                print('error')
                print(c)
                print(i)
    
        for c in range(0,len(CCRCs2clustering)):
            dfi = df_pf_ind_levels.loc[ind_dfi_range,'Combination_{0:.0f}'.format(CCRCs2clustering[c])]
            
            heatmap[i,c]=dfi.mean()
    
    return heatmap.T, CCRCs2clustering, df_pf_ind_levels
    
            
    


