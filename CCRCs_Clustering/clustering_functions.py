from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

import xlsxwriter 

from utils import *
from create_df import *
from pu_conversion import *
from knowledge_extraction_from_cluster import *
from CCRCs_clusters_selection import *

def final_df(df,columns_remove,rows_remove): # Return final df (w/o removed columns and rows)
    return df.drop(columns_remove, axis=1, errors='ignore').drop(rows_remove, axis=0, errors='ignore')

#%%

def CCRCs_clustering(CCRCs2clustering,heatmap):
        #df_X_PF_IPC, df_pf_ind,indicator, combinations,n_powerflows,path_data,indicators_plot_labels):
    sil_score=np.zeros([len(CCRCs2clustering),1])
    
    for i in range(2,len(CCRCs2clustering)+1):
        est=KMeans(n_clusters=i, n_init=15, max_iter=500, init='k-means++')#'random')#
        est.fit(heatmap)
    
        labels = est.labels_
    
        sil_score[i-2,0]=silhouette_score(heatmap, labels, metric='euclidean')
   
    
    n_clusters=np.argmax(sil_score)+2
    
    #%% Final clustering
    
    est=KMeans(n_clusters, n_init=15, max_iter=500, init='k-means++')#'random')#
    
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
                     excluding_totally_unst_comb=True, plot=True):
    
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
    
            
    


