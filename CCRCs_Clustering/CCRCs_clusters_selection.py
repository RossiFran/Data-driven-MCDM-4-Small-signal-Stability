import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import seaborn as sns

from matplotlib.patches import Rectangle
import random
import matplotlib.cm as cm

def clusters_selection(df_CCRs_PF_ind_stable,sorted_clusters_rules,labels):

    pf_list=[]
    selected_clusters=pd.DataFrame()
    df_CCRs_PF_ind_stable_pu = df_CCRs_PF_ind_stable.copy(deep=True)
    df_CCRs_PF_ind_stable_pu[['Pg1','Pg2','Pg3','Qg1','Qg2','Qg3','Pl2','Ql2','Pl5','Ql5','Pl7','Ql7','Pl9','Ql9','Pmmc3']] = df_CCRs_PF_ind_stable_pu[['Pg1','Pg2','Pg3','Qg1','Qg2','Qg3','Pl2','Ql2','Pl5','Ql5','Pl7','Ql7','Pl9','Ql9','Pmmc3']]/500e6
    
    for i in range(0,len(sorted_clusters_rules)):
        cl=sorted_clusters_rules.loc[i,'cluster']
        
        ind_cl=list(labels.query('lab == @cl')['Combination'])
        #ind_cl=list(labels.query('lab == @cl').index) ## USAR ['Unnamed: 0'] y no index si cojes las labels del excel
    
        
        rule=sorted_clusters_rules.loc[i,'rules']
        rule=rule.replace('->','and')
        rule=rule.replace('"','')
        rule=rule[:-5]
        pf=df_CCRs_PF_ind_stable_pu.query('Combination == @ind_cl').query(rule)['Powerflow'].unique()
        
        flag=0
        for p in pf:
            if p in pf_list:
                continue
            else:
                flag=1
                pf_list.append(p)
                
        if flag==1:
            selected_clusters.loc[i,'cluster']=cl#.append(cl)
            selected_clusters.loc[i,'Combinations']=str(ind_cl)#.append(cl)
            selected_clusters.loc[i,'rule']=rule#.append(cl)
            selected_clusters.loc[i,'ind value']=sorted_clusters_rules.loc[i,'val']#.append(cl)
            
            
            
        if len(pf_list)==100:
            break
     
    selected_clusters_unique=np.unique(np.array(selected_clusters['cluster']))
    
    return selected_clusters, selected_clusters_unique

def cluster_indicator_avg_level(cl_list, labels, df_pf_ind_levels):
    
    lab_ind_avg= pd.DataFrame(columns=['cluster_lab','indicator_avg','num_CCRCs'])
    for n in cl_list:
        comb_cl=list(labels.query('lab==@n')['Combination'])
        ind_avg_level_in_CCRC=[]
    
        for c in comb_cl:
            df_CCRC=df_pf_ind_levels[['Combination_{0:.0f}'.format(c)]]
            ind_avg_level_in_CCRC.append(df_CCRC.mean())
            
        lab_ind_avg.loc[n,'cluster_lab']=n
        lab_ind_avg.loc[n,'indicator_avg']=np.array(ind_avg_level_in_CCRC).mean()  
        lab_ind_avg.loc[n,'num_CCRCs']=len(comb_cl)
            
    lab_ind_avg=lab_ind_avg.sort_values(by='indicator_avg')
    
    return lab_ind_avg

def Rearranged_Attribute_Matrix(df_pf_ind_levels, heatmap, cl_list, labels, type_matrix=str(), plot=False,indicators_plot_labels=None, indicator=None, path='./Results/', save_plot=False):
    
    lab_ind_avg = cluster_indicator_avg_level(cl_list, labels, df_pf_ind_levels)
    
    sorter=list(lab_ind_avg['cluster_lab'])
    idx_comb=[]
    for s in sorter:
        idx_comb.append(list(labels.query('lab == @s').index))
        
    idx_comb=np.hstack(idx_comb)
    
    heatmap_rearranged_rows=heatmap[idx_comb,:]
    
    heatmap_rearranged_col_avg=pd.DataFrame(heatmap_rearranged_rows.mean(axis=0))
    heatmap_rearranged_col_avg.columns=['avg']
    heatmap_rearranged_col_avg=heatmap_rearranged_col_avg.sort_values(by='avg')
    heatmap_rearranged=heatmap_rearranged_rows[:,heatmap_rearranged_col_avg.index]
    
    if plot:
        plot_heatmap(heatmap_rearranged, indicators_plot_labels, indicator, path, save_plot, type_matrix)
    
    return heatmap_rearranged  
    

def plot_heatmap(heatmap_rearranged, indicators_plot_labels, indicator, path, save_plot, type_matrix):
    cmap = matplotlib.cm.hot
    cmap_reversed = matplotlib.cm.get_cmap('hot',100)
    
    red = np.array([139/255,37/255,0,1])
    
    yellow=np.array([1,1,0.3,1])
    green=np.array([0,1,0,1])
    drkgreen=np.array([0.21,0.71,0.52,1])
    orange=np.array([1,165/255,0,1])
    orangered=np.array([1,69/255,0,1])
    
    newcolors_stab = cmap_reversed(np.linspace(0, 1, 100))
    newcolors_stab[:25, :] = green
    newcolors_stab[25:50, :] = yellow
    newcolors_stab[50:75,:] = orange
    newcolors_stab[75:99,:] = orangered
    newcolors_stab[99:,:] = red
    
    newcmp_stab = ListedColormap(newcolors_stab)
    
    xgrid = np.arange(heatmap_rearranged.shape[1])+1
    ygrid = np.arange(heatmap_rearranged.shape[0])+1
    

    fig = plt.figure(figsize=(8,8))
    bx= fig.add_subplot()
    
    bb=bx.pcolormesh(xgrid, ygrid, pd.DataFrame(heatmap_rearranged),cmap=newcmp_stab,edgecolors='w', linewidths=0.05)
    bx.set_frame_on(False) # remove all spines      
    
    plt.setp(bx.get_xticklabels(), visible=False)
    plt.setp(bx.get_yticklabels(), visible=False)
    bx.tick_params(axis='both', which='both', length=0)
    bx.set_ylabel('CCRCs', fontsize=30, labelpad=32)
    bx.set_xlabel('Stability Maps Sub-regions', fontsize=30) #[$X_A$,$X_B$,$X_C$] Ranges
    #bx.set_title('Attributes Matrix \n Rearranged according to Clusters', fontsize=30)
    bx.set_title('Rearranged Attributes Matrix', fontsize=30)
    
    cb=fig.colorbar(bb, ax=bx,ticks=[1,2,3,4,5])
    cb.set_label(indicators_plot_labels[indicator], size=25, rotation=270, labelpad=25)
    cb.ax.tick_params(labelsize=25)
    
    fig.tight_layout()
    
    if save_plot:   
        plt.savefig(path+'Rearrang_att_matrix_'+indicator+type_matrix+'.pdf')
    
def set_intersection(n_powerflows, indicators_list, path, plot=False, save_plot=False):
    
    clusters_results=dict()
    selected_clusters = dict()
    ind_cl_sel = dict()
    ind_cl_not_sel = dict()
    cluster_summary=dict()

    clusters_filter=pd.DataFrame(columns=['CCRC']+indicators_list)
    
    for indicator in indicators_list:
        
        # TODO: check this od unnamed:0 
        clusters_results[indicator] = pd.read_excel(path+'Clustering_results_'+indicator+'.xlsx',sheet_name='comb_clusters').sort_values(by='Unnamed: 0').reset_index(drop=True)
        selected_clusters[indicator]=list(pd.read_excel(path+'Clustering_results_'+indicator+'.xlsx',sheet_name='selected_clusters')['cluster'].unique())
        
        sel_cl_list=selected_clusters[indicator]
        ind_cl_sel[indicator]=clusters_results[indicator].query('lab == @sel_cl_list').index
        ind_cl_not_sel[indicator]=clusters_results[indicator].query('lab != @sel_cl_list').index

        clusters_results[indicator].loc[ind_cl_sel[indicator],'marker']='*'
        clusters_results[indicator].loc[ind_cl_not_sel[indicator],'marker']='o'

        cluster_summary[indicator]=dict()
        cluster_summary[indicator]['cls_res']=clusters_results[indicator]
        cluster_summary[indicator]['ind_sel']=ind_cl_sel[indicator]
        cluster_summary[indicator]['ind_nonsel']=ind_cl_not_sel[indicator]

        # TODO: change this
        clusters_filter['CCRC']= clusters_results[indicator]['Combination']
        clusters_filter[indicator]=clusters_results[indicator]['marker']
        
        clusters_filter.loc[list(clusters_filter.query(indicator+' == "*"').index),indicator]=clusters_results[indicator].loc[list(clusters_filter.query(indicator+ '== "*"').index),'lab']
         
    clusters_filter_unique=clusters_filter.iloc[clusters_filter.drop('CCRC',axis=1).drop_duplicates().index]
    mask = clusters_filter_unique[indicators_list].eq("o").all(axis=1)
    clusters_filter_unique = clusters_filter_unique[~mask]
        
    CCRC_dict=dict({'selected_CCRC':[],'associated_CCRC':[], 'rules':[]})
    
    for i in list(clusters_filter_unique.index):
        
        CCRC = clusters_filter_unique.loc[i,'CCRC']
        CCRC_dict['selected_CCRC'].append(CCRC)
        
        sequence=[]
        for indicator in indicators_list:
            sequence.append(clusters_filter_unique.loc[i,indicator])
        
        df = clusters_filter[indicators_list] == sequence
        rows_all_true = df[df.all(axis=1)].index
         
        CCRC_dict['associated_CCRC'].append(list(set(clusters_filter.loc[rows_all_true,'CCRC']) - set([CCRC])))
        
    if plot:
        plt_set_intersection(CCRC_dict, indicators_list, cluster_summary, clusters_results, save_plot, path)
        
    return CCRC_dict
        
def plt_set_intersection(CCRC_dict, indicators_list, cluster_summary, clusters_results, save_plot, path='./Results/'):
    fs=20
    plt.rcParams.update({"figure.figsize" : [16,3.5],
                         "text.usetex": True,
                         "font.family": "serif",
                         "font.serif": "Computer Modern",
                         "axes.labelsize": fs,
                         "axes.titlesize": fs,
                         "legend.fontsize": fs,
                         "xtick.labelsize": fs,
                         "ytick.labelsize": fs,
                         "savefig.dpi": 130,
                        'legend.fontsize': fs,
                        'legend.handlelength': 2,
                        'legend.loc': 'upper right'})
    
    cmap = cm.get_cmap('tab20c', 256)  # Get the colormap
    
    fig=plt.figure()
    ax=fig.add_subplot()
    # ax.grid()
    
    random_idx=0
    for k in range(0,len(CCRC_dict['selected_CCRC'])):
      
        i=CCRC_dict['selected_CCRC'][k]
        cc=[random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)]
        ax.add_patch(Rectangle((i-0.5, 0.7), 1, 4,
                 facecolor = cmap(random_idx), alpha=1,
                 fill=True,
                 lw=0))
        for j in CCRC_dict['associated_CCRC'][k]:
            ax.add_patch(Rectangle((j-0.5, 0.7), 1, 4,
                 facecolor = cmap(random_idx), alpha=1,
                 fill=True,
                 lw=0))
        random_idx=random_idx+20
    ax.set_xticks(list(CCRC_dict['selected_CCRC'])) 
    
    for pos, indicator in enumerate(indicators_list):
        
        if 'H2' in indicator:
            color_marker= 'k'
        else:
            color_marker= 'gray'
    
        
        cl_df= cluster_summary[indicator]['cls_res']
        ind_sel= cluster_summary[indicator]['ind_sel']
        ind_nonsel= cluster_summary[indicator]['ind_nonsel']
        
        x=cl_df['Combination'].iloc[ind_sel]
        y=np.ones([len(ind_sel),1])*(int(pos)+1)
        ax.scatter(x,y,s=350,c=color_marker, marker='d')#list(clusters_h2_vdc['marker']))
        
        # loop through each x,y pair
        for k in range(0,len(x)):
        #for i,j in zip(x,y):
            i=np.array(x)[k]
            j=y[k][0]
            corr = -.05 # adds a little correction to put annotation in marker's centrum
            ax.annotate(np.array(cl_df['lab'].iloc[ind_sel])[k]+1,  xy=(i -0.3 , j + corr), c='w')
        
        x=clusters_results[indicator].loc[ind_nonsel,'Combination']
        y=np.ones([len(ind_nonsel),1])*(int(pos)+1)
        ax.scatter(x,y,s=350,c='k', marker='d')#list(clusters_h2_vdc['marker']))
        ax.scatter(x,y,s=250,c='w', marker='d')#list(clusters_h2_vdc['marker']))
        
        # loop through each x,y pair
        for k in range(0,len(x)):
        #for i,j in zip(x,y):
            i=np.array(x)[k]
            j=y[k][0]
            #corr = -0.05 # adds a little correction to put annotation in marker's centrum
            ax.annotate(0,  xy=(i - 0.3, j - 0.05))
    
    
    
    ax.set_ylim([0.5,4.5])
    ax.set_yticks([1,2,3,4])#,2,2.25])
    xticks=list(np.arange(1,100,5))
    xticks[-1]=95
    ax.set_xticks(xticks)#,2,2.25])
    ax.set_xticklabels([str(i) for i in xticks])
    ax.set_yticklabels(['$\hat\mathcal{H}_{V_{DC}}$','$\hat\mathcal{K}_{V_{DC}}$','$\hat\mathcal{H}_{f}$','$\hat\mathcal{K}_{freq}$'])#,'Stab','$\hat\mathcal{H}_{En}$'
    ax.tick_params(axis='y')
    ax.set_xlabel('CCRC')
    fig.tight_layout()
    
    if save_plot:
        plt.savefig(path+'indicators_intersection.pdf', format='pdf', bbox_inches='tight')
    

    
    
