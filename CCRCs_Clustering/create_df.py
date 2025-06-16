import pandas as pd
import numpy as np

def PFs_x_CCRCs(n_powerflows,indicator,combinations,path_data):
    df_pf_ind=pd.DataFrame()
    for c in range(1,len(combinations)+1):
        df_stab_inds = pd.read_excel(path_data+'Stab_H2_DCgain_CCRC_'+str(c)+'.xlsx')
    
        df_pf_ind[['Combination_{0:.0f}'.format(c)]]= df_stab_inds[[indicator]]
        if 'DCgain' in indicator:
            df_pf_ind.loc[df_stab_inds.query('Stab == 0').index, 'Combination_{0:.0f}'.format(c)]=65535
            
    return df_pf_ind.reset_index(drop=True)

def df_1st_exploration(df_pf, CCRCs_list, combinations, n_pf, path_data, save_file=True):
        
    df=pd.DataFrame()  
    for i in CCRCs_list:
        df_pf['Combination']=i
        df_pf['IPCA'] = combinations[i-1][0]
        df_pf['IPCB'] = combinations[i-1][1]
        df_pf['IPCC'] = combinations[i-1][2]
        df_pf['IPCD'] = combinations[i-1][3]
        df_pf['IPCE'] = combinations[i-1][4]
        df_pf['IPCF'] = combinations[i-1][5]
        df_pf['Powerflow']=np.arange(1,n_pf+1)
        df_stab_inds = pd.read_excel(path_data+'Stab_H2_DCgain_CCRC_'+str(i)+'.xlsx')
        df_pf['Stable']=df_stab_inds['Stab']
        df_pf['H2_freq']=df_stab_inds['H2_freq']
        df_pf['H2_vdc']=df_stab_inds['H2_vdc']
        df_pf['DCgain_freq']=df_stab_inds['DCgain_freq']
        df_pf['DCgain_vdc']=df_stab_inds['DCgain_vdc']
        
        df=pd.concat([df,df_pf],axis=0)
    
    df=df.reset_index(drop=True)
            
    if save_file:
        df.to_csv(path_data+'df_1st_exploration'+'.csv', index=True, index_label='Case')    
    
    return df
            
