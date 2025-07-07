import pandas as pd

def create_dataset_with_selected_CCRCs(path_data,path_source, CCRCs_list,combinations, save_dataset=True, filename='df_selected_combinations.csv'):

    df=pd.DataFrame()
    df_CCRC=pd.DataFrame()
      
    for i in CCRCs_list: 
        
        X_PF = pd.read_excel(path_source+'X_PF_CCRC_{0:.0f}.xlsx'.format(i))    
        X_IPC = pd.read_excel(path_source+'X_IPC_CCRC_{0:.0f}.xlsx'.format(i))
        Y_stab_h2_dcgain = pd.read_excel(path_source+'Stab_H2_DCgain_CCRC_'+str(i)+'.xlsx')

        df_CCRC=pd.concat([X_PF,X_IPC],axis=1)
    
        df_CCRC['Combination']=i
        df_CCRC['IPCA'] = combinations[i-1][0]
        df_CCRC['IPCB'] = combinations[i-1][1]
        df_CCRC['IPCC'] = combinations[i-1][2]
        df_CCRC['IPCD'] = combinations[i-1][3]
        df_CCRC['IPCE'] = combinations[i-1][4]
        df_CCRC['IPCF'] = combinations[i-1][5]
        df_CCRC['Stable']=Y_stab_h2_dcgain['Stab']
        df_CCRC['H2_freq']=Y_stab_h2_dcgain['H2_freq']
        df_CCRC['H2_vdc']=Y_stab_h2_dcgain['H2_vdc']
        df_CCRC['DCgain_vdc']=Y_stab_h2_dcgain['DCgain_vdc']
        df_CCRC['DCgain_freq']=Y_stab_h2_dcgain['DCgain_freq']
        
        df=pd.concat([df,df_CCRC],axis=0)
     
    df=df.reset_index(drop=True)
    df.to_csv(path_data+filename, index=False, header=True)    

    return df