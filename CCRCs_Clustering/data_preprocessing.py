import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
#%%

def preprocess_data(df_X_PF_IPC):
    #TODO: fix this in the files
    df_X_PF_IPC.rename(columns = {'P2l':'Pl2', 'P5l':'Pl5', 'P9l':'Pl9','Q2l':'Ql2', 'Q5l':'Ql5', 'Q9l':'Ql9'}, inplace = True)
    
    Sb, Vb, Fb, Ib, Zb, Wb = system_bases()
    
    columns_V, columns_PQ, columns_I = group_columns(df_X_PF_IPC)

    columns_remove = set()
    rows_remove = set()
    #% Cnversion in p.u.
    for v in columns_V:
        df_X_PF_IPC[v+'_pu'] = df_X_PF_IPC[v].apply(lambda x: x/Vb)
    for v in columns_PQ:
        df_X_PF_IPC[v+'_pu'] = df_X_PF_IPC[v].apply(lambda x: x/Sb)
    for v in columns_I:
        df_X_PF_IPC[v+'_pu'] = df_X_PF_IPC[v].apply(lambda x: x/Ib)
        
    # Remove non p.u. columns
    for c in columns_V:
        remove_columns(columns_remove,[c])
    for c in columns_PQ:
        remove_columns(columns_remove,[c])
    for c in columns_I:
        remove_columns(columns_remove,[c])
        
    # data clean

    remove_columns(columns_remove,['Combination'])
    remove_columns(columns_remove,['Powerflow'])

    # Remove _c_ columns
    removed_c_ = []
    for c in df_X_PF_IPC:
        if '_c_' in c:
            remove_columns(columns_remove,[c])
            removed_c_.append(c)

    removed_only1val = []
    # Remove columns with only 1 value
    for c in df_X_PF_IPC:
        if df_X_PF_IPC[c].unique().size == 1:
            remove_columns(columns_remove,[c])
            removed_only1val.append(c)
        
    removed_duplicated = []
    # Remove duplicated columns
    for c in df_X_PF_IPC.columns[df_X_PF_IPC.columns.duplicated()]:
        remove_columns(columns_remove,[c])
        removed_duplicated.append(c)
        
    # Remove correlated columns  
    correlated_features = get_correlated_columns(final_df(df_X_PF_IPC,columns_remove,set()))      
    removed_correlated = []
    
    for var in correlated_features:
        var_node=[None, None]
        var_type=[None, None]
        # P Pdc, Q, i isum, V, v vDC vsum, theta, etheta 
        type_priorities=['P','Q','i','V','v','t','e']
        always_conserve=['P','Q']
        mmc_nodes=[2,4,6,5,9,10]
        th_nodes=[1,8]
        g_nodes=[3,6,11]

        var_priority=[None, None]
    
        for i, v in enumerate(var):
            #type
            var_type[i]=v[0]
    
            #node
            if 'mmc' in v: #mmc
                var_node[i] = mmc_nodes[int(v[v.index('mmc')+3])-1]
            elif 'dc' in v: #dc associated to mmc
                var_node[i] = mmc_nodes[int(v[v.index('dc')-1])-1]
            elif ('th' in v) and not ('theta' in v): #thevenin
                var_node[i] = th_nodes[int(v[v.index('th')+2])-1]
            elif ('g' in v) and not ('ang' in v): #generator
                var_node[i] = g_nodes[int(v[v.index('g')+1])-1]
            elif '_pu' in v: #variable in pu
                var_node[i] = int(v[-4])
            else:
                var_node[i] = int(v[-1])
                                
            #priority
            if var_type[i] in always_conserve:
                var_priority[i]=0
            else:
                var_priority[i]=type_priorities.index(var_type[i])
            
    
        if var_node[0] != var_node[1]: # don't remove different nodes
            continue
    
        if var_priority[0]==0 and var_priority[1]==0: #priority 0 == don't remove
            continue
    
        if var_priority[0] < var_priority[1]: # remove variable with highest priority
            remove_columns(columns_remove,[var[1]])
            removed_correlated.append(var[1])
        else:
            remove_columns(columns_remove,[var[0]])
            removed_correlated.append(var[0])

    # Modulo vq vd y theta
    columns_vq = final_df(df_X_PF_IPC,columns_remove,set()).columns[final_df(df_X_PF_IPC,columns_remove,set()).columns.str.startswith('vq0') +
                                      final_df(df_X_PF_IPC,columns_remove,set()).columns.str.startswith('vdiffq0') +
                                      final_df(df_X_PF_IPC,columns_remove,set()).columns.str.startswith('vnq0')].tolist()
    columns_vd = final_df(df_X_PF_IPC,columns_remove,set()).columns[final_df(df_X_PF_IPC,columns_remove,set()).columns.str.startswith('vd0') +
                                      final_df(df_X_PF_IPC,columns_remove,set()).columns.str.startswith('vdiffd0') +
                                      final_df(df_X_PF_IPC,columns_remove,set()).columns.str.startswith('vnd0')].tolist()

    # Make sure it exists vd0 for each vq0
    for vq in columns_vq:
        index_q = vq.find('q0')
        vq_to_d = vq[:index_q]+'d'+vq[index_q+1:]
        # vq_to_d = vq[:1] + 'd' + vq[1+1:]
        if vq_to_d in columns_vd:
            feature_creation_2var(df_X_PF_IPC, vq, vq_to_d, relation='module', id_name='v'+vq[index_q+2:])
            # theta = atan(d/q)
            feature_creation_2var(df_X_PF_IPC, vq_to_d, vq, relation='angle', id_name='v'+vq[index_q+2:])
            
    for e in columns_vq:
        remove_columns(columns_remove,[e])
    for e in columns_vd:
        remove_columns(columns_remove,[e])
        
        
    # Modulo iq id y theta
    columns_iq = final_df(df_X_PF_IPC,columns_remove,set()).columns[final_df(df_X_PF_IPC,columns_remove,set()).columns.str.startswith('iq0') +
                                      final_df(df_X_PF_IPC,columns_remove,set()).columns.str.startswith('idiffq0') +
                                      final_df(df_X_PF_IPC,columns_remove,set()).columns.str.startswith('inq0')].tolist()
    columns_id = final_df(df_X_PF_IPC,columns_remove,set()).columns[final_df(df_X_PF_IPC,columns_remove,set()).columns.str.startswith('id0') +
                                      final_df(df_X_PF_IPC,columns_remove,set()).columns.str.startswith('idiffd0') +
                                      final_df(df_X_PF_IPC,columns_remove,set()).columns.str.startswith('ind0')].tolist()

    # Make sure it exists id0 for each iq0
    for iq in columns_iq:
        index_q = iq.find('q0')
        iq_to_d = iq[:index_q]+'d'+iq[index_q+1:]
        if iq_to_d in columns_id:
            feature_creation_2var(df_X_PF_IPC, iq, iq_to_d, relation='module', id_name='i'+iq[index_q+2:])
            feature_creation_2var(df_X_PF_IPC, iq_to_d, iq, relation='angle', id_name='i'+iq[index_q+2:])
            
    for e in columns_iq:
        remove_columns(columns_remove,[e])
    for e in columns_id:
        remove_columns(columns_remove,[e])
        
        
    # Potencia aparente
    columns_P = final_df(df_X_PF_IPC,columns_remove,set()).columns[final_df(df_X_PF_IPC,columns_remove,set()).columns.str.startswith('P')].tolist()
    columns_Q = final_df(df_X_PF_IPC,columns_remove,set()).columns[final_df(df_X_PF_IPC,columns_remove,set()).columns.str.startswith('Q')].tolist()
    # Make sure it exists P for each Q
    for P in columns_P:
        P_to_Q = 'Q' + P[1:]
        if P_to_Q in columns_Q:
            feature_creation_2var(df_X_PF_IPC, P, P_to_Q, relation='module', id_name='S'+P[1:])


     # Categorical features   
    categorical = {'lt 0': lambda x: x < 0}

    for v in columns_P:
        df_X_PF_IPC[v+' lt 0'] = df_X_PF_IPC[v].apply(lambda x: x<0)
        df_X_PF_IPC['abs_'+v] = abs(df_X_PF_IPC[v])

    for v in columns_Q:
        df_X_PF_IPC[v+' lt 0'] = df_X_PF_IPC[v].apply(lambda x: x<0)
        df_X_PF_IPC['abs_'+v] = abs(df_X_PF_IPC[v])


    
    
    X = final_df(df_X_PF_IPC,columns_remove,rows_remove)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X

def group_columns(df):
    columns_V =  df.columns[df.columns.str.startswith('V')+ df.columns.str.startswith('v')].tolist()
    columns_PQ = df.drop('Powerflow',axis=1, errors='ignore').columns[df.drop('Powerflow',axis=1, errors='ignore').columns.str.startswith('P') + df.drop('Powerflow',axis=1, errors='ignore').columns.str.startswith('Q')].tolist()
    columns_I = df.columns[df.columns.str.startswith('i')].tolist()

    return columns_V, columns_PQ, columns_I

def feature_creation_2var(df, var1, var2, relation='module', id_tag=None, id_name=None):
    
    name = var1+'_'+var2 if id_name=='' else id_name
    func = None
    tag = ''
    
    # if relation[:9]=='sumofpow_':
    #     tag = 'sop'+str(n)+'_' if id_tag==None else id_tag
    #     n = int(relation[9:])
    #     func = lambda x1,x2: np.power(x1,n) + np.power(x2,n)
    
    if relation=='module':
        tag = 'mod_' if id_tag==None else id_tag
        func = lambda x1,x2: np.sqrt(np.power(x1,2) + np.power(x2,2))
    
    elif relation=='angle':
        tag = 'ang_' if id_tag==None else id_tag
        func = lambda x1,x2: np.arctan(x1/x2)
        
    elif relation=='circle':
        tag = 'circ_' if id_tag==None else id_tag
        func = lambda x1,x2: np.sqrt(np.power(x1,2) + np.power(x2,2))
        
    elif relation=='oblique' or relation=='sum':
        tag = 'sum_' if id_tag==None else id_tag
        func = lambda x1,x2: x1 + x2
        
    elif relation=='quadrants':
        tag = 'prod_' if id_tag==None else id_tag
        func = lambda x1,x2: x1*x2
     
    df[tag+name] = func(df[var1], df[var2])
    
    return

def remove_rows(rows_remove,*row_list): # Rows to remove from the df
    rows_remove.update(row_list)
def remove_columns(columns_remove,column_list): # Columns to remove from the df
    columns_remove.update(column_list)

def get_correlated_columns(df, c_threshold=0.999, method='pearson'):

    correlated_features = []
    correlation = df.corr(method=method)
    for i in correlation.index:
        for j in correlation:
            if i!=j and abs(correlation.loc[i,j])>=c_threshold:
                if tuple([j,i]) not in correlated_features:
                    correlated_features.append(tuple([i,j]))
                    
    return correlated_features
    
def final_df(df,columns_remove,rows_remove): # Return final df (w/o removed columns and rows)
    return df.drop(columns_remove, axis=1, errors='ignore').drop(rows_remove, axis=0, errors='ignore')

def system_bases():
	Sb=500e6
	Vb=280e3
	Fb=50
	Ib=Sb/(np.sqrt(3)*Vb)
	Zb=Vb**2/Sb
	Wb=2*np.pi*Fb
	
	return Sb, Vb, Fb, Ib, Zb, Wb 
