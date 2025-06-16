import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

def knowledge_extraction(df_CCRs_PF_ind, labels, combinations, indicator):
    df_CCRs_PF_ind_stable=df_CCRs_PF_ind.query('Stable == 1')
    
    # h2_vdc_scaler=MinMaxScaler()
    # h2_freq_scaler=MinMaxScaler()
    
    
    # df_CCRs_PF_ind_stable['H2_vdc_sc']=h2_vdc_scaler.fit_transform(df_CCRs_PF_ind_stable[['H2_vdc']])
    # df_CCRs_PF_ind_stable['H2_freq_sc']=h2_freq_scaler.fit_transform(df_CCRs_PF_ind_stable[['H2_freq']])
    
    # df_CCRs_PF_ind['H2_vdc_sc']=df_CCRs_PF_ind_stable['H2_vdc_sc']
    # df_CCRs_PF_ind['H2_freq_sc']=df_CCRs_PF_ind_stable['H2_freq_sc']
    
    #%%
    
    df_CCRs_PF_ind_stable['Sth_1']=np.sqrt(df_CCRs_PF_ind_stable['Pth1']**2+df_CCRs_PF_ind_stable['Qth1']**2)/500e6
    df_CCRs_PF_ind_stable['Sth_2']=np.sqrt(df_CCRs_PF_ind_stable['Pth2']**2+df_CCRs_PF_ind_stable['Qth2']**2)/500e6
    
    df_CCRs_PF_ind_stable['Sd_ac1']=np.sqrt((df_CCRs_PF_ind_stable['Pl2']+df_CCRs_PF_ind_stable['Pl5'])**2+(df_CCRs_PF_ind_stable['Ql2']+df_CCRs_PF_ind_stable['Ql5'])**2)/500e6
    df_CCRs_PF_ind_stable['Sd_ac2']=np.sqrt((df_CCRs_PF_ind_stable['Pl7']+df_CCRs_PF_ind_stable['Pl9'])**2+(df_CCRs_PF_ind_stable['Ql7']+df_CCRs_PF_ind_stable['Ql9'])**2)/500e6
    
    df_CCRs_PF_ind_stable['Sg_1']=np.sqrt(df_CCRs_PF_ind_stable['Pg1']**2+df_CCRs_PF_ind_stable['Qg1']**2)/500e6
    df_CCRs_PF_ind_stable['Sg_2']=np.sqrt(df_CCRs_PF_ind_stable['Pg2']**2+df_CCRs_PF_ind_stable['Qg2']**2)/500e6
    df_CCRs_PF_ind_stable['Sg_3']=np.sqrt(df_CCRs_PF_ind_stable['Pg3']**2+df_CCRs_PF_ind_stable['Qg3']**2)/500e6
    
    rules_list=[]
    rule_dic_sunburst={}
    last_block_rule_dic_sunburst={}
    clusters_rules=pd.DataFrame()
    
    for l in np.unique(labels['lab']):
        
        ind_cl=list(labels.query('lab == @l')['Combination'])#.index)
    
        comb_cluster=np.array(combinations)[list(np.array(ind_cl)-1)]
        comb_list=list(comb_cluster)
        
        df_cl=df_CCRs_PF_ind_stable.query('Combination == @ind_cl')
        if len(df_cl)<=1:
            continue
            
        X_cl=df_cl[['Pg1','Pg2','Pg3','Qg1','Qg2','Qg3','Pl2','Ql2','Pl5','Ql5','Pl7','Ql7','Pl9','Ql9','Pmmc3']]/500e6#.reset_index(drop=True)#[['Pgtot','Pthtot','Pltot']]#final_df(df_cl)#
        Y_cl=df_cl[[indicator]]
 
        X_train=X_cl.copy().reset_index(drop=True)
        Y_train=Y_cl.copy().reset_index(drop=True)
        model_rules= DecisionTreeRegressor(min_samples_leaf=0.1)
        
        model_rules.fit(X_train,Y_train)
     
        df_leaves, node_indicator, leaf_id, features_name, n_rules = leaves_analysis(model_rules, X_train, Y_train)
        
        rules_acc, dic_rules=rules_extraction(model_rules, df_leaves,node_indicator,features_name, n_rules, leaf_id, X_train)

        rules_strings=rules_strings_for_graphviz(dic_rules)

        df_leaves['cluster']=l
        df_leaves['rules']=pd.DataFrame(rules_strings)

        clusters_rules=pd.concat([clusters_rules,df_leaves],axis=0)
            
    sorted_clusters_rules=clusters_rules.sort_values(by=['val','n_cases']).reset_index(drop=True)
    
    return sorted_clusters_rules, df_CCRs_PF_ind_stable

#%%
def leaves_analysis(model_rules,X_train, Y_train):
    df_leaves = pd.DataFrame()#columns=['leaf', 'val', 'n_cases','err'])
    # list of nodes that are leaves
    leaves_list = list(np.unique(_leaves_id(model_rules,X_train)))

    leaf_count=0
    for leaf in leaves_list:
        val,std, n_cases = _attributes_leaf(model_rules, leaf, X_train, Y_train)#rgr_train[['DI_loc']])
        
        df_leaves.loc[leaf_count,'leaf']=leaf
        df_leaves.loc[leaf_count,'val']=val[0]
        df_leaves.loc[leaf_count,'std']=std[0]
        df_leaves.loc[leaf_count,'n_cases']=n_cases
        leaf_count=leaf_count+1
    
    df_leaves=df_leaves.sort_values(by='n_cases', ascending=False).reset_index(drop=True)
    
    node_indicator = model_rules.decision_path(X_train)
    leaf_id = model_rules.apply(X_train)
    
    # X_train_tree=X_train.reset_index(drop=True)
    # Y_train_tree=Y_train.reset_index(drop=True)
    
    features_name=X_train.columns
    
    n_rules=len(df_leaves)
    
    return df_leaves, node_indicator, leaf_id, features_name, n_rules
    
#%% Useful functions

from sklearn.metrics import r2_score

def metric_n(model, x, y, n, metric):
    results = []
    split_len = int(len(x)/n)
    
    for i in range(n):
        x_i = x.iloc[i*split_len:(i+1)*split_len]
        y_i = y.iloc[i*split_len:(i+1)*split_len]
        
        y_pred_i = model.predict(x_i)
        if metric == precision_score:
            results.append(metric(y_i, y_pred_i, zero_division=1))
        else:
            results.append(metric(y_i, y_pred_i))
    
    return results

def accuracy_n(model, x, y, n): # Return list of accuracies by spliting x and y into n splits
# Example usage: accuracy_n(dtc_model, X_eval, Y_eval, 6) --> [0.776, 0.715, 0.782, 0.794, 0.753, 0.741]  
    return metric_n(model, x, y, n, accuracy_score)

def precision_n(model, x, y, n):
    return metric_n(model, x, y, n, precision_score)

def recall_n(model, x, y, n):
    return metric_n(model, x, y, n, recall_score)

def f1_n(model, x, y, n):
    return metric_n(model, x, y, n, f1_score)

def r2_n(model, x, y, n):
    return metric_n(model, x, y, n, r2_score)

def acc_prec_rec_f1(y, y_pred):
    a=accuracy_score(y, y_pred)
    p=precision_score(y, y_pred, zero_division=1)
    r=recall_score(y, y_pred)
    f=f1_score(y, y_pred)
    return a, p, r, f

def r2_acc_prec_rec_f1(y, y_pred):
    a=accuracy_score(y, y_pred)
    p=precision_score(y, y_pred, zero_division=1)
    r=recall_score(y, y_pred)
    f=f1_score(y, y_pred)
    return a, p, r, f

def acc_prec_rec_f1_n(model, x, y, n):
    return np.array(metric_n(model, x, y, n, acc_prec_rec_f1))

def rgr_acc_prec_rec_f1_n(model, x, y, n):
    y2=np.zeros([len(y),1])
    y3=np.array(y)
    ii=np.where(y3<1)
    y2[ii]=1
    y=pd.DataFrame(y2)    
    return np.array(metric_n(model, x, y, n, acc_prec_rec_f1))

def acc_prec_f1(y, y_pred):
    a=accuracy_score(y, y_pred)
    p=precision_score(y, y_pred, zero_division=1)
    #r=recall_score(y, y_pred)
    f=f1_score(y, y_pred)
    return a, p, f
def acc_prec_f1_n(model, x, y, n):
    return np.array(metric_n(model, x, y, n, acc_prec_f1))


#%%
def _attributes_leaf(model, leaf, X, Y):
        leaf_indexes = _samples_in_leaf(model, X, leaf)
        
        n_cases_on_leaf = len(leaf_indexes)
         
        return list(Y.loc[leaf_indexes].mean()),list(Y.loc[leaf_indexes].std()), n_cases_on_leaf

def _samples_in_leaf(model, X, leaf): 
        node_indicator = model.decision_path(X) # save the path (from root to leaf) of each sample
        leaf_id = model.apply(X) # save the index of the final leaf of each sample

        leaves=np.unique(leaf_id) # node indices of all the leaves
        leaf_id_df=pd.DataFrame(leaf_id, index=X.index)
        leaf_id_df.columns=['leaf']

        return leaf_id_df.loc[leaf_id_df['leaf']==leaf].index # return the index of the samples belonging to each leaf

def _leaves_id(model, X): 
    return model.apply(X) # save the index of the final leaf of each sample

#%%
def rules_extraction(model_rules, df_leaves,node_indicator,features_name, n_rules, leaf_id, X_train):#, X_eval_tree,Y_eval_tree):
    dic_rules={'rule':[]}
    dic_list=[]
    feature = model_rules.tree_.feature
    threshold = model_rules.tree_.threshold
    
    for i in range(0,n_rules):
        dic_rule={'feature':[],'inequality':[],'threshold':[]}
        l=df_leaves[['leaf']].iloc[i][0]
        leaf_id = model_rules.apply(X_train)
        leaf_samples=np.where(leaf_id==l)

        X_subset=X_train.iloc[leaf_samples]
        #X_subset_descr=X_subset.describe()

        sample_id = X_subset.index[0]
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[
            node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
        ]

        #print("Rules used to predict sample {id}:\n".format(id=sample_id))
        rule_string=str()

        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                
                continue

            # check if value of the split feature for sample 0 is below threshold
            if np.array(X_train)[sample_id, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            rule_string= rule_string + "{feature} {inequality} {threshold} and ".format(
                node=node_id,
                #sample=sample_id,
                feature= features_name[feature[node_id]],#X_train.columns[feature[node_id]],
                #value=np.array(X_train_tree)[sample_id, feature[node_id]],
                inequality=threshold_sign,
                threshold=threshold[node_id],
                )
            dic_rule['feature'].append(features_name[feature[node_id]])
            dic_rule['inequality'].append(threshold_sign)
            dic_rule['threshold'].append(threshold[node_id])
        rule_string=rule_string[0:len(rule_string)-5]    
        #rule_string=rule_string+' -> ' + str(df_leaves['val'][i]) +'('+str(df_leaves['std'][i])+')'
        dic_rules['rule'].append(rule_string)
        dic_list.append(dic_rule)
               
    return dic_rules, dic_list


#%%

def rules_strings_for_graphviz(dic_rules):
    rules_strings=[]
    last_box=[]
    for i in range(0,len(dic_rules)):
        rule=pd.DataFrame(dic_rules[i])
        _,idx=np.unique(rule[['feature']], return_index=True)
        unique_feat=np.array(rule.iloc[np.sort(idx)][['feature']]).ravel()
        rule_string=str()

        for j in unique_feat:#range(0,len(unique_feat))
            rule_feat=rule.query('feature == @j')
            for k in ['>','<=']:
                rule_feat_ineq=rule_feat.query('inequality == @k')
                if len(rule_feat_ineq)>0:
                    if k == '>':
                        rule_feat_ineq=rule_feat_ineq.sort_values(by='threshold',ascending=False)
                        rule_string=rule_string +'"' + j + k + str(rule_feat_ineq[['threshold']].iloc[0][0].round(3))+'"'+' -> '
                    else:
                        rule_feat_ineq=rule_feat_ineq.sort_values(by='threshold',ascending=True)
                        rule_string=rule_string +'"' + j + k + str(rule_feat_ineq[['threshold']].iloc[0][0].round(3))+'"'+' -> '
        # if df_leaves['val'][i]==1:
        #     if rules_acc['accuracy'][i]=='No test':
        #         rule_string=rule_string + '"Stable (No_test)"'
        #         last_box.append('"Stable (No_test)"')
        #     else:
        #         rule_string=rule_string + '"Stable ({0:.1f}%)"'.format(rules_acc['accuracy'][i]*100)
        #         last_box.append('"Stable ({0:.1f}%)"'.format(rules_acc['accuracy'][i]*100))
        # else:
        #     if rules_acc['accuracy'][i]=='No test':
        #         rule_string=rule_string + '"Untable (No_test)"'
        #         last_box.append('"Untable (No_test)"')
        #     else:
        #         rule_string=rule_string + '"Unstable ({0:.1f}%)"'.format(rules_acc['accuracy'][i]*100)
        #         last_box.append('"Unstable ({0:.1f}%)"'.format(rules_acc['accuracy'][i]*100))

        rules_strings.append(rule_string)
    return rules_strings#, last_box

    
# import graphviz
# from graphviz import Source
# for r in range(0,len(rules_strings)):
#     #g = graphviz.Digraph('Rule_{0:.0f}'.format(r+1), filename='Rule_{0:.0f}'.format(r+1))
#     #g.attr(rankdir='LR', size='8,2')
#     g='digraph g{\n rankdir=LR;\n'+rules_strings[r]+'\n'+last_box[r]+' [shape = rectangle];\n}'
#     src = Source(g)
#     src.render('Rules/Rule_{0:.0f}.gv'.format(r+1), view=True ) 

#%%

# def rules_strings_for_Rsunburst(dic_rules,l):
#     rules_strings=[]
#     last_box=[]
#     for i in range(0,len(dic_rules)):
#         rule=pd.DataFrame(dic_rules[i])
#         _,idx=np.unique(rule[['feature']], return_index=True)
#         unique_feat=np.array(rule.iloc[np.sort(idx)][['feature']]).ravel()
#         rule_string=str('sequences_1$V1[{id}]<-"'.format(id=l+1))

#         for j in unique_feat:#range(0,len(unique_feat))
#             rule_feat=rule.query('feature == @j')
#             for k in ['>','<=']:
#                 rule_feat_ineq=rule_feat.query('inequality == @k')
#                 if len(rule_feat_ineq)>0:
#                     if k == '>':
#                         rule_feat_ineq=rule_feat_ineq.sort_values(by='threshold',ascending=False)
#                         rule_string=rule_string + j + k + str(rule_feat_ineq[['threshold']].iloc[0][0].round(3))+'-'
#                     else:
#                         rule_feat_ineq=rule_feat_ineq.sort_values(by='threshold',ascending=True)
#                         rule_string=rule_string + j + k + str(rule_feat_ineq[['threshold']].iloc[0][0].round(3))+'-'
#         if df_leaves['val'][i]==1:
#             if rules_acc['accuracy'][i]=='No test':
#                 rule_string=rule_string + 'Stable (No_test)"'
#                 last_box.append('Stable (No_test)"')
#             else:
#                 rule_string=rule_string + 'Stable ({0:.1f}%)"'.format(rules_acc['accuracy'][i]*100)
#                 last_box.append('Stable ({0:.1f}%)"'.format(rules_acc['accuracy'][i]*100))
#         else:
#             if rules_acc['accuracy'][i]=='No test':
#                 rule_string=rule_string + 'Untable (No_test)"'
#                 last_box.append('Untable (No_test)"')
#             else:
#                 rule_string=rule_string + 'Unstable ({0:.1f}%)"'.format(rules_acc['accuracy'][i]*100)
#                 last_box.append('Unstable ({0:.1f}%)"'.format(rules_acc['accuracy'][i]*100))

#         rules_strings.append(rule_string)
#     return rules_strings, last_box

# rules_strings, last_box=rules_strings_for_Rsunburst(dic_rules,l)
