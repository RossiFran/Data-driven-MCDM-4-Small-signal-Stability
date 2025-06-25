import pandas as pd
import matplotlib.pyplot as plt # plotting
from sklearn.model_selection import KFold, cross_val_score

def compare_models(models_list, X, Y, scoring, n_folds=6, plot=False, save_plot=False, path=None):
  
    '''
    REQUIRED: X_train, Y_train, PFI_features, scorer
    '''
    
    
    '''
    Fits and compares the performance of given sklearn models using cross validation
    
    Parameters:
        - models_list (list): list of tuples (Model_name (str), model). 
            Example: models_list = [('DTC', DecisionTreeClassifier()), ('MLP', MLPClassifier())]
    
        - X (pd.DataFrame): features dataset
        - Y (pd.Series): target dataset
        
        - n_folds (int): number of folds for the cross validation
        
        - scoring (str or callable): scorer (see sklearn.model_selection.cross_val_score docs)
            Example: scoring='accuracy' or scoring=make_scorer(fbeta_score, beta=0.5)
    
    Returns:
        df_model_results (pd.DataFrame): contains for each tested model the mean, std and list of results from the cv
    '''
    
    df_model_results = pd.DataFrame(columns=['Model','Mean','Std','cv_results'])
    
    results = []
    names = []
    
    for name, model in models_list:
        kfold = KFold(n_splits=n_folds, shuffle=True)
        cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        df_model_results.loc[len(df_model_results.index)] = [name, cv_results.mean(), cv_results.std(), cv_results]
    
    display(df_model_results.drop('cv_results',axis=1).sort_values(by='Mean',ascending=False))

    if plot:
        fig=plt.figure(figsize=(8,4))    
        plt.boxplot(df_model_results['cv_results'])
        plt.xticks(ticks=range(1,len(df_model_results)+1), labels=df_model_results['Model'])#,size=15)
        plt.ylabel(r'$F_\beta$')
        fig.tight_layout()
        plt.grid()
        
        if save_plot:
            plt.savefig(path+'models_comparison.pdf', format='pdf', bbox_inches='tight')
            
    return df_model_results

