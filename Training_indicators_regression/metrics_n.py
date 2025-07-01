from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from numpy import array
from sklearn.metrics import confusion_matrix
from numpy import average, std
import warnings
from sklearn.metrics import make_scorer

'''
==== LIST OF FUNCTIONS ====

metric_n(model, x, y, n, metric)

prec_unstable(y_true,y_pred)

get_metrics(model, x, y, metrics, n=1)

print_metrics(results, names=None)

'''



'''
Returns a list of the results obtained from applying a metric to a fitted model predictions using n splits

Parameters:
    - model (object): fitted model, assumed to implement the scikit-learn estimator interface
        Example: model=DecisionTreeClassifier().fit(X_train, Y_train)
    
    - x (array-like or pd.DataFrame()): features test set
    - y (array-like or pd.Series()): target test set
    
    - n (int): number of data sub-splits
    
    - metric (function): function to use as metric
        Example: metric = accuracy_score
        
Returns: 
    - results (list): list of results (float) of the applied metrics
        Example: [0.98,0.87,0.95]
'''
def metric_n(model, x, y, n, metric, scorer=None):
    results = []
    split_len = int(len(x)/n)
    
    for i in range(n):
        x_i = x.iloc[i*split_len:(i+1)*split_len]
        y_i = y.iloc[i*split_len:(i+1)*split_len]
        
        y_pred_i = model.predict(x_i)
        if metric == precision_score:
            results.append(metric(y_i, y_pred_i, zero_division=1))
        elif metric == scorer:
            results.append(metric(model,x_i,y_i))
        else:
            results.append(metric(y_i, y_pred_i))
    
    return results


'''
Returns the precision with inverted outputs, i.e.: TN/(TN+FN)

Parameters:
    - y_true (array-like): real output values
    - y_pred (array-like): predicted output values
   
Returns:
    - prec_unst (float): precision score
'''
def prec_unstable(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec_unst=tn/(tn+fn)
    return prec_unst


'''
Returns the results of applying metrics to the predictions of a scikit-learn model

Parameters:
    - model (object): fitted model, assumed to implement the scikit-learn estimator interface
        Example: model=DecisionTreeClassifier().fit(X_train, Y_train)
    
    - x (array-like or pd.DataFrame()): features test set
    - y (array-like or pd.Series()): target test set
    
    - metrics (list): list of functions to use as metrics
        Example: metrics = [accuracy_score, precision_score, recall_score, f1_score]
    
    - n (int): number of data sub-splits
    
Returns: 
    - results (list): list of lists of results (float) of the applied metrics
        Example: [[0.98,0.87,0.95],[0.87,0.89,0,85]]
'''
def get_metrics(model, x, y, metrics, scorer=None, n=1):
    results = []
    for m in metrics:
        results.append(metric_n(model, x, y, n, m, scorer))
        
    return results


'''
Prints a clean and visual results summary

Parameters:
    - results (list): list of results
        Example: results=get_metrics_n(model,x,y,metrics,n)
    
    - names (list or None): list of names of the metrics (tied to the results)
        Example: names=['Accuracy', 'Precision']
        
Returns:
    - void
'''
def print_metrics(results, CrossVal_res=None, model_name=None, names=None):
    
    if names == None:
        names = []
        
    for i in range(len(results)):
        if names==None:
            names.append('Metric '+str(i))
        print('\t' + names[i] + ': %.3f (%.3f)'
                % (average(results[i]), std(results[i])) )
        if all(CrossVal_res!=None):
            CrossVal_res.loc[names[i],model_name]=str(average(results[i]))+'\pm'+str(std(results[i]))

def print_metrics_on_file(results, f, names=None):
    
    if names == None:
        names = []
        
    for i in range(len(results)):
        if names==None:
            names.append('Metric '+str(i))
        f.write('\t' + names[i] + ': %.3f (%.3f) \n'
                % (average(results[i]), std(results[i])) )












'''
==== DEPRECATED ====
'''

def acc_prec_rec_f1(y, y_pred):
    warnings.warn('DeprecationWarning: acc_prec_rec_f1(y, y_pred) is deprecated in this version and will raise an exception in following versions. Use get_metrics() and get_metrics_n() instead.')
    a=accuracy_score(y, y_pred)
    p=precision_score(y, y_pred, zero_division=1)
    pu=prec_unstable(y, y_pred)
    r=recall_score(y, y_pred)
    f=f1_score(y, y_pred)
    
    return a, p, pu, r, f

def acc_prec_rec_f1_n(model, x, y, n):
    return array(metric_n(model, x, y, n, acc_prec_rec_f1))

def get_metrics_n(model, x, y, metrics, n):
    warnings.warn('DeprecationWarning: get_metrics_n() is deprecated in this version and will raise an exception in following versions. Use get_metrics(n=n_splits) instead.')
    results = []
    for m in metrics:
        results.append(metric_n(model, x, y, n, m))
        
    return results

def accuracy_n(model, x, y, n): # Return list of accuracies by spliting x and y into n splits
# Example usage: accuracy_n(dtc_model, X_eval, Y_eval, 6) --> [0.776, 0.715, 0.782, 0.794, 0.753, 0.741]  
    warnings.warn('DeprecationWarning: accuracy_n() is deprecated in this version and will raise an exception in following versions. Use get_metrics(n=n_splits) instead.')
    return metric_n(model, x, y, n, accuracy_score)

def precision_n(model, x, y, n):
    warnings.warn('DeprecationWarning: precision_n() is deprecated in this version and will raise an exception in following versions. Use get_metrics(n=n_splits) instead.')
    return metric_n(model, x, y, n, precision_score)

def recall_n(model, x, y, n):
    warnings.warn('DeprecationWarning: recall_n() is deprecated in this version and will raise an exception in following versions. Use get_metrics(n=n_splits) instead.')
    return metric_n(model, x, y, n, recall_score)

def f1_n(model, x, y, n):
    warnings.warn('DeprecationWarning: f1_n() is deprecated in this version and will raise an exception in following versions. Use get_metrics(n=n_splits) instead.')
    return metric_n(model, x, y, n, f1_score)

def prec_unstable_n(model, x, y, n): 
    warnings.warn('DeprecationWarning: prec_unstable_n() is deprecated in this version and will raise an exception in following versions. Use get_metrics(n=n_splits) instead.')
    return metric_n(model, x, y, n, prec_unstable)

def set_scorer(df):
    if df[['Stable']].mean()[0]<0.5:
        scorer=make_scorer(fbeta_score, beta=2)
    else:
        scorer=make_scorer(fbeta_score, beta=0.5)
 
    return scorer