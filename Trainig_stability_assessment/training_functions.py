from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from metrics_n import *


def PFI_fun(estimator, X_train, Y_train, X_test, Y_test, scorer):
    '''
    REQUIRES: estimator, X_train, Y_train, X_test, Y_test, scorer
    '''

    considered_features = X_train.columns.to_list()
    
    estimator.fit(X_train[considered_features], Y_train)
    
    r = permutation_importance(estimator, X_test[considered_features], Y_test, 
                               n_repeats=30, random_state=0, scoring=scorer)
    PFI_features=[]
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            PFI_features.append(X_train.columns[i])#estimator.feature_names_in_[i])
            # print(f"{estimator.feature_names_in_[i]:<8}"
            #     f"\t {r.importances_mean[i]:.3f}"
            #     f" ({r.importances_std[i]:.3f})")
    
    
    return PFI_features

def GSkFCV(param_grid, X_train, Y_train, estimator, scorer):
    '''
    REQUIRES: param_grid, X_train, Y_train, PFI_features, estimator, scorer
    '''
    
    n_folds = 6
    seed = 23
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=kfold, scoring=scorer, verbose=1)
    grid_search.fit(X_train, Y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    params = grid_search.cv_results_['params']
    for mean, stdev, param in sorted(zip(means, stds, params), key=lambda x: x[0], reverse=True)[:5]:
        print("%f (%f) with: %r" % (mean, stdev, param))

    return best_model, best_params, means, stds, params


