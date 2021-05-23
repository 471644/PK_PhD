pip install bayesian-optimization

from bayes_opt import BayesianOptimization
def bayesion_opt_lgbm(X, y, init_iter=3, n_iters=7, random_state=11, seed=101):
    dtrain = lgb.Dataset(data=X, label=y)
    def f1_metric(preds, dtrain):
        labels = dtrain.get_label()
        return 'f1', f1_score(labels, preds), True
# Objective Function
    def hyp_lgbm(num_leaves, feature_fraction, bagging_fraction, max_depth,
                 learning_rate, num_iterations):
        params = {
            'application': 'binary',
            'early_stopping_round': 100
        }  # Default parameters
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['num_iterations'] = int(round(num_iterations))
        params['learning_rate'] = learning_rate
        cv_results = lgb.cv(params,
                            dtrain,
                            nfold=10,
                            seed=seed,
                            categorical_feature=[],
                            stratified=False,
                            verbose_eval=None,
                            metrics=['auc'])
#         print(cv_results)
        return np.max(cv_results['auc-mean'])

# Domain space-- Range of hyperparameters
    pds = {
        'num_leaves': (10, 80),
        'feature_fraction': (0.1, 1),
        'bagging_fraction': (0.1, 1),
        'max_depth': (5, 25),
        'learning_rate': (0.01, 0.1),
        'num_iterations': (50, 400)
    }
    # Surrogate model
    optimizer = BayesianOptimization(hyp_lgbm, pds, random_state=random_state)
    # Optimize
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)
    # Print Best Parameters
    print('Best Parameters Are :',optimizer.max['params'])
    

bayesion_opt_lgbm(np.array(train[Features]), train.label.values.flatten(), init_iter=10, n_iters=15, random_state=77, seed = 101)