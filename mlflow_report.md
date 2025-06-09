
## 📊 MLflow Report: `laptop-pricing-model`

**Run ID**: `211650eabbba4ef2a9cf2c7aed6fc36b`  
**Model Version**: `5`

### 🔢 Metrics
- `R2`: 0.6421370506286621
- `MSE`: 3026852.0
- `training_mean_squared_error`: 753320.3125
- `training_mean_absolute_error`: 539.8137817382812
- `training_r2_score`: 0.9220167994499207
- `training_root_mean_squared_error`: 867.9402701223166
- `training_score`: -0.10151685029268265
- `best_cv_score`: -0.15979987382888794
- `MAPE`: 0.1563664972782135
- `MAE`: 958.0400390625

### ⚙️ Parameters
- `best_learning_rate`: 0.1
- `best_max_depth`: 5
- `best_n_estimators`: 100
- `cv`: 5
- `error_score`: nan
- `estimator`: XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             feature_weights=None, gamma=None, grow_policy=None,
             importance_type=None, interaction_constraints=None,
             learning_rate=None, max_bin=None, max_cat_threshold=None,
             max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
             max_leaves=None, min_child_weight=None, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=None,
             n_jobs=None, num_parallel_tree=None, ...)
- `n_jobs`: None
- `param_grid`: {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7, 9], 'learning_rate': [0.01, 0.1, 0.2, 0.3]}
- `pre_dispatch`: 2*n_jobs
- `refit`: True
- `return_train_score`: False
- `scoring`: make_scorer(mean_absolute_percentage_error, greater_is_better=False, response_method='predict')
- `verbose`: 0
