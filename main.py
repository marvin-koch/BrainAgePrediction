
# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import Lasso
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectFromModel
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression, chi2, r_regression,f_classif, SelectPercentile
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
import lightgbm as lgb

X_train = pd.read_csv('X_train.csv').drop(columns = ['id'])
y_train = pd.read_csv('y_train.csv')['y']
X_test_full = pd.read_csv('X_test.csv')
X_test = X_test_full.drop(columns = ['id'])

threshold_rows = 200  # 70For example, allowing up to 3 missing values per row

# Find rows with missing values in feature matrix
missing_rows_X = X_train[X_train.isnull().sum(axis=1) > threshold_rows].index

# Find the intersection of rows with missing values in both X and y
rows_to_remove = set(missing_rows_X)

# Remove rows with too many missing values from X and y
X_train = X_train.drop(index=rows_to_remove)
y_train = y_train.drop(index=rows_to_remove)

threshold_columns = 175  # 175For example, allowing up to 3 missing values per column

# Find rows with missing values in feature matrix
columns_to_remove = X_train.columns[X_train.isnull().sum() > threshold_columns]
X_train_cleaned = X_train.drop(columns=columns_to_remove)
X_test = X_test.drop(columns=columns_to_remove)

# Step 1: Imputation of Missing Values
# imputer = KNNImputer(n_neighbors=5)
# imputer = IterativeImputer()
# imputer = SimpleImputer()
imputer = IterativeImputer(max_iter=10, random_state=1, n_nearest_features = 15, verbose = 0)#, estimator=RandomForestRegressor(n_estimators=15))


X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test) #TODO

# X_train, y_train, X_test = fill_missing_values(X_train, y_train, X_test)

# xtrain = pd.DataFrame(X_train)
# xtest = pd.DataFrame(X_test)
 
# # save the dataframe as a csv file
# xtrain.to_csv("xtrain.csv")
# xtest.to_csv("xtest.csv")


# Step 2: Outlier Detection
from scipy import stats


# outlier_indices = np.where(z_scores > 4)

# clf = IsolationForest(n_estimators=100, warm_start=True)
# clf.fit(X_train)  # fit 10 trees  
# clf.set_params(n_estimators=20)  # add 10 more trees  
# clf.fit(X_train)  # fit the added trees  
# clf.fit(X_test)
# clf = IsolationForest(random_state=1)  # Adjust contamination according to your dataset

# # Fit the model and predict outliers
# outliers = clf.fit_predict(X_train)

# # Identify and remove outliers from the dataset
# X_train = X_train[outliers != -1]
# y_train = y_train[outliers != -1]

z_scores = np.abs(stats.zscore(X_train_imp))
mask = ~(np.any(z_scores > 6, axis=1))
X_train = pd.DataFrame(X_train[mask])
y_train = pd.DataFrame(y_train[mask])

# clf = LocalOutlierFactor() # modify here
# mask = pd.Series(clf.fit_predict(X_train_imp))
# X_train = pd.DataFrame(X_train[mask])
# y_train = pd.DataFrame(y_train[mask])



    


# Step 4: Standardization (Using StandardScaler)
# scaler = RobustScaler(quantile_range=(0.1,99.9))
# # Step 3: Feature Selection (Using Gradient Boosting Feature Importance)
# model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.1, max_depth=4, min_samples_split=5, loss="squared_error", max_features='sqrt', random_state=1)
# model.fit(X_train_imp, y_train)

X_test = pd.DataFrame(X_test)

# X_test = pd.DataFrame(X_test)
X_train_corr_ = X_train.corr()

X_train_too_correlated = (X_train_corr_.mask(
    np.tril(np.ones([len(X_train_corr_)]*2, dtype=bool))).abs() > 0.98).any()

X_train = X_train.loc[:, (~X_train_too_correlated)]
X_test = X_test.loc[:, (~X_train_too_correlated)]

X_train_mask = X_train.isna()
X_test_mask = (X_test).isna()
 
imputer = IterativeImputer(max_iter=10, random_state=1, n_nearest_features = 15, verbose = 0) #estimator=RandomForestRegressor( n_estimators=4,
# imputer = KNNImputer(n_neighbors=6, weights='uniform')
imputer.fit(X_train)
X_train = (imputer.transform(X_train))
X_test = (imputer.transform(X_test))


# Select the top k features based on importance
k = 150
# feature_importances = model.feature_importances_
# top_feature_indices = feature_importances.argsort()[-k:][::-1]
# X_train_selected_ = X_train[:, top_feature_indices]
# X_test_selected = X_test[:, top_feature_indices]

# selector = SelectPercentile()
# X_train_selected = selector.fit_transform((X_train), y_train)
# X_test_selected = selector.transform((X_test))

selector = SelectKBest(f_regression, k=k)
X_train_selected = selector.fit_transform((X_train), np.array(y_train).ravel())
X_test_selected = selector.transform((X_test))
X_train_mask = selector.transform(X_train_mask)
X_test_mask = selector.transform(X_test_mask)

clf_selec = Lasso(alpha=0.02, max_iter=5000)
model_select = SelectFromModel(clf_selec.fit(X_train_selected, np.array(y_train).ravel()), prefit=True)
X_train_selected = model_select.transform(X_train_selected)
X_test_selected = model_select.transform(X_test_selected)
X_train_mask = model_select.transform(X_train_mask)
X_test_mask = model_select.transform(X_test_mask)

X_train = (pd.DataFrame(X_train_selected)).mask(np.array(X_train_mask))
X_test = (pd.DataFrame(X_test_selected)).mask(np.array(X_test_mask))

imputer = IterativeImputer(max_iter=10, random_state=1, n_nearest_features = 15, verbose = 0)
imputer.fit(X_train)
X_train_selected = (imputer.transform(X_train))
X_test_selected = (imputer.transform(X_test))




# scaler = RobustScaler()
# X_train_selected = scaler.fit_transform(X_train_selected)
# X_test_selected = scaler.transform(X_test_selected)

# svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=63, epsilon=0.1))
svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=500, epsilon=0.05))
lasso = make_pipeline(StandardScaler(), Lasso(alpha =0.0005, random_state=1))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
params = {'loss': 'squared_error', 'n_estimators': 250, 'learning_rate': 0.025, 'subsample': 0.75, 'max_depth': 6, 'min_samples_split': 5, 'n_iter_no_change': 100, 'validation_fraction': 0.1, 'random_state': 0, 'verbose': 1}

# GBoost = GradientBoostingRegressor(**params)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                            max_depth=4, max_features='sqrt',
                                            min_samples_leaf=15, min_samples_split=10,
                                            loss='huber', random_state =5)
other_params = {'learning_rate': 0.1, 'n_estimators': 300, 'max_depth': 8, 'min_child_weight': 2, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 3, 'reg_lambda': 2}
xg_reg_selected = xgb.XGBRegressor(**other_params)
# xg_reg_selected.fit(X_train_selected, y_train)
# y_pred_xg = xg_reg_selected.predict(X_test_selected)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

# model_lgb.fit(X_train_selected, y_train)
# y_pred_lgb = model_lgb.predict(X_test_selected)

best_model = StackingRegressor(estimators=[('SVR', svr), ('XG', xg_reg_selected), ('GBoost', GBoost), ('lgb', model_lgb)],final_estimator=lasso)
best_model.fit(X_train_selected, np.array(y_train).ravel())


# best_model = svr
# best_model.fit(X_train_selected, y_train)
cv_scores = cross_val_score(best_model, X_train_selected, np.array(y_train).ravel(), cv=6, scoring='r2')
mean_r2 = np.mean(cv_scores)
std_r2 = np.std(cv_scores)


print(f'Mean R-squared (R^2) Score: {mean_r2}')
print(f'Standard Deviation of R-squared (R^2) Score: {std_r2}')
print("Scores:", cv_scores)
best_model.fit(X_train_selected, y_train)
# Step 7: Make predictions on the test set

y_test_pred = best_model.predict(X_test_selected)

# y_test_pred = (0.7*y_test_pred + 0.15*y_pred_xg + 0.15*y_pred_lgb)

# Step 8: Prepare the submission file
submission = pd.DataFrame({'id': X_test_full['id'], 'y': y_test_pred})
submission.to_csv('submission.csv', index=False)