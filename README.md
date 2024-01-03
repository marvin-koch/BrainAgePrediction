# Brain Age Prediction using MRI Data - ETH Advanced Machine Learning 2023 Task 1
In this task, we first preprocessed our data. We began with removing all rows and columns which had too many missing values above a threshold. The threshold was a hyperparmater that we had to tune. In our case we removed all columns with more than 175 missing values and all rows with more than 200 missing values.
We then performed the first temporary imputation of missing values with sklearn’s iterative imputer, who's parameters we had to tune too. 
We then performed outlier detection using the z_score and removed all data points which were 6 standard deviations or above from the mean. 
Afterwards, we removed all columns which where too closely correlated with the help of pandas’ corr() function. Afterwards, we recalculated the missing values using iterative imputer again. 
Next, we used sklearn’s SelectKBest, with k=150, followed by Lasso with alpha=0.02, to choose the best columns to keep. Finally we recomputed the missing values again.

The model that we used was a stacking regressor, using SVR, XGBoost, GradientBoostingRegressor and LGBMRegressor as estimators. The final estimator was a Lasso. We tuned the hyperparameters of each model individually using sklearn’s grid search. We also tried using other estimators and models such as kernel ridge regression, linear regression and Gaussian Processes. We evaluated our model using cross-validation.
