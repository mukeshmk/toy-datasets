import time
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, svm, gaussian_process, ensemble
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV
# from sklearn.fml import FMLClient as fml

# Start Time of the execution of the program
start_time = time.time()

# import some data to play with
ds = datasets.load_diabetes()
X = ds.data
Y = ds.target

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=123)

models = []
models.append(('LR', linear_model.LinearRegression()))
models.append(('HR', linear_model.HuberRegressor()))
models.append(('LS', linear_model.Lasso()))
models.append(('RG', linear_model.Ridge()))
models.append(('BR', linear_model.BayesianRidge()))
models.append(('LSVR', svm.LinearSVR()))
models.append(('GPR', gaussian_process.GaussianProcessRegressor()))
models.append(('ABC', ensemble.AdaBoostRegressor()))
models.append(('BR', ensemble.BaggingRegressor()))
models.append(('GBR', ensemble.GradientBoostingRegressor()))
models.append(('RFR', ensemble.RandomForestRegressor()))

# finding the best model
names = []
scores = []
for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    scores.append(r2_score(y_test, y_pred))
    names.append(name)

tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)

# k-fold validation
names = []
scores = []
strat_k_fold = StratifiedKFold(n_splits=5, random_state=10)

for name, model in models:
    result_set = cross_validate(model, X, Y, cv=strat_k_fold, scoring=('r2'))
    names.append(name)
    scores.append(result_set.get('test_score').max())

kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)

# GridSearchCV for LinearRegression
param_grid = [
     {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
]

grid = GridSearchCV(linear_model.LinearRegression(), param_grid, cv=strat_k_fold, scoring=('r2'))
grid.fit(X, Y)

print(grid.best_params_)
print(grid.best_estimator_)

# LinearRegression on the best params
linreg_new = linear_model.LinearRegression()

initial_score = cross_validate(linreg_new, X, Y, cv=strat_k_fold, scoring=('r2')).get('test_score').mean()
print("Final accu   racy : {} ".format(initial_score))

print("--- %s seconds --- for %s" % ((time.time() - start_time), model.__class__))

# GridSearchCV for BaggingRegressor
estimators = np.arange(start=5, stop=25, step=1)
samples = np.arange(start=0.1, stop=1.0, step=0.1)
features = np.arange(start=0.1, stop=1.0, step=0.1)
param_grid = [
     {'n_estimators': estimators, 'max_samples': samples, 'max_features': features}
]

grid = GridSearchCV(ensemble.BaggingRegressor(), param_grid, cv=strat_k_fold, scoring=('r2'))
grid.fit(X, Y)

print(grid.best_params_)
print(grid.best_estimator_)

# BaggingRegressor on the best params
bagreg_new = ensemble.BaggingRegressor()

initial_score = cross_validate(bagreg_new, X, Y, cv=strat_k_fold, scoring=('r2')).get('test_score').max()
print("Final accu   racy : {} ".format(initial_score))

print("--- %s seconds --- for %s" % ((time.time() - start_time), model.__class__))


# GridSearchCV for GradientBoostingRegressor=1)
lr = np.arange(start=0.1, stop=1.0, step=0.1)
estimators = np.arange(start=70, stop=130, step=10)
max_depth = np.arange(start=1, stop=4, step=1)
param_grid = [
     {'loss' : ['ls', 'lad', 'huber', 'quantile'], 'learning_rate': lr, 'n_estimators': estimators, 'max_depth': max_depth}
]

grid = GridSearchCV(ensemble.GradientBoostingRegressor(), param_grid, cv=strat_k_fold, scoring=('r2'))
grid.fit(X, Y)

print(grid.best_params_)
print(grid.best_estimator_)

print("--- %s seconds --- for %s" % ((time.time() - start_time), model.__class__))

# GradientBoostingRegressor on the best params
gbrreg_new = ensemble.GradientBoostingRegressor()

initial_score = cross_validate(gbrreg_new, X, Y, cv=strat_k_fold, scoring=('r2')).get('test_score').max()
print("Final accu   racy : {} ".format(initial_score))

print("--- %s seconds --- for %s" % ((time.time() - start_time), model.__class__))


# f = fml()
# f._jprint(f.publish(model, "Accuracy", acc, str(iris.data)))

# f.publish(model, "Accuracy", acc, str(iris.data))

# f._jprint(f.retrieve_all_metrics(str(iris.data)))

# f._jprint(f.retrieve_best_metric(str(iris.data), False))