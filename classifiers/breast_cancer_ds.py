import time
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, svm, gaussian_process, ensemble, tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.fml import FMLClient as fml

# Start Time of the execution of the program
start_time = time.time()

# init FMLearn
f = fml()

# import some data to play with
db = datasets.load_breast_cancer()
X = db.data
Y = db.target

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=123)

models = []
models.append(('LR', linear_model.LogisticRegression(C=1e5, solver='lbfgs', max_iter=10000, multi_class='auto')))
models.append(('SVC', svm.SVC(gamma='scale', max_iter=10000)))
models.append(('LSVC',svm.LinearSVC(max_iter=10000)))
models.append(('ABC', ensemble.AdaBoostClassifier()))
models.append(('BC', ensemble.BaggingClassifier()))
models.append(('GBC', ensemble.GradientBoostingClassifier()))
models.append(('RFC', ensemble.RandomForestClassifier(n_estimators=10)))
models.append(('GPC', gaussian_process.GaussianProcessClassifier()))

names = []
scores = []

# finding the best model

for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
    names.append(name)
    
    # f.publish(model, "Accuracy", score, str(db.data))

tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)

# k-fold validation

strat_k_fold = StratifiedKFold(n_splits=5, random_state=10)

names = []
scores = []
for name, model in models:
    score = cross_val_score(model, X, Y, cv=strat_k_fold, scoring='accuracy').mean()
    names.append(name)
    scores.append(score)
    
    # f.publish(model, "Accuracy", score, str(db.data))

kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)

print("--- %s seconds --- for %s" % ((time.time() - start_time), "Cross-Validation"))

'''
# running GridSearchCV for LogisticRegression

c_values = list(np.arange(1, 10))
param_grid = [
    {'C': c_values, 'penalty': ['l1'], 'solver' : ['liblinear'], 'multi_class' : ['ovr']},
    {'C': c_values, 'penalty': ['l2'], 'solver' : ['liblinear', 'newton-cg', 'lbfgs'], 'multi_class' : ['ovr']},
    {'C': c_values, 'penalty': ['l2'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'multi_class' : ['ovr']}
]

grid = GridSearchCV(linear_model.LogisticRegression(max_iter=10000), param_grid, cv=strat_k_fold, scoring='accuracy', iid=False)
grid.fit(X, Y)

print(grid.best_params_)
print(grid.best_estimator_)

print("--- %s seconds --- for %s" % ((time.time() - start_time), "Grid Search for Logistic Regression"))

''''''
# LogisticRegression on the best params
logreg_new = linear_model.LogisticRegression(C=2, multi_class='ovr', penalty='l2', solver='newton-cg', max_iter=10000)

initial_score = cross_val_score(logreg_new, X, Y, cv=strat_k_fold, scoring='accuracy').mean()
print("Final accu   racy : {} ".format(initial_score))

# send to FMLearn
f._jprint(f.publish(logreg_new, "Accuracy", initial_score, str(db.data), grid.best_params_))

print("--- %s seconds --- for %s" % ((time.time() - start_time), "Best Params for LR"))
''''''
# running GridSearchCV for AdaBoostClassifier

learning_rate = list(np.arange(start=0.05, stop=1, step=0.05))
n_estimators = list(np.arange(start=30, stop=100, step=5))
param_grid = [
    {'learning_rate': learning_rate, 'algorithm': ['SAMME.R'], 'n_estimators' : n_estimators},
    {'learning_rate': learning_rate, 'algorithm': ['SAMME'], 'n_estimators' : n_estimators},
    {'learning_rate': learning_rate, 'algorithm': ['SAMME.R'], 'n_estimators': n_estimators}
]

grid = GridSearchCV(ensemble.AdaBoostClassifier(), param_grid, cv=strat_k_fold, scoring='accuracy', iid=False)
grid.fit(X, Y)

print(grid.best_params_)
print(grid.best_estimator_)

print("--- %s seconds --- for %s" % ((time.time() - start_time), "GridSearchCV for ABC"))

''''''
# AdaBoostClassifier on the best params
abc_new = ensemble.AdaBoostClassifier(algorithm='SAMME', base_estimator=None,
                   learning_rate=0.75, n_estimators=90,
                   random_state=None)
initial_score = cross_val_score(abc_new, X, Y, cv=strat_k_fold, scoring='accuracy').mean()
print("Final accu   racy : {} ".format(initial_score))

print("--- %s seconds --- for %s" % ((time.time() - start_time), "Best Params for ABC"))
'''
'''
c_values = list(np.arange(1, 10))
param_grid = [
    {'C': c_values, 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma' : ['scale', 'auto'], 'max_iter' : [10000]}
]

grid = GridSearchCV(svm.SVC(), param_grid, cv=strat_k_fold, scoring='accuracy', iid=False)
grid.fit(X, Y)

print(grid.best_params_)
print(grid.best_estimator_)

print("--- %s seconds --- for %s" % ((time.time() - start_time), "GridSearchCV for SVC"))
'''

'''
c_values = list(np.arange(1, 10))
param_grid = [
    {'C': c_values, 'penalty': ['l1', 'l2'], 'loss' : ['hinge', 'squared_hinge'], 'max_iter' : [10000]}
]

grid = GridSearchCV(svm.LinearSVC(), param_grid, cv=strat_k_fold, scoring='accuracy', iid=False)
grid.fit(X, Y)

print(grid.best_params_)
print(grid.best_estimator_)

print("--- %s seconds --- for %s" % ((time.time() - start_time), "GridSearchCV for LinearSVC"))
'''

'''
# running GridSearchCV for BaggingClassifier

n_estimators = list(np.arange(start=30, stop=100, step=5))
param_grid = [
    {'n_estimators' : n_estimators}
]

grid = GridSearchCV(ensemble.BaggingClassifier(), param_grid, cv=strat_k_fold, scoring='accuracy', iid=False)
grid.fit(X, Y)

print(grid.best_params_)
print(grid.best_estimator_)

print("--- %s seconds --- for %s" % ((time.time() - start_time), "GridSearchCV for BaggingClassifier"))
'''


# running GridSearchCV for GradientBoostingClassifier

learning_rate = list(np.arange(start=0.05, stop=1, step=0.05))
n_estimators = list(np.arange(start=30, stop=100, step=5))
max_depth = list(np.arange(start=3, stop=10, step=1))
param_grid = [
    {'n_estimators' : n_estimators, 'loss' : ['deviance', 'exponential'], 'learning_rate': learning_rate, 'max_depth': max_depth}
]

grid = GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid, cv=strat_k_fold, scoring='accuracy', iid=False)
grid.fit(X, Y)

print(grid.best_params_)
print(grid.best_estimator_)

print("--- %s seconds --- for %s" % ((time.time() - start_time), "GridSearchCV for GradientBoostingClassifier"))




# Send to FMLearn
# f._jprint(f.publish(abc_new, "Accuracy", initial_score, str(db.data), grid.best_params_))

# f.publish(model, "Accuracy", acc, str(db.data))

# f._jprint(f.retrieve_all_metrics(str(db.data)))

# f._jprint(f.retrieve_best_metric(str(db.data), False))