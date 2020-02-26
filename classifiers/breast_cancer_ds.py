import time
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, svm, gaussian_process, ensemble, tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
# from sklearn.fml import FMLClient as fml

# Start Time of the execution of the program
start_time = time.time()

# import some data to play with
iris = datasets.load_breast_cancer()
X = iris.data
Y = iris.target

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
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)

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

kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)

print("--- %s seconds --- for %s" % ((time.time() - start_time), model.__class__))

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
print("--- %s seconds --- for %s" % ((time.time() - start_time), model.__class__))

# LogisticRegression on the best params
logreg_new = linear_model.LogisticRegression(C=2, multi_class='ovr', penalty='l2', solver='newton-cg', max_iter=10000)

initial_score = cross_val_score(logreg_new, X, Y, cv=strat_k_fold, scoring='accuracy').mean()
print("Final accu   racy : {} ".format(initial_score))

print("--- %s seconds --- for %s" % ((time.time() - start_time), model.__class__))

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

# AdaBoostClassifier on the best params
abc_new = ensemble.AdaBoostClassifier(algorithm='SAMME', base_estimator=None,
                   learning_rate=0.75, n_estimators=90,
                   random_state=None)
initial_score = cross_val_score(abc_new, X, Y, cv=strat_k_fold, scoring='accuracy').mean()
print("Final accu   racy : {} ".format(initial_score))

print(grid.best_params_)
print(grid.best_estimator_)

print("--- %s seconds --- for %s" % ((time.time() - start_time), model.__class__))

# f = fml()
# f._jprint(f.publish(model, "Accuracy", acc, str(iris.data)))

# f.publish(model, "Accuracy", acc, str(iris.data))

# f._jprint(f.retrieve_all_metrics(str(iris.data)))

# f._jprint(f.retrieve_best_metric(str(iris.data), False))