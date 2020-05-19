import os
import time
import pandas as pd
import numpy as np
from sklearn import linear_model, svm, gaussian_process, ensemble, tree
from sklearn import preprocessing as pp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.fml import FMLClient as fml

# Start Time of the execution of the program
start_time = time.time()

# init FMLearn
f = fml(debug=True)

# import some data to play with
path = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(path + "/data/covtype.data", sep=',')
soil_type = []
for i in range(0, 40):
    soil_type.append('Soil_Type_' + str(i))
columns = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_1', 'Wilderness_Area_2',
'Wilderness_Area_3', 'Wilderness_Area_4']
columns += soil_type + ['Cover_Type']
df.columns = columns
df.dropna(inplace=True)

print(df.head())
print(df.shape)

X = df[df.columns.difference(['Cover_Type'])]
y = df['Cover_Type']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=123)

f.set_dataset(x_train, y_train, x_train, y_train)

models = []
#models.append(('LR', linear_model.LogisticRegression(C=1e5, solver='lbfgs', max_iter=50000, multi_class='auto')))
#models.append(('SVC', svm.SVC(gamma='scale', max_iter=50000)))
#models.append(('LSVC',svm.LinearSVC(max_iter=50000)))
models.append(('ABC', ensemble.AdaBoostClassifier()))
models.append(('BC', ensemble.BaggingClassifier()))
#models.append(('GBC', ensemble.GradientBoostingClassifier()))
models.append(('RFC', ensemble.RandomForestClassifier(n_estimators=10)))
#models.append(('GPC', gaussian_process.GaussianProcessClassifier()))

names = []
scores = []

# finding the best model

for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
    names.append(name)
    #f.publish(model, "Accuracy", score)

tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)

# k-fold validation
names = []
scores = []
strat_k_fold = StratifiedKFold(n_splits=5, random_state=10)

for name, model in models:
    score = cross_val_score(model, X, y, cv=strat_k_fold, scoring='accuracy').mean()
    names.append(name)
    scores.append(score)

kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)

print("--- %s seconds --- for %s" % ((time.time() - start_time), "K fold validation on all algos"))

# GridSearchCV for BaggingClassifier

n_estimators = [10]#list(np.arange(start=10, stop=70, step=5))
param_grid = [
    {'n_estimators' : n_estimators}
]

grid = GridSearchCV(ensemble.BaggingClassifier(), param_grid, cv=strat_k_fold, scoring='accuracy', iid=False)
grid.fit(X, y)

print(grid.best_params_)
print(grid.best_estimator_)

print("--- %s seconds --- for %s" % ((time.time() - start_time), "GridSearchCV for BaggingClassifier"))

# GridSearcgCV for RFC

n_estimators = list(np.arange(start=10, stop=70, step=5))
max_depth = list(np.arange(start=3, stop=10, step=1))
param_grid = [
    {'n_estimators' : n_estimators, 'max_depth': max_depth}
]

grid = GridSearchCV(ensemble.RandomForestClassifier(), param_grid, cv=strat_k_fold, scoring='accuracy', iid=False)
grid.fit(X, y)

print(grid.best_params_)
print(grid.best_estimator_)

print("--- %s seconds --- for %s" % ((time.time() - start_time), "GridSearchCV for GradientBoostingClassifier"))
