import os
import time
import pandas as pd
import numpy as np
from sklearn import linear_model, svm, gaussian_process, ensemble, tree
from sklearn import preprocessing as pp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.fml import FMLClient as fml

# One Hot Encoding
def ohe_feature(df, feature, drop_additional_feature=True):
    encoder = pp.OneHotEncoder(categories='auto', sparse=False)
    data = encoder.fit_transform(df[feature].values.reshape(len(df[feature]), 1))
    # creating the encoded df
    ohedf = pd.DataFrame(data, columns=[feature + ': ' + str(i.strip('x0123_')) for i in encoder.get_feature_names()])
    # to drop the extra column of redundant data
    if drop_additional_feature:
        ohedf.drop(ohedf.columns[len(ohedf.columns) - 1], axis=1, inplace=True)
    # concat the ohe df with the original df
    df = pd.concat([df, ohedf], axis=1)
    # to drop the original column in the df
    del df[feature]

    return df, encoder

# Label Encoding
def label_encode_feature(df, feature):
    encoder = pp.LabelEncoder()
    data = encoder.fit_transform(df[feature].values.reshape(len(df[feature]), 1))
    # to drop the original column in the df
    del df[feature]
    # creating the encoded df
    ledf = pd.DataFrame(data, columns=[feature])
    # concat the ohe df with the original df
    df = pd.concat([df, ledf], axis=1)

    return df, encoder

# Start Time of the execution of the program
start_time = time.time()

# init FMLearn
f = fml(debug=True)

# import some data to play with
path = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(path + "/data/magic04.data", sep=',')
df.columns =['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
df.dropna(inplace=True)

print(df.head())
print(df.shape)

df, _ = ohe_feature(df, 'class')

X = df[df.columns.difference(['class: g'])]
y = df['class: g']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=123)

f.set_dataset(x_train, y_train, x_train, y_train)

models = []
models.append(('LR', linear_model.LogisticRegression(C=1e5, solver='lbfgs', max_iter=50000, multi_class='auto')))
models.append(('SVC', svm.SVC(gamma='scale', max_iter=50000)))
#models.append(('LSVC',svm.LinearSVC(max_iter=50000)))
models.append(('ABC', ensemble.AdaBoostClassifier()))
models.append(('BC', ensemble.BaggingClassifier()))
models.append(('GBC', ensemble.GradientBoostingClassifier()))
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

strat_k_fold = StratifiedKFold(n_splits=5, random_state=10)

names = []
scores = []
for name, model in models:
    score = cross_val_score(model, X, y, cv=strat_k_fold, scoring='accuracy').mean()
    names.append(name)
    scores.append(score)
    
    # f.publish(model, "Accuracy", score, str(db.data))

kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)

print("--- %s seconds --- for %s" % ((time.time() - start_time), "Cross-Validation"))

# running GridSearchCV for GradientBoostingClassifier

learning_rate = list(np.arange(start=0.05, stop=1, step=0.05))
n_estimators = list(np.arange(start=30, stop=100, step=5))
max_depth = list(np.arange(start=3, stop=10, step=1))
param_grid = [
    {'n_estimators' : n_estimators, 'loss' : ['deviance', 'exponential'], 'learning_rate': learning_rate, 'max_depth': max_depth}
]

grid = GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid, cv=strat_k_fold, scoring='accuracy', iid=False)
grid.fit(X, y)

print(grid.best_params_)
print(grid.best_estimator_)

print("--- %s seconds --- for %s" % ((time.time() - start_time), "GridSearchCV for GradientBoostingClassifier"))

