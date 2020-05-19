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
df = pd.read_csv(path + "/data/adult.data", sep=', ')
df.columns =['age','workclass','fnlwgt','education','education-num', 'marital-status', 'occupation',
     'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']
df.dropna(inplace=True)

df_test = pd.read_csv(path + "/data/adult.test", sep=', ')
df_test.columns =['age','workclass','fnlwgt','education','education-num', 'marital-status', 'occupation',
     'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']
df_test.dropna(inplace=True)

df = df.append(df_test)
df = df.reset_index()

print(df.head())

df, _ = ohe_feature(df, 'workclass')
df, _ = ohe_feature(df, 'education')
df, _ = ohe_feature(df, 'marital-status')
df, _ = ohe_feature(df, 'occupation')
df, _ = ohe_feature(df, 'relationship')
df, _ = ohe_feature(df, 'race')
df, _ = ohe_feature(df, 'sex')

df, _ = label_encode_feature(df, 'native-country')

df, _ = ohe_feature(df, 'class')

X = df[df.columns.difference(['class: <=50K'])]
y = df['class: <=50K']

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

# GridSearchCV for LoggisticRegression
c_values = list(np.arange(1, 10))
param_grid = [
    {'C': c_values, 'penalty': ['l1'], 'solver' : ['liblinear'], 'multi_class' : ['ovr']},
    {'C': c_values, 'penalty': ['l2'], 'solver' : ['liblinear', 'lbfgs'], 'multi_class' : ['ovr']},
    {'C': c_values, 'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear', 'sag', 'saga'], 'multi_class' : ['ovr']}
]

grid = GridSearchCV(linear_model.LogisticRegression(max_iter=50000), param_grid, cv=strat_k_fold, scoring='accuracy', iid=False)
grid.fit(X, y)

print(grid.best_params_)
print(grid.best_estimator_)

print("--- %s seconds --- for %s" % ((time.time() - start_time), "Grid Search on Logistic Regression"))

# LogisticRegression on the best params
logreg_new = linear_model.LogisticRegression(C=1, multi_class='ovr', penalty='l2', solver='liblinear')

initial_score = cross_val_score(logreg_new, X, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Final accu   racy : {} ".format(initial_score))

print("--- %s seconds --- for %s" % ((time.time() - start_time), "Best Param Execution on LR"))
