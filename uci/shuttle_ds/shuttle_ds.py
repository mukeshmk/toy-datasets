import os
import pandas as pd
from sklearn import linear_model, svm, gaussian_process, ensemble, tree
from sklearn import preprocessing as pp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.fml import FMLClient as fml

# init FMLearn
f = fml(debug=True)

# import some data to play with
path = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(path + "/data/shuttle.trn", sep=' ')
df.columns =['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'class']
df.dropna(inplace=True)

df_test = pd.read_csv(path + "/data/shuttle.tst", sep=' ')
df_test.columns =['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'class']
df_test.dropna(inplace=True)

df = df.append(df_test)
df = df.reset_index()

del df['index']

print(df.head())
print(df.shape)

X = df[df.columns.difference(['class'])]
y = df['class']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=123)

f.set_dataset(x_train, y_train, x_train, y_train)

models = []
#models.append(('LR', linear_model.LogisticRegression(C=1e5, solver='lbfgs', max_iter=50000, multi_class='auto')))
#models.append(('SVC', svm.SVC(gamma='scale', max_iter=50000)))
#models.append(('LSVC', svm.LinearSVC(max_iter=50000)))
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
