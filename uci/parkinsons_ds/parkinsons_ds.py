import os
import pandas as pd
from sklearn import linear_model, svm, gaussian_process, ensemble, tree
from sklearn import preprocessing as pp
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.fml import FMLClient as fml

# init FMLearn
f = fml(debug=True)

# import some data to play with
path = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(path + "/data/parkinsons_updrs.data", sep=',')
df.dropna(inplace=True)

print(df.head())
print(df.shape)

X = df[df.columns.difference(['PPE'])]
y = df['PPE']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=123)

f.set_dataset(x_train, y_train, x_train, y_train)

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

names = []
scores = []

# finding the best model

for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = mean_squared_error(y_test, y_pred)
    scores.append(score)
    names.append(name)
    #f.publish(model, "MSE", score)

tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)
