import time
import numpy as np
from sklearn import datasets, linear_model, svm, gaussian_process, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.fml import FMLClient as fml

# Start Time of the execution of the program
start_time = time.time()

# import some data to play with
linnerud = datasets.load_linnerud()
X = linnerud.data
Y = linnerud.target

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=123)

model = linear_model.LinearRegression()
# model = linear_model.Lasso()
# model = linear_model.Ridge()
# model = gaussian_process.GaussianProcessRegressor()
# model = ensemble.BaggingRegressor()
# model = ensemble.RandomForestRegressor()

# Create an instance of Logistic Regression Classifier and fit the data.
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# The mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error: %.2f'% mse)

print("--- %s seconds --- for %s" % ((time.time() - start_time), model.__class__))

f = fml()
# f._jprint(f.publish(model, "MSE", mse, str(linnerud.data)))

# f.publish(model, "MSE", mse, str(linnerud.data))

# f._jprint(f.retrieve_all_metrics(str(linnerud.data)))

f._jprint(f.retrieve_best_metric(str(linnerud.data), False))