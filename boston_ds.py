import time
import numpy as np
from sklearn import datasets, linear_model, svm, gaussian_process, ensemble
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.fml import FMLClient as fml

# Start Time of the execution of the program
start_time = time.time()

# Load the boston dataset
boston = datasets.load_boston()

# Use only one feature
boston_X = boston.data[:, np.newaxis, 2]

# Split the data into training/testing sets
boston_X_train = boston_X[:-20]
boston_X_test = boston_X[-20:]

# Split the targets into training/testing sets
boston_y_train = boston.target[:-20]
boston_y_test = boston.target[-20:]

# Create regressor object

regr = linear_model.LinearRegression()
# regr = linear_model.HuberRegressor()
# regr = linear_model.Lasso()
# regr = linear_model.Ridge()
# regr = linear_model.BayesianRidge()
# regr = ensemble.AdaBoostRegressor()
# regr = ensemble.BaggingRegressor()
# regr = ensemble.GradientBoostingRegressor()
# regr = ensemble.RandomForestRegressor()


# Train the model using the training sets
regr.fit(boston_X_train, boston_y_train)

# Make predictions using the testing set
boston_y_pred = regr.predict(boston_X_test)

# The mean squared error
mse = mean_squared_error(boston_y_test, boston_y_pred)
print('Mean squared error: %.2f'% mse)
# The coefficient of determination: 1 is perfect prediction
r2 = r2_score(boston_y_test, boston_y_pred)
print('Coefficient of determination: %.2f' % r2)

print("--- %s seconds --- for %s" % ((time.time() - start_time), regr.__class__))

f = fml()
# f._jprint(f.publish(regr, "MSE", mse, str(boston.data)))

# f.publish(regr, "MSE", mse, str(boston.data))

# f._jprint(f.retrieve_all_metrics(str(boston.data)))

f._jprint(f.retrieve_best_metric(str(boston.data)))