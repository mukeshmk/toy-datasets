import time
import numpy as np
from sklearn import datasets, linear_model, svm, gaussian_process, ensemble
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.fml import FMLClient as fml

# Start Time of the execution of the program
start_time = time.time()

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create regressor object

# regr = linear_model.LinearRegression()
# regr = linear_model.LogisticRegression()
regr = linear_model.HuberRegressor()
# regr = linear_model.Lasso()
# regr = linear_model.Ridge()
# regr = linear_model.BayesianRidge()
# regr = svm.LinearSVC()
# regr = gaussian_process.GaussianProcessRegressor()
# regr = ensemble.AdaBoostRegressor()
# regr = ensemble.BaggingRegressor()
# regr = ensemble.GradientBoostingRegressor()
# regr = ensemble.RandomForestRegressor()


# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The mean squared error
mse = mean_squared_error(diabetes_y_test, diabetes_y_pred)
print('Mean squared error: %.2f'% mse)
# The coefficient of determination: 1 is perfect prediction
r2 = r2_score(diabetes_y_test, diabetes_y_pred)
print('Coefficient of determination: %.2f' % r2)

print("--- %s seconds --- for %s" % ((time.time() - start_time), regr.__class__))

f = fml()
# f._jprint(f.publish(regr, "MSE", mse, str(diabetes.data)))

# f.publish(regr, "MSE", mse, str(diabetes.data))

# f._jprint(f.retrieve_all_metrics(str(diabetes.data)))

f._jprint(f.retrieve_best_metric(str(diabetes.data), True))