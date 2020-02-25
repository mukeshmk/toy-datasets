import time
import numpy as np
from sklearn import datasets, linear_model, svm, gaussian_process, ensemble
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.fml import FMLClient as fml

# Start Time of the execution of the program
start_time = time.time()

# import some data to play with
wine = datasets.load_wine()
X = wine.data
Y = wine.target

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=123)

model = linear_model.LogisticRegression(C=1e5)
# model = svm.SVC()
# model = svm.LinearSVC()
# model = ensemble.AdaBoostClassifier()
# model = ensemble.BaggingClassifier()
# model = ensemble.GradientBoostingClassifier()
# model = ensemble.RandomForestClassifier()
# model = gaussian_process.GaussianProcessClassifier()

# Create an instance of Logistic Regression Classifier and fit the data.
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# The mean squared error
acc = accuracy_score(y_test, y_pred)*100
print('Accuracy: %.2f%%'% acc)

print("--- %s seconds --- for %s" % ((time.time() - start_time), model.__class__))

f = fml()
# f._jprint(f.publish(model, "Accuracy", acc, str(wine.data)))

# f.publish(model, "Accuracy", acc, str(wine.data))

# f._jprint(f.retrieve_all_metrics(str(wine.data)))

f._jprint(f.retrieve_best_metric(str(wine.data), False))