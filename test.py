import time
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.fmlearn import FMLClient as fml

# Start Time of the execution of the program
start_time = time.time()

# init FMLearn
f = fml(debug=False)

# import some data to play with
db = datasets.load_breast_cancer()
X = db.data
Y = db.target

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=123)

f.set_dataset(x_train, y_train, x_train, y_train)

model = linear_model.LogisticRegression(C=1e5, solver='lbfgs', max_iter=10000, multi_class='auto')
#model.fit(x_train, y_train)
#y_pred = model.predict(x_test)
#score = accuracy_score(y_test, y_pred)
score = 0

# Send to FMLearn
#f._jprint(f.publish(model, "Accuracy", score, {'c':1e5, 'solver':'lbfgs', 'max_iter':10000, 'multi_class':'auto'}))

#f._jprint(f.publish(model, "Accuracy", 1.05))

#f._jprint(f.retrieve_all_metrics())

f._jprint(f.retrieve_all_metrics())
f._jprint(f.predict_metric())
f._jprint(f.retrieve_best_metric(min = False))
f._jprint(f.retrieve_best_metric(min = True))
