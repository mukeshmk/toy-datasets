import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import preprocessing as pp

#df_movie=pd.read_csv('data/ml-1m/movies.dat', sep = '::', engine='python')
#df_movie.columns =['MovieIDs','MovieName','Category']
#df_movie.dropna(inplace=True)
#print(df_movie.head())

df_rating = pd.read_csv("data/ml-1m/ratings.dat",sep='::', engine='python')
df_rating.columns =['ID','MovieID','Ratings','TimeStamp']
df_rating.dropna(inplace=True)
#print(df_rating.head())

df_user = pd.read_csv("data/ml-1m/users.dat",sep='::',engine='python')
df_user.columns =['UserID','Gender','Age','Occupation','Zip-code']
df_user.dropna(inplace=True)
#print(df_user.head())

df = df_rating.merge(df_user, left_on='ID', right_on='UserID')
df = df.drop(['TimeStamp', 'UserID', 'Zip-code'], axis=1)
df.dropna(inplace=True)
#print(df.head())

ohe_gender = pp.OneHotEncoder(categories='auto', sparse=False)
ohe_gender_data = ohe_gender.fit_transform(df['Gender'].values.reshape(len(df['Gender']), 1))
dfOneHotGender = pd.DataFrame(ohe_gender_data, columns=['Gender: '+str(i.strip('x0123_')) for i in ohe_gender.get_feature_names()])
df = pd.concat([df, dfOneHotGender], axis=1)
del df['Gender']

X = df[['ID', 'MovieID', 'Gender: M', 'Age', 'Occupation']]
y = df[['Ratings']]
#print(X.head())
#print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print('Training...')
log_reg = linear_model.LogisticRegression(C=1e5, solver='lbfgs', max_iter=10000, multi_class='auto')
log_reg.fit(X_train, y_train)

print('Testing...')
y_pred = log_reg.predict(X_test)

print(accuracy_score(y_test, y_pred))