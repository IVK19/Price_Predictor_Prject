from pandas import read_excel, DataFrame
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib

dataset = read_excel('modified_notebooks_data_frame_7_132.xlsx')

X = dataset.drop('цена', axis=1).values
y = dataset['цена'].values

model = RandomForestRegressor(n_estimators=52, max_features=13)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'Accuracy on training set: {model.score(X_train, y_train)}')
print(f'Accuracy on testing set: {model.score(X_test, y_test)}')
print(f'R Squared Error: {r2_score(y_test, y_pred)}')

joblib.dump(model, 'random_forest_model.joblib')