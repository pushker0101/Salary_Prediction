import numpy as np
import pandas as pd

file = pd.read_csv('Salary_dataset.csv')
print(file.head())
file.drop('Unnamed: 0', axis=1, inplace=True)
print(file.head())



file.isnull().sum()


import matplotlib.pyplot as plt

plt.scatter(file.YearsExperience, file.Salary)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')

from sklearn import metrics, linear_model, model_selection
x = file.drop(columns = ['Salary'])
y = file['Salary']

x.shape, y.shape

y=y.values.reshape(-1,1)
y.shape

from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x,y, test_size = 0.1 , random_state = 42)

lr_model = linear_model.LinearRegression()
lr_model.fit(x_train,y_train)

y_pred = lr_model.predict(x_test)


accuracy = metrics.r2_score(y_test, y_pred)
MAE = metrics.mean_absolute_error(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)
print(" Linear Regression Model Accuracy = ", accuracy*100, "%")
print("Mean Absolute Error = ", MAE)
print("Mean Squared Error = ", MSE)
