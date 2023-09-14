import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import metrics as sm
from sklearn import linear_model

# Importing Dataset , salary dataset is located at D:\\Krishna_Python\\ML_Assignment\\Salary_dataset.csv

salary=pd.read_csv("D:\\Krishna_Python\\ML_Assignment\\Salary_dataset.csv")
salary.head()
#View summry of data
salary.info()
#find null values

salary.isna().sum()
#find duplicates
salary.duplicated().sum()
#columns
salary.columns
#drop unwanted column
salary.drop(["Unnamed: 0"],axis=1,inplace=True)
salary.head()

#splitting salary data into train and test
X = salary['YearsExperience']
y = salary['Salary']
plt.scatter(X,y,color = "blue")

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test

#Data Transformation
X_train = np.array(X_train).reshape((len(X_train),1))
y_train = np.array(y_train).reshape((len(y_train),1))
X_train,y_train
X_test = np.array(X_test).reshape(len(X_test), 1)
y_test = np.array(y_test).reshape(len(y_test), 1)
X_test,y_test
#Training Model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

#Predicting Salary using Linear Model
y_train_pred = model.predict(X_train)
y_train_pred
plt.figure()
plt.scatter(X_train, y_train, color='blue', label="True Value")
plt.plot(X_train, y_train_pred, color='black', linewidth=2, label="Prediction")
plt.xlabel("Years of Experiences")
plt.ylabel("Salary")
plt.title('Prediction Result of Training Data')
plt.legend()
plt.show()

#we need to check if the linear model can perform well on our test set (unknown data).
y_test_pred = model.predict(X_test)
y_test_pred
plt.figure()
plt.scatter(X_test, y_test, color='green', label='True Value')
plt.plot(X_test, y_test_pred, color='black', linewidth=2, label='Prediction')
plt.xlabel("Years of Experiences")
plt.ylabel("Salary")
plt.title('Prediction Result of Test data')
plt.legend()
plt.show()

#Model Evaluation
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))




""" Implementing linear regression from scratch """
#import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


salary=pd.read_csv("D:\\Krishna_Python\\ML_Assignment\\Salary_dataset.csv")
salary.head()


X = salary['YearsExperience'].values
Y = salary['Salary'].values

  #creating linear regression function

def linear_regression(x, y):
    #calculate the length of a variable.
    N = len(x)
    print(N)
    x_mean = x.mean()
    y_mean = y.mean()

    # B1 -slope and B0- intercept.
    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean)**2).sum()
    B1 = B1_num / B1_den
    B0 = y_mean - (B1*x_mean)
    reg_line = 'y = {} + {}β'.format(B0, round(B1, 3))
    return (B0, B1, reg_line)

N = len(X)
x_mean = X.mean()
y_mean = Y.mean()

  #Calculating numerator and deniminator seperately
B1_num = ((X - x_mean) * (Y - y_mean)).sum()
B1_den = ((X - x_mean)**2).sum()
B1 = B1_num / B1_den
B0 = y_mean - (B1 * x_mean)

  #Regression Line Fit verification
def corr_coef(x, y):
    N = len(x)
    num = (N * (x*y).sum()) - (x.sum() * y.sum())
    den = np.sqrt((N * (x**2).sum() - x.sum()**2) * (N * (y**2).sum() - y.sum()**2))
    R = num / den
    return R

#printing data
B0, B1, reg_line = linear_regression(X, Y)
print('Regression Line: ', reg_line)
R = corr_coef(X, Y)
print('Correlation Coef.: ', R)
print('R square value: ', R**2)
print('"Goodness of Fit": ', R**2)



plt.figure(figsize=(12,5))
plt.scatter(X, Y, s=300, linewidths=1, edgecolor='black')
text = '''X Mean: {} Years
  Y Mean: ${}
  R: {}
  R^2: {}
  y = {} + {}X'''.format(round(X.mean(), 2),
                         round(Y.mean(), 2),
                         round(R, 4),
                         round(R**2, 4),
                         round(B0, 3),
                         round(B1, 3))

plt.text(x=1, y=100000, s=text, fontsize=12, bbox={'facecolor': 'grey', 'alpha': 0.2, 'pad': 10})
plt.title('How Experience Affects Salary')
plt.xlabel('Years of Experience', fontsize=15)
plt.ylabel('Salary', fontsize=15)
plt.plot(X, B0 + B1*X, c = 'r', linewidth=5, alpha=.5, solid_capstyle='round')
plt.scatter(x=X.mean(), y=Y.mean(), marker='*', s=10**2.5, c='r') # average point
reg_line = 'y = {} + {}β'.format(B0, round(B1, 3))


###prediction function
def predict(B0, B1, new_x):
    y = B0 + B1 * new_x
    return y

#linear regression error for both train and test 
# split data into a training and test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test

#using test data
y_pred=[]
for elem in X_test:
  y_pred.append(predict(B0, B1, elem))

print("Mean absolute error for test set :", metrics.mean_absolute_error(y_test,y_pred))
print("Mean square error for test set:", metrics.mean_squared_error(y_test,y_pred))

#using train data
y_pred=[]
for elem in X_train:
  y_pred.append(predict(B0, B1, elem))

print("Mean absolute error (MAE) training set:", metrics.mean_absolute_error(y_train,y_pred))
print("Mean square error (MSE) training set:", metrics.mean_squared_error(y_train,y_pred))


#Maximum likelihood estimator f

#segregating dependent and independent variables
X_m = salary['YearsExperience'][:100]
Y_m = salary['Salary'][:100]

#Splitting data in training set and test set
x_train,x_test,y_train,y_test=train_test_split(X_m,Y_m,test_size=0.3,random_state=42)

#log likelihood function declaration
def lik(parameters, x, y):
    m = parameters[0]
    b = parameters[1]
    sigma = parameters[2]

    y_exp = m * x + b

    L = np.sum(stats.norm.logpdf(y - y_exp, loc = 0, scale=sigma))
    return -L


def constraints(parameters):
    sigma = parameters[2]
    return sigma

cons = {
    'type': 'ineq',
    'fun': constraints
}

from scipy.optimize import minimize
import scipy.stats as stats
#Executing MLE
lik_model = minimize(lik, np.array([2, 2, 2]), args=(X_m,Y_m,), constraints=cons)
lik_model

#Plotting datapoint using MLE
fig, ax = plt.subplots()
ax.plot(X,Y, 'b*')
xx = np.linspace(np.min(X), np.max(X), 100)
yy = lik_model.x[0] * xx +  lik_model.x[1]
ax.plot(xx,yy, 'r-')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

#Calculating predicted values for test set
y_pred_mle = lik_model.x[0] * x_test +  lik_model.x[1]
y_pred_mle
# Report error for both training and testing set

#Printing mean absolute error for test set
print("Mean absolute error (MAE) using MLE:", metrics.mean_absolute_error(y_test,y_pred_mle))

#Printing mean square error for test set
print("Mean square error (MSE) using MLE:", metrics.mean_squared_error(y_test,y_pred_mle))



#Calculating predicted values for training set
y_pred_for_train_mle = lik_model.x[0] * x_train +  lik_model.x[1]

#Printing mean absolute error for training set
print("Mean absolute error (MAE) for training set using MLE:", metrics.mean_absolute_error(y_train,y_pred_for_train_mle))

#Printing mean square error for training set
print("Mean square error (MSE) for training set using MLE:", metrics.mean_squared_error(y_train,y_pred_for_train_mle))






