import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

# This code models ridge and lasso regressions and finds the best fit by calculating different values of alpha for
# ridge regression and looking at the mean squared errors of every model.


df = pd.read_csv("std_df_full.csv")  # read data set

# dummies = pd.get_dummies(df[['vaxView', 'demographicClass']])
# print(dummies)
y = df['value']  # assign the predictor to y
X_ = df.drop(['value', 'lowerLimit', 'confidenceInterval', 'upperError'],
             axis=1).astype('float64')  # dropping the predictor

# Define the feature set X.
X = pd.concat([X_], axis=1)
X.info()

# Ridge Regression
alphas = 10**np.linspace(10, -2, 100)*0.5
print("Alphas;", alphas)

ridge = Ridge(normalize=True)
coefs = []

for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

np.shape(coefs)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.show()

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)


ridge2 = Ridge(alpha=4, normalize=True)
ridge2.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred2 = ridge2.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge2.coef_, index=X.columns)) # Print coefficients
print("MSE with alpha 4 = ", mean_squared_error(y_test, pred2))          # Calculate the test MSE

ridge3 = Ridge(alpha=10**10, normalize=True)
ridge3.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred3 = ridge3.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge3.coef_, index=X.columns)) # Print coefficients
print("MSE with alpha 10**10 = ", mean_squared_error(y_test, pred3))          # Calculate the test MSE

ridge2 = Ridge(alpha=0, normalize=True)
ridge2.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred = ridge2.predict(X_test)            # Use this model to predict the test data
print(pd.Series(ridge2.coef_, index=X.columns)) # Print coefficients
print("MSE with alpha 0 = ", mean_squared_error(y_test, pred))           # Calculate the test MSE

ridgecv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', normalize=True)
ridgecv.fit(X_train, y_train)
print(ridgecv.alpha_)

ridge4 = Ridge(alpha=ridgecv.alpha_, normalize=True)
ridge4.fit(X_train, y_train)
print("MSE with alphas calculated using generalized cross-validation = ",
      mean_squared_error(y_test, ridge4.predict(X_test)))
ridge4.fit(X, y)
print(pd.Series(ridge4.coef_, index=X.columns))


# Lasso Regression

lasso = Lasso(max_iter=10000, normalize=True)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()
ax.plot(alphas * 2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.show()

lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)
lassocv.fit(X_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)
print("\n\nLasso MSE:", mean_squared_error(y_test, lasso.predict(X_test)))
print(pd.Series(lasso.coef_, index=X.columns))
