"""
Evaluating regression performance

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 244). Packt Publishing. Kindle Edition. 
"""

"""
1) The mean squared error evaluates a regression model. It measures the squared loss corresponding to the expected value.
Sometimes the square root is taken on top of the MSE, in order to bring it back into the original scale of the 
target variable estimated. 
This root term is called root mean squared error (RMSE). RMSE has the benefit of penalizing large errors more
since we first calculate the square of an error. 

2) The mean absolute error (MAE) measures absolute loss. It uses the same scale as the target variable and gives
us an idea of how close the predictions are to the actual values.

*** For both the MSE/MAE: the smaller the value, the better the regression model. 

3) R^2 (r-squared) indicates the goodness of the fit of a regression model. It is the fraction of the dependent
variable variation that a regression model is able to explain. The variant adjusted R^2 adjusts for the number of features
in a model relative to the number of data points.
"""

from sklearn import datasets

# we will work on the diabetes dataset again & fine-tune the parameters of the 
# linear regression model using the grid search technique

diabetes = datasets.load_diabetes()
num_test = 30   # the last 30 samples as testing set
X_train = diabetes.data[:-num_test, :]
y_train = diabetes.target[:-num_test]
X_test = diabetes.data[-num_test:, :]
y_test = diabetes.target[-num_test:]

param_grid = {
    "alpha": [1e-07, 1e-06, 1e-05],
    "penalty": [None, "12"],
    "eta0": [0.03, 0.05, 0.1],
    "max_iter": [500, 1000]
}

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV

regressor = SGDRegressor(loss = 'squared_error',
                         learning_rate = 'constant',
                         random_state = 42)
grid_search = GridSearchCV(regressor, param_grid, cv = 3)

# we obtain the optimal set of parameters

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
regressor_best = grid_search.best_estimator_

# we predict the testing set with the optimal model

predictions = regressor_best.predict(X_test)

# we evaluate the performance on testing sets based on the MSE, MAE, & R^2

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print(mean_squared_error(y_test, predictions))

print(mean_absolute_error(y_test, predictions))

print(r2_score(y_test, predictions))