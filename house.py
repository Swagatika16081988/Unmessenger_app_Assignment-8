import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import pickle

# Read the dataset
df = pd.read_csv(r"C:\Users\Swagatika Samal\Desktop\TRAINITY\CERTIFICATES\Final Assesment\data.csv")

# Select relevant columns
columns = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'price']
df = df[columns]

# Split features (X) and target variable (y)
X = df[['bedrooms', 'bathrooms', 'floors', 'yr_built']]
y = df['price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Define the hyperparameter space
model = XGBRegressor()

n_estimators   = [100, 200, 500]
learning_rates = [0.03, 0.1, 0.3]
objectives     = ['reg:squarederror']

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators' : n_estimators,
    'learning_rate': learning_rates,
    'objective'    : objectives
}

# Perform grid search
grid_cv = GridSearchCV(estimator=model,
                       param_grid=hyperparameter_grid,
                       scoring='neg_mean_absolute_error',
                       return_train_score=True)

grid_cv.fit(X_train, y_train)

# Get the best estimator from grid search
best_regressor = grid_cv.best_estimator_

# Make predictions on test data
y_pred = best_regressor.predict(X_test)

# Save the trained model
pickle.dump(best_regressor, open('model.pkl', 'wb'))



# Input data provided as a list of integers
inputt = [2, 2, 3, 1976]

# Convert the list into a numpy array and reshape it
final = np.array(inputt).reshape(1, -1)

# Make predictions with the trained regressor
b = best_regressor.predict(final)

print(b)

