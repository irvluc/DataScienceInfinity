####################################################################################
# Histogram-Based Gradient Boosting for Regression - Basic Template
####################################################################################

# Import required Python packages

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd


# Import sample data

my_df = pd.read_csv(
    r"C:\Users\irvinluc\OneDrive - Diageo\Documents\16. Data Science Infinity\data\sample_data_regression.csv"
)


# Split data into input and output objects

X = my_df.drop(["output"], axis=1)
y = my_df["output"]

# Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Instantiate our model object

regressor = HistGradientBoostingRegressor(random_state=42)


# Train our model

regressor.fit(X_train, y_train)

# Assess model accuracy

y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)
