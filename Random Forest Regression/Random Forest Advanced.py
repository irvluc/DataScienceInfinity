#############################################################################
# Random Forest for Regression  - ABC Grocery Task
#############################################################################

# 1. Import libraries

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance


# 2. Import sample data


# 2a. Import
data_for_model = pickle.load(
    open(
        r"c:\Users\irvinluc\OneDrive - Diageo\Documents\16. Data Science Infinity\data\abc_regression_modelling.p",
        "rb",
    )
)

# 2b. Drop unnecessary columns

data_for_model.drop("customer_id", axis=1, inplace=True)

# 2c. Shuffle data

data_for_model = shuffle(data_for_model, random_state=42)

# 3. Deal with missing values

data_for_model.isna().sum()
data_for_model.dropna(how="any", inplace=True)

# Do not need to deal with outliers with decision trees as they split the data and deal with them for you

# 5. Split input variables and output variable

X = data_for_model.drop(["customer_loyalty_score"], axis=1)
y = data_for_model["customer_loyalty_score"]

# 6. Split out training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Deal with categorical variables

categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse=False, drop="first")

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder_feature_names)
X_train = pd.concat(
    [X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1
)
X_train.drop(categorical_vars, axis=1, inplace=True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder_feature_names)
X_test = pd.concat(
    [X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1
)
X_test.drop(categorical_vars, axis=1, inplace=True)


# 9. Model Training

regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)


# 10. Model assessment

# Predict on the test test
y_pred = regressor.predict(X_test)

# Calculate R-squared - used to measure how well the model explains the variation around the mean for the dependent variable in a single linear regression i.e. all features included in the model will explain x% of the variation around the mean of the predicted variable
r_squared = r2_score(y_test, y_pred)
print(r_squared)

# Cross validation
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv=cv, scoring="r2")
cv_scores.mean()

# Calculate adjusted R-squared - used to measure how well the model explains the variation around the mean for the dependent variable in a multiple linear regression i.e. all features included in the model will explain x% of the variation around the mean of the predicted variable
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (
    num_data_points - num_input_vars - 1
)
print(adjusted_r_squared)

# Calculate RMSE - used to assess the accuracy of a model's predictions i.e. the stability of a model
root_mean_squared = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(root_mean_squared)


# Feature Importance

regressor.feature_importances_

feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis=1)
feature_importance_summary.columns = ["input_variable", "feature_importance"]
feature_importance_summary.sort_values(by="feature_importance", inplace=True)


plt.barh(
    feature_importance_summary["input_variable"],
    feature_importance_summary["feature_importance"],
)
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()


# Permutation Importance - decrease in model performance when input features are randomly shuffed, thus getting rid of the input feature's relationship with the output variable. TLDR; we kill some input variables to understand how much accuracy reduces when they are removed.

result = permutation_importance(
    regressor, X_test, y_test, n_repeats=10, random_state=42
)


permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat(
    [feature_names, permutation_importance], axis=1
)
permutation_importance_summary.columns = ["input_variable", "permutation_importance"]
permutation_importance_summary.sort_values(by="permutation_importance", inplace=True)


plt.barh(
    permutation_importance_summary["input_variable"],
    permutation_importance_summary["permutation_importance"],
)
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()


# Predictions under the hood for first customer in the list

new_data = [X_test.iloc[0]]
regressor.estimators_

predictions = []
tree_count = 0
for tree in regressor.estimators_:
    prediction = tree.predict(new_data)[0]
    predictions.append(prediction)
    tree_count += 1

print(predictions)

sum(predictions) / tree_count


# Saving model and one hot encoder as pickle files

pickle.dump(
    regressor,
    open(
        r"C:\Users\irvinluc\OneDrive - Diageo\Documents\16. Data Science Infinity\Random Forest Regression\random_forest_regression_model.p",
        "wb",
    ),
)
pickle.dump(
    one_hot_encoder,
    open(
        r"C:\Users\irvinluc\OneDrive - Diageo\Documents\16. Data Science Infinity\Random Forest Regression\one_hot_encoder.p",
        "wb",
    ),
)
