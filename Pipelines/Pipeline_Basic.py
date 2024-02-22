##################################################################################
# Pipelines - Basic Template
##################################################################################

# Import required Python packages

import pandas as pd 
import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.utils import shuffle

# Import sample data

data_for_model = pd.read_pickle(r"C:\Users\Luca\Documents\Documents\DataScienceInfinity\data\abc_regression_modelling.p", )


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


# Specify numeric and categorical features

numeric_features = ['distance_from_store', 'credit_score', 'total_sales', 'total_items',
       'transaction_count', 'product_area_count', 'average_basket_value']


categorical_features = ["gender"]


##################################################################################
# Set up pipelines
##################################################################################

# Numerical Feature Transformer - imputes and scales numerical data

numeric_transformer = Pipeline(steps = [("imputer", SimpleImputer()),
                                         ("scaler", StandardScaler())])

# Categorical Feature Transformer - replaces missing values with u for unknown and one hot encodes


categorical_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy = "constant", fill_value="u")), 
                                             ("ohe", OneHotEncoder(handle_unknown="ignore"))])

# Preprocessing Pipeline



preprocessing_pipeline = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])


##################################################################################
# Apply the Pipeline
##################################################################################


# Gradient Boosting Regressor (faster)

clf = Pipeline(steps = [("preprocessing_pipeline", preprocessing_pipeline),
                        ("regressor", GradientBoostingRegressor(random_state=42))])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
r2_score(y_test, y_pred)



# Histogram-Based Gradient Boosting Regressor (faster)

clf = Pipeline(steps = [("preprocessing_pipeline", preprocessing_pipeline),
                        ("regressor", HistGradientBoostingRegressor(random_state=42))])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
r2_score(y_test, y_pred)

##################################################################################
# Save the Pipeline
##################################################################################

import joblib
joblib.dump(clf, "Pipelines/model.joblib")


##################################################################################
# Import pipeline object and test predictions on new data
##################################################################################

# Import required Python packages

import joblib 
import pandas as pd 
import numpy as np

# Import pipeline

clf = joblib.load("Pipelines/model.joblib")

# Create new data

new_data = pd.DataFrame({"distance_from_store" : [2.1, 4.2, 1],
                         "credit_score" : [0.1, 0.89, 0.8],
                         "total_sales" : [3241, 382, 7830], 
                         "total_items" : [32, 392, 283], 
                         "transaction_count" : [43, 23, 67], 
                         "product_area_count" : [1,3,4],
                         "average_basket_value" : [np.nan, 73.22, 89.22],
                         "gender" : ["M", "M", "F"]})

# Pass new data in and receive predictions


clf.predict(new_data)


