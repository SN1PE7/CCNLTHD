import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# load file data
df = pd.read_csv('dataset/housing.csv')

# chia tap train + test
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# preprocessing
num_features = ["longitude", "latitude", "housing_median_age", "total_rooms",
                "total_bedrooms", "population", "households", "median_income"]
cat_features = ["ocean_proximity"]

# pipeline num data
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # fill missing data = trung vi
    ('scaler', StandardScaler())
])

# pipeline category data
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), # fill missing data = most frequent 
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])

# gom lai thanh preprocessor
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# pipeline hoan` chinh?
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42)) # dung` random forest
])

# bat dau` train
full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE = {rmse:.2f}")

# luu model
os.makedirs('model', exist_ok=True)
joblib.dump(full_pipeline, 'model/model.joblib')