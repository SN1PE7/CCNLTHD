import pandas as pd
import numpy as np
import joblib
import os
import json
import datetime

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin

# load file data
df = pd.read_csv('dataset/housing.csv')

# Lọc bỏ các căn nhà có giá >= 500,000 để mô hình không học sai phân khúc giá cao
df_filtered = df[df["median_house_value"] < 500000].copy()

# chia tap train + test
X = df_filtered.drop("median_house_value", axis=1)
y = df_filtered["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# FEATURE ENGINEERING
# Xác định index cần dùng (theo thứ tự: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income)
# index: total_rooms=3, total_bedrooms=4, population=5, households=6
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        return np.c_[X, rooms_per_household, bedrooms_per_room, population_per_household]

# preprocessing
num_features = ["longitude", "latitude", "housing_median_age", "total_rooms",
                "total_bedrooms", "population", "households", "median_income"]
cat_features = ["ocean_proximity"]

# pipeline num data

#num_pipeline = Pipeline([
#    ('imputer', SimpleImputer(strategy='median')), # fill missing data = trung vi
#    ('scaler', StandardScaler())
#])
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()), 
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



#full_pipeline = Pipeline([
#    ('preprocessor', preprocessor),
#    ('model', RandomForestRegressor(n_estimators=100, random_state=42)) # dung` random forest
#])

# Định nghĩa Model với các tham số
rf_model = RandomForestRegressor(
    n_estimators= 200,
    max_features= 'log2',
    max_depth= None,
    random_state=42,
    n_jobs=-1  # Chạy đa luồng cho tốc độ tối đa
)

# pipeline hoan` chinh?
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', rf_model) # dung` random forest
])

# bat dau` train
full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
print(f"RMSE = {rmse:.2f}")
print(f"R-squared Score = {r2:.4f}")

metadata = {
    "metrics": {
        "r2_score": round(r2, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2)
    }
}

# luu model
os.makedirs("model", exist_ok=True)
joblib.dump(full_pipeline, 'model/model.joblib')
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=1)