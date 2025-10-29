from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

# Custom Mean Target Encoder
class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.category_means_ = {}
        self.global_means_ = {}

    def fit(self, X, y):
        X = pd.DataFrame(X)  # ensure DataFrame
        y = pd.Series(y)
        for col in X.columns:
            self.category_means_[col] = y.groupby(X[col]).mean()
            self.global_means_[col] = y.mean()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X_encoded = pd.DataFrame(index=X.index)
        for col in X.columns:
            X_encoded[col] = X[col].map(self.category_means_[col])
            X_encoded[col] = X_encoded[col].fillna(self.global_means_[col])
        return X_encoded.values  # shape = (n_samples, n_columns)

mean_cols = [0, 1]
ordinal_col_1 = [2]
ordinal_col_2 = [3]
one_hot_cols = [4, 5, 6, 7, 8]

condition_categories = ['salvage', 'fair', 'missing', 'good', 'like new', 'new', 'excellent']
cylinder_categories = ['3 cylinders', '4 cylinders', 'missing', '5 cylinders', '6 cylinders',
                               'other', '8 cylinders', '10 cylinders', '12 cylinders']

category_encoder = ColumnTransformer([
    ("mean_encoder", MeanTargetEncoder(), mean_cols),
    ("ordinal_encoder_1", OrdinalEncoder(categories=[condition_categories]), ordinal_col_1),
    ("ordinal_encoder_2", OrdinalEncoder(categories=[cylinder_categories]), ordinal_col_2), 
    ("one_hot_encoder", OneHotEncoder(sparse_output=False), one_hot_cols)
])

category_scaler = ColumnTransformer([
    ("scaler", StandardScaler(), mean_cols+ordinal_col_1+ordinal_col_2)
], remainder="passthrough")

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", category_encoder),
    ("scaler", category_scaler)
])