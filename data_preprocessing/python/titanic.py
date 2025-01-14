"""
Coding Exercise 3: Encoding Categorical Data for Machine Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

dataset = pd.read_csv("data_preprocessing/python/titanic.csv")
categorical_features = dataset.select_dtypes(include=["object"]).columns
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), categorical_features)], remainder="passthrough")
X = ct.fit_transform(dataset.drop('Survived', axis=1))
X = np.array(X)
le = LabelEncoder()
y = le.fit_transform(dataset['Survived'])
print(X)
print(y)