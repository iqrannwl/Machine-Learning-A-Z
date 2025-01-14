import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


dataset = pd.read_csv("data_preprocessing/python/Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#impute missing values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


#encode catagorical variable independenta variable
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X  = np.array(ct.fit_transform(X))

#encode catagorical variable dependent variable
le = LabelEncoder()
y = le.fit_transform(y)

print(X)
print(y)
