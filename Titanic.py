import xgboost as xgb
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('train.csv')
X = df.drop('Survived', axis=1)
y = df['Survived']

model = xgb.XGBClassifier()
model.fit(X, y)