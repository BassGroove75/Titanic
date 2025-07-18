import xgboost as xgb
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('train.csv')

target_col = "Survived"
feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

x = df.drop(target_col, axis=1)
y = df[feature_cols]

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(x, y)
 #Wibble Wobble