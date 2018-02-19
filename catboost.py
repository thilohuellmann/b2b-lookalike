import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

#Read trainig and testing files
train = pd.read_csv("train.csv", delimiter="\t")
test = pd.read_csv("test.csv", delimiter="\t")

X_train = train.drop(['label'], axis=1)
y_train = train.label

X_test = test.drop(['label'], axis=1)
y_test = test.label

categorical_features_indices = np.where(X_train.dtypes == np.object)[0]

#Identify the datatype of variables
#train.dtypes

# Model

model = CatBoostClassifier(iterations=200, 
                           learning_rate=0.02, 
                           depth=8,
                           use_best_model=True, 
                           loss_function='Logloss', 
                           od_type='Iter', 
                           od_wait=50, 
                           eval_metric='Accuracy') # which settings?
                           
model.fit(X_train, y_train, cat_features=categorical_features_indices, use_best_model=True, eval_set=(X_test, y_test),plot=True)

