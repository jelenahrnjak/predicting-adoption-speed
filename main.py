import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb


df = pd.read_csv('train.csv')
print(df.describe())
df = df.drop(['Name', 'Description', 'RescuerID', 'PetID', 'VideoAmt', 'PhotoAmt'], axis=1)

# Separate the labels and data
labels = df.columns.tolist()
data = df.values.tolist()

# Split the data into features and target variable
X = [row[:-1] for row in data] # All columns except the last one are features
y = [row[-1] for row in data] # Last column is the target variable

# popravi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the Random Forest classifier on the training set
rf = RandomForestClassifier()#n_estimators=2000, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the performance of the classifier on the validation set
rf_val_predictions = rf.predict(X_val)
rf_val_accuracy = accuracy_score(y_val, rf_val_predictions)
print('Validation accuracy for Random Forest:', rf_val_accuracy)

# Evaluate the performance of the classifier on the test set
rf_test_predictions = rf.predict(X_test)
rf_test_accuracy = accuracy_score(y_test, rf_test_predictions)
print('Test accuracy for Random Forest:', rf_test_accuracy)

# GRADIENT BOOSTING

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)


gb_val_predictions = gb.predict(X_val)
gb_val_accuracy = accuracy_score(y_val, gb_val_predictions)
print('\nValidation accuracy for Gradient Boosting:', gb_val_accuracy)

gb_test_predictions = gb.predict(X_test)
gb_test_accuracy = accuracy_score(y_test, gb_test_predictions)
print('Test accuracy for Gradient Boosting:', gb_test_accuracy)

# XGBoost

xgb = xgb.XGBClassifier()
xgb.fit(X_train, y_train)


xgb_val_predictions = xgb.predict(X_val)
xgb_val_accuracy = accuracy_score(y_val, xgb_val_predictions)
print('\nValidation accuracy for XGBoost:', xgb_val_accuracy)

xgb_test_predictions = xgb.predict(X_test)
xgb_test_accuracy = accuracy_score(y_test, xgb_test_predictions)
print('Test accuracy for XGBoost:', xgb_test_accuracy)

# Bagging
bagging = BaggingClassifier()
bagging.fit(X_train, y_train)

bagging_val_predictions = bagging.predict(X_val)
bagging_val_accuracy = accuracy_score(y_val, bagging_val_predictions)
print('\nValidation accuracy for Bagging:', bagging_val_accuracy)

bagging_test_predictions = bagging.predict(X_test)
bagging_test_accuracy = accuracy_score(y_test, bagging_test_predictions)
print('Test accuracy for Bagging:', bagging_test_accuracy)

# LightGBM

X_train2 = np.array(X_train)
y_train2 = np.array(y_train)

X_val2 = np.array(X_val)
y_val2 = np.array(y_val)

train_data = lgb.Dataset(X_train2, label=y_train2)
val_data = lgb.Dataset(X_val2, label=y_val2)

params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 5,
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'n_jobs': -1,
    'seed': 42
}

# Train the LightGBM model
lgb_model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=1000,
)

# Evaluate the performance of the model on the validation set
lgb_val_predictions = lgb_model.predict(X_val)
lgb_val_predictions = [list(x).index(max(x)) for x in lgb_val_predictions]
lgb_val_accuracy = accuracy_score(y_val, lgb_val_predictions)
print('Validation accuracy for LightGBM:', lgb_val_accuracy)

# Evaluate the performance of the model on the test set
lgb_test_predictions = lgb_model.predict(X_test)
lgb_test_predictions = [list(x).index(max(x)) for x in lgb_test_predictions]
lgb_test_accuracy = accuracy_score(y_test, lgb_test_predictions)
print('Test accuracy for LightGBM:', lgb_test_accuracy)