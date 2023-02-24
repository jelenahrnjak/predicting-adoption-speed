import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb


def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(['Name', 'Description', 'RescuerID', 'PetID', 'VideoAmt', 'PhotoAmt', 'State'], axis=1)
    data = df.values.tolist()
    return data


def split_data(data, test_size=0.1, val_size=0.2, random_state=42):
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_random_forest(X_train, y_train): # ovo za sve
    rf = RandomForestClassifier()
    best_params = grid_search(X_train, y_train, rf)
    rf = RandomForestClassifier(**best_params)
    rf.fit(X_train, y_train)
    return rf

def train_gradient_boosting(X_train, y_train):
    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)
    return gb

def train_xgboost(X_train, y_train):
    xgboost = xgb.XGBClassifier()
    xgboost.fit(X_train, y_train)
    return xgboost

def train_bagging(X_train, y_train):
    bagging = BaggingClassifier()
    bagging.fit(X_train, y_train)
    return bagging

def train_light_gbm(X_train, y_train, X_val, y_val):
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
        verbose_eval=False
    )

    return lgb_model

def evaluate_model(model, X_val, y_val, X_test, y_test):
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions, average='weighted')
    print('Validation accuracy:', val_accuracy)
    print('Validation F1 score:', val_f1)

    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average='weighted')
    print('Test accuracy:', test_accuracy)
    print('Test F1 score:', test_f1)


def grid_search(X_train, y_train, rfc):
    # param_grid = {
    #     'n_estimators': [299.5, 300, 300.5],
    #     'max_depth': [34.5, 35, 35.5],
    #     'min_samples_split': [25.5, 25, 24.5],
    # }

    param_grid = {
        'n_estimators': [250, 300, 350], #broj drveca u sumi
        'max_depth': [20, 35, 45], #maksimalna dubina
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'random_state': [42]
    }

    # Create a Grid Search object
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Fit the Grid Search object to the training data
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters and corresponding score
    print('Best hyperparameters:', grid_search.best_params_)
    print('Best score:', grid_search.best_score_)

    best_params = grid_search.best_params_
    return best_params

def main():
    print("u mainu sam")
    data = load_data('train.csv')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    model = train_random_forest(X_train, y_train)
    print('Random forest:')
    evaluate_model(model, X_val, y_val, X_test, y_test)

if __name__ == '__main__':
    main()