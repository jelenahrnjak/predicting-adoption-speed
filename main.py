import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from joblib import dump, load
import os


rf_filename = 'random_forest_model.joblib'
gb_filename = 'gradient_boosting_model.joblib'
xgb_filename = 'xgb_model.joblib'
bagging_filename = 'bagging_model.joblib'

from eda import menu
from sentiments import sent_menu
from sentiments import sent_menu_comb


def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(['Name', 'Description', 'RescuerID', 'PetID', 'VideoAmt', 'PhotoAmt', 'State'], axis=1)
    data = df.values.tolist()
    return data


def load_data_for_merge(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(['Name', 'Description', 'RescuerID', 'PetID', 'VideoAmt', 'PhotoAmt', 'State'], axis=1)
    return df

def split_data(data, test_size=0.1, val_size=0.2, random_state=42):
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_random_forest(X_train, y_train):
    if os.path.isfile(rf_filename) and os.path.getsize(rf_filename) > 0:
        rf = load(rf_filename)
    else:
        rf = RandomForestClassifier()
        param_grid = {
            'n_estimators': [300, 350],
            'max_depth': [35, 10],
            'min_samples_split': [10, 25, 50],
            'random_state': [42],
            'max_features': ['auto'],
            'min_samples_leaf': [1]
        }
        best_params = grid_search(X_train, y_train, rf)
        rf = RandomForestClassifier(**best_params)
        dump(rf, rf_filename)

    rf.fit(X_train, y_train)
    return rf


def train_gradient_boosting(X_train, y_train):
    if os.path.isfile(gb_filename) and os.path.getsize(gb_filename) > 0:
        gb = load(gb_filename)
    else:
        gb = GradientBoostingClassifier()
        param_grid = {'n_estimators': [200],
                      'max_depth': [2],
                      'learning_rate': [0.1],
                      'min_samples_split': [2],
                      'min_samples_leaf': [4],
                      'subsample': [1.0],
                      'max_features': ['log2']}

        # param_grid = {'n_estimators': [50, 100, 200],
        #               'max_depth': [2, 3, 4],
        #               'learning_rate': [0.1, 0.01, 0.001],
        #               'min_samples_split': [2, 4, 6],
        #               'min_samples_leaf': [1, 2, 4],
        #               'subsample': [0.5, 0.75, 1.0],
        #               'max_features': ['sqrt', 'log2', None]}
        best_params = grid_search(X_train, y_train, gb, param_grid)
        gb = GradientBoostingClassifier(**best_params)
        dump(gb, gb_filename)

    gb.fit(X_train, y_train)
    return gb


def train_xgboost(X_train, y_train):
    if os.path.isfile(xgb_filename) and os.path.getsize(xgb_filename) > 0:
        xgboost = load(xgb_filename)
    else:
        xgboost = xgb.XGBClassifier()

        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 1],
            'subsample': [0.5, 0.7, 1],
            'colsample_bytree': [0.5, 0.7, 1],
        }
        best_params = grid_search(X_train, y_train, xgboost, param_grid)
        xgboost = xgb.XGBClassifier(**best_params)
        dump(xgboost, xgb_filename)

    xgboost.fit(X_train, y_train)
    return xgboost


def train_bagging(X_train, y_train):
    if os.path.isfile(bagging_filename) and os.path.getsize(bagging_filename) > 0:
        bagging = load(bagging_filename)
    else:
        bagging = BaggingClassifier()

        param_grid = {
            'n_estimators': [50, 100, 150],  # 150
            'max_samples': [0.5, 0.7, 1],  # 0.5
            'max_features': [0.5, 0.7, 1],  # 0.5
        }
        best_params = grid_search(X_train, y_train, bagging, param_grid)
        bagging = BaggingClassifier(**best_params)
        dump(bagging, bagging_filename)

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



def grid_search(X_train, y_train, rfc, param_grid):
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print('Best hyperparameters:', grid_search.best_params_)
    print('Best score:', grid_search.best_score_)
    best_params = grid_search.best_params_
    return best_params


def main():
    while True:
        print("Select an option:")
        print("1 - Exploratory data analysis")
        print("2 - Model fitting")
        print("3 - Sentiment analysis without combining")

        print("5 - Sentiment analysis with combining")
        print("X - Quit")

        choice = input("Enter option number: ")

        if choice == "1":
            menu()
        elif choice == "2":

            data = load_data('train.csv')
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
            rf_model = train_random_forest(X_train, y_train)
            print('Random forest:')
            evaluate_model(rf_model, X_val, y_val, X_test, y_test)

            gb_model = train_gradient_boosting(X_train, y_train)
            print('Gradient boosting:')
            evaluate_model(gb_model, X_val, y_val, X_test, y_test)

            xgb_model = train_xgboost(X_train, y_train)
            print('XGBoost:')
            evaluate_model(xgb_model, X_val, y_val, X_test, y_test)

            bagging_model = train_bagging(X_train, y_train)
            print('Bagging:')
            evaluate_model(bagging_model, X_val, y_val, X_test, y_test)
        elif choice == "3":
            sent_menu()
        elif choice == "5":
            data = load_data_for_merge('train.csv')
            print(type(data))
            data_and_sentiments = sent_menu_comb(data)
            print(data_and_sentiments.head())
        elif choice == "x" or choice == "X":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select an option from menu.")


if __name__ == '__main__':
    main()
