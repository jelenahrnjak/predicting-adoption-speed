import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from joblib import dump, load
import os
from sklearn.impute import SimpleImputer


rf_filename = 'random_forest_model.joblib'
gb_filename = 'gradient_boosting_model.joblib'
xgb_filename = 'xgb_model.joblib'
bagging_filename = 'bagging_model.joblib'

from eda import menu
from image_work import main_images
from sentiments import sent_menu
from sentiments import sent_menu_comb


def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(['Name', 'Description', 'RescuerID', 'VideoAmt', 'PhotoAmt', 'State'], axis=1) #izbaciti one sa quantity > 1
    df['PetID'] = df['PetID'].astype(str)
    df = df.set_index('PetID') #podesavanje da se spaja posle po id-u
    return df

def split_data(data, test_size=0.1, val_size=0.2, random_state=42):
    data.columns = data.columns.astype(str)
    imputer = SimpleImputer(strategy='mean')
    df2 = imputer.fit_transform(data)
    df = pd.DataFrame(df2, columns=data.columns)
    data = df.values.tolist()
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def combine_image_adn_tabular_data(image_data, tabular_data):
    if not image_data:
        raise ValueError("image_data is empty or None")
    combined_data = pd.concat([pd.DataFrame.from_dict(image_data, orient='index'), tabular_data], axis=1)
    return combined_data

def train_random_forest(X_train, y_train):
    if os.path.isfile(rf_filename) and os.path.getsize(rf_filename) > 0:
        rf = load(rf_filename)
    else:
        rf = RandomForestClassifier()
        param_grid = {
            'n_estimators': [100, 300, 400],
            'max_depth': [30, 20, 10],
            'min_samples_split': [10, 30, 50],
            'random_state': [42],
            'max_features': ['auto'],
            'min_samples_leaf': [1, 3]
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
        param_grid = {'n_estimators': [20, 200, 400],
                       'max_depth': [2, 4, 6],
                       'learning_rate': [0.1, 0.01, 0.001],
                       'min_samples_split': [2, 4 , 7],
                       'min_samples_leaf': [1, 3, 5],
                       'subsample': [0.5, 0.75, 1.0],
                       'max_features': ['sqrt', 'log2', None]}
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
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 1],
            'subsample': [0.5, 0.7, 1],
            'colsample_bytree': [0.2, 0.5, 1],
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
            'n_estimators': [150, 200, 300], #150
            'max_samples': [0.5, 0.7, 0.2], #0.5
            'max_features': [0.5, 0.7, 0.2], #0.5
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
    print('Validation accuracy: ', val_accuracy)
    print('Validation F1 score: ', val_f1)

    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average='weighted')
    print('Test accuracy: ', test_accuracy)
    print('Test F1 score: ', test_f1)

    weights = 'quadratic'

    val_kappa = cohen_kappa_score(y_val, val_predictions, weights)
    test_kappa = cohen_kappa_score(y_test, test_predictions, weights=weights)
    print('Validation quadratic weighted kappa: ', val_kappa)
    print('Test quadratic weighted kappa: ', test_kappa)



def grid_search(X_train, y_train, rfc, param_grid):
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print('Best hyperparameters:', grid_search.best_params_)
    print('Best score:', grid_search.best_score_)
    best_params = grid_search.best_params_
    return best_params


def model_fitting(data):
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

def main():
    while True:
        print("Select an option:")
        print("1 - Exploratory data analysis")
        print("2 - Model fitting")
        print("3 - Sentiment analysis without combining")
        print("4 - Model fitting with image and tabular data")
        print("5 - Sentiment analysis with combining with tabular data")
        print("6 - Model fitting with combined tabular data, images and sentiments ")
        print("X - Quit")

        choice = input("Enter option number: ")

        if choice == "1":
            menu()
        elif choice == "2":
            data = load_data('train.csv')
            model_fitting(data)
        elif choice == "3":
            sent_menu()
        elif choice == "4":

            tabular_data = load_data('train.csv')
            image_data = main_images('train_images')
            combined_data = combine_image_adn_tabular_data(image_data, tabular_data)
            model_fitting(combined_data)

        elif choice == "5":

            data = load_data('train.csv')
            print(type(data))
            data_and_sentiments = sent_menu_comb(data)
            print(data_and_sentiments.head())
            model_fitting(combined_data)

        elif choice == "6":

            data = load_data('train.csv')
            data_and_sentiments = sent_menu_comb(data)
            image_data = main_images('train_images')
            combined_data = combine_image_adn_tabular_data(image_data, data_and_sentiments)
            model_fitting(combined_data)

        elif choice == "x" or choice == "X":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select an option from menu.")


if __name__ == '__main__':
    main()
