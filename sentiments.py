import json
import os
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score
from unidecode import unidecode
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from joblib import dump, load

nltk.download('stopwords')
nltk.download('punkt')

sentiment_df_file = 'sent.csv'
# putanja do foldera sa .json datotekama
json_folder = "C:\\Users\\Kristina\\OneDrive\\Desktop\\Master\\SIAP\\petfinder-adoption-prediction\\train_sentiment"

rf_filename = 'random_forest_model.joblib'
gb_filename = 'gradient_boosting_model.joblib'
xgb_filename = 'xgb_model.joblib'
bagging_filename = 'bagging_model.joblib'

# putanja do .csv datoteke sa tabelarnim podacima o zivotinjama
csv_file = 'train.csv'

stemmer = SnowballStemmer("english")


def soundex(word):
    word = word.lower()

    # pretvori riječ u niz znakova
    word_list = list(word)

    for i in range(len(word_list)):
        if word_list[i] in ['b', 'f', 'p', 'v']:
            word_list[i] = '1'
        elif word_list[i] in ['c', 'g', 'j', 'k', 'q', 's', 'x', 'z']:
            word_list[i] = '2'
        elif word_list[i] in ['d', 't']:
            word_list[i] = '3'
        elif word_list[i] == 'l':
            word_list[i] = '4'
        elif word_list[i] in ['m', 'n']:
            word_list[i] = '5'
        elif word_list[i] == 'r':
            word_list[i] = '6'

    # ukloni duplikate
    soundex_code = ''.join([word_list[i] for i in range(1, len(word_list)) if word_list[i] != word_list[i - 1]])

    # ukloni nule i dopuni do četiri znaka
    soundex_code = soundex_code.replace('0', '')
    soundex_code = soundex_code.ljust(4, '0')

    return soundex_code


def split_data(x, y, test_size=0.1, val_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def tokenize_text(text):
    # izbacivanje znakova interpukcije
    text = text.translate(str.maketrans('', '', string.punctuation))
    # prebacivanje u mala slova
    text = text.lower()
    # izbacivanje dijakritika
    text = unidecode(text)

    # print(text)
    # Tokenizacija teksta
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # izračunavanje Soundex koda za svaku riječ
    soundex_tokens = [soundex(word) for word in tokens]
    soundex_dict = {}
    filtered_tokens = []
    for i, token in enumerate(tokens):
        soundex_code = soundex_tokens[i]
        if soundex_code not in soundex_dict:
            soundex_dict[soundex_code] = token
            filtered_tokens.append(token)

    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    return stemmed_tokens


def lsa_analysis(tfidf_matrix):
    # latent semantic analysis - tehnika za smanjenje dimenzionalnosti
    n_components = [5000, 10000, 12500, 15000, 20000]
    variance = []

    for n in n_components:
        lsa_model = TruncatedSVD(n_components=n)
        lsa = lsa_model.fit_transform(tfidf_matrix)
        variance.append(sum(lsa_model.explained_variance_ratio_))

    print(variance)


def grid_search(X_train, y_train, rfc, param_grid):
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print('Best hyperparameters:', grid_search.best_params_)
    print('Best score:', grid_search.best_score_)
    best_params = grid_search.best_params_
    return best_params


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


def model_fitting(data, y):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, y)
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


def sent_menu_comb(tabular_data):
    df = pd.read_csv(csv_file)

    d = pd.DataFrame(columns=['Description'])
    s = pd.DataFrame(columns=['Score'])
    m = pd.DataFrame(columns=['Magnitude'])

    for index, row in df.iterrows():

        description = str(row["Description"])
        pet_id = row['PetID']

        json_path = os.path.join(json_folder, pet_id + '.json')

        if os.path.exists(json_path):
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)
                score = data['documentSentiment']['score']
                magnitude = data['documentSentiment']['magnitude']
        else:
            score = 0
            magnitude = 0

        tokens = tokenize_text(description)
        text = ' '.join(tokens)

        d = d.append({'Description': text}, ignore_index=True)
        s = s.append({'Score': score}, ignore_index=True)
        m = m.append({'Magnitude': magnitude}, ignore_index=True)

    if not os.path.exists(sentiment_df_file):

        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(d['Description'])

        # lsa_analysis(tfidf_matrix)
        lsa_model = TruncatedSVD(n_components=12500)  # 12500 dobijeno analizom varijansi u fji iznad
        lsa = lsa_model.fit_transform(tfidf_matrix)
        preprocessed_text = pd.DataFrame(lsa)
        preprocessed_text.to_csv(sentiment_df_file, index=False)

    else:
        preprocessed_text = pd.read_csv(sentiment_df_file)

    s = s['Score']
    m = m['Magnitude']

    print(type(data))
    print(type(preprocessed_text))
    print(type(s))
    print(type(m))

    combined_data = pd.concat([tabular_data, preprocessed_text, s, m], axis=1)

    return combined_data
    #model_fitting(combined_data, y.astype('int'))


def sent_menu():
    df = pd.read_csv(csv_file)

    x = pd.DataFrame(columns=['Description', 'Score', 'Magnitude'])
    y = pd.DataFrame(columns=['AdoptionSpeed'])

    d = pd.DataFrame(columns=['Description'])
    s = pd.DataFrame(columns=['Score'])
    m = pd.DataFrame(columns=['Magnitude'])

    for index, row in df.iterrows():

        description = str(row["Description"])
        adoption_speed = row['AdoptionSpeed']
        pet_id = row['PetID']

        json_path = os.path.join(json_folder, pet_id + '.json')

        if os.path.exists(json_path):
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)
                score = data['documentSentiment']['score']
                magnitude = data['documentSentiment']['magnitude']
        else:
            score = 0
            magnitude = 0

        tokens = tokenize_text(description)
        text = ' '.join(tokens)

        x = x.append(
            {'Description': text, 'Score': score, 'Magnitude': magnitude}, ignore_index=True)

        y = y.append(
            {'AdoptionSpeed': adoption_speed}, ignore_index=True)

        d = d.append({'Description': text}, ignore_index=True)
        s = s.append({'Score': score}, ignore_index=True)
        m = m.append({'Magnitude': magnitude}, ignore_index=True)

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(d['Description'])

    s = s['Score']
    m = m['Magnitude']
    y = y['AdoptionSpeed']

    # lsa_analysis(tfidf_matrix)
    lsa_model = TruncatedSVD(n_components=12500)  # 12500 dobijeno analizom varijansi u fji iznad
    lsa = lsa_model.fit_transform(tfidf_matrix)

    combined_data = pd.concat([pd.DataFrame(lsa), s, m], axis=1)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(combined_data, y.astype('int'))

    svm = SVC()
    svm.fit(X_train, y_train)

    evaluate_model(svm, X_val, y_val, X_test, y_test)

    # accuracy = svm.score(X_test, y_test)
    # print(accuracy)

