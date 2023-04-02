import json
import os
import string

import nltk
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from unidecode import unidecode
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy.sparse import hstack
import numpy as np
from sklearn.decomposition import TruncatedSVD

nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('punkt')

# putanja do foldera sa .json datotekama
json_folder = "C:\\Users\\Kristina\\OneDrive\\Desktop\\Master\\SIAP\\petfinder-adoption-prediction\\train_sentiment"

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


def sent_menu():
    df = pd.read_csv(csv_file)

    x = pd.DataFrame(columns=['Description', 'Score', 'Magnitude'])
    y = pd.DataFrame(columns=['AdoptionSpeed'])

    d = pd.DataFrame(columns=['Description'])
    s = pd.DataFrame(columns=['Score'])
    m = pd.DataFrame(columns=['Magnitude'])

    a = pd.DataFrame(columns=['Age'])
    b = pd.DataFrame(columns=['Breed1'])
    g = pd.DataFrame(columns=['Gender'])
    ms = pd.DataFrame(columns=['MaturitySize'])
    fl = pd.DataFrame(columns=['FurLength'])

    for index, row in df.iterrows():

        description = str(row["Description"])
        adoption_speed = row['AdoptionSpeed']
        pet_id = row['PetID']

        age = row['Age']
        breed = row['Breed1']
        gender = row['Gender']
        mat_size = row['MaturitySize']
        fur = row['FurLength']

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

        a = a.append({'Age': age}, ignore_index=True)
        b = b.append({'Breed1': breed}, ignore_index=True)
        g = g.append({'Gender': gender}, ignore_index=True)
        ms = ms.append({'MaturitySize': mat_size}, ignore_index=True)
        fl = fl.append({'FurLength': fur}, ignore_index=True)

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(d['Description'])

    s = s['Score']
    m = m['Magnitude']
    y = y['AdoptionSpeed']
    a = a['Age']
    b = b['Breed1']
    g = g['Gender']
    ms = ms['MaturitySize']
    fl = fl['FurLength']

    # lsa_analysis(tfidf_matrix)
    lsa_model = TruncatedSVD(n_components=12500)  # 12500 dobijeno analizom varijansi u fji iznad
    lsa = lsa_model.fit_transform(tfidf_matrix)

    df_other_all = pd.concat([pd.DataFrame(lsa), s, m, a, b, g, ms, fl], axis=1)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_other_all, y.astype('int'))

    svm = SVC()
    svm.fit(X_train, y_train)

    evaluate_model(svm, X_val, y_val, X_test, y_test)

    # accuracy = svm.score(X_test, y_test)
    # print(accuracy)
