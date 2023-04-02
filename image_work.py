import os
import cv2
import numpy as np
import pandas as pd
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.python.estimator import keras

filename = 'my_model.h5'

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def load_all_images(images_path):
    image_path_dict = {}

    for filename in os.listdir(images_path):
        if filename.endswith('.jpg'):
            try:
                row_id, image_num = filename[:-4].split('-')
            except ValueError:
                continue
            image_path = os.path.join(images_path, filename)
            if row_id not in image_path_dict:
                image_path_dict[row_id] = [image_path]
            else:
                image_path_dict[row_id].append(image_path)  #ako vec ima slika za tu zivotinju

    return image_path_dict

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)
    return image
def extract_features(image_path):
    image = load_image(image_path)
    features = model.predict(np.array([image]))
    features = features.flatten()
    return features
def main_images(images_folder):
    image_features_dict = {}

    image_path_dict = load_all_images(images_folder)

    for row_id, image_paths in image_path_dict.items():
        features_list = []
        for image_path in image_paths:
            features = extract_features(image_path)
            features_list.append(features)
        image_features_dict[row_id] = np.mean(features_list, axis=0)

    return image_features_dict

# Load tabular data
def load_tabular_data(data_path):
    data = pd.read_csv(data_path)
    data['PetID'] = data['PetID'].astype(str)
    data = data.set_index('PetID')
    return data

# Combine image and tabular data
def combine_data(image_data, tabular_data):
    combined_data = pd.concat([pd.DataFrame.from_dict(image_data, orient='index'), tabular_data], axis=1)
    return combined_data

# Build and train model
def build_model(input_shape):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def split_data(data, test_size=0.1, val_size=0.2, random_state=42):
    data = data.values.tolist()
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(combined_data, X_train, y_train, X_val, y_val):
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        return keras.models.load_model(filename)
    else:
        X = combined_data.drop('AdoptionSpeed', axis=1)
        y = pd.get_dummies(combined_data['AdoptionSpeed'])
        input_shape = (X_train.shape[1],)
        model = build_model(input_shape)
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
        model.save(filename)
        return model

# Main function
def train_images(data_path, images_folder):
    # Load image data
    image_data = main_images(images_folder)

    # Load tabular data
    tabular_data = load_tabular_data(data_path)

    # Combine data
    combined_data = combine_data(image_data, tabular_data)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(combined_data)

    # Train model
    model = train_model(combined_data, X_train, y_train, X_val, y_val)

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

    # Evaluate model
    X_test = combined_data.drop('AdoptionSpeed', axis=1)
    y_test = pd.get_dummies(combined_data['AdoptionSpeed'])
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
    print('Accuracy:', accuracy)