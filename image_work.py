import os
import cv2
import numpy as np
from keras.applications import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import preprocess_input
import pickle

pickle_file = 'images_features.pkl'
filename_for_model = 'my_model.h5'

# model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = VGG16(weights='imagenet', include_top=False)


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
                image_path_dict[row_id].append(image_path)  # ako vec ima slika za tu zivotinju

    return image_path_dict


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (112, 112))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)
    return image


def extract_features(image_path):
    image = load_image(image_path)
    features = model.predict(np.array([image]))
    features = features.flatten()
    return features


def get_all_features(images_folder):
    default_vector = np.zeros((2048,), dtype=np.float32)

    image_features_dict = {}

    image_path_dict = load_all_images(images_folder)

    for row_id, image_paths in image_path_dict.items():
        features_list = []
        for image_path in image_paths:
            features = extract_features(image_path)
            features_list.append(features)
        if len(features_list) > 0:
            image_features_dict[row_id] = np.mean(features_list, axis=0)
        else:
            image_features_dict[row_id] = default_vector

    return image_features_dict


def main_images(images_folder):
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            image_features_dict =  pickle.load(f)
    else:
        image_features_dict = get_all_features(images_folder)
        with open(pickle_file, 'wb') as f:
            pickle.dump(image_features_dict, f)

    return image_features_dict

