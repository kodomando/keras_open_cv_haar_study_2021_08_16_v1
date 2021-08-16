import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os, cv2, random
from tqdm import tqdm
import pickle

DATADIR = './Dataset'
CATEGORIES = ['mask_weared_incorrect', 'with_mask', 'without_mask']
IMG_SIZE = 128
training_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR, category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

create_training_data()

random.shuffle(training_data)

for sample in training_data[:-1]:
    print(sample[1])

test_feat = []
test_label = []

for features, labels in training_data:
    test_feat.append(features)
    test_label.append(labels)

test_feat = np.array(test_feat)
print(test_feat.shape)
test_feat = test_feat.reshape((-1, IMG_SIZE, IMG_SIZE, 1))
print(test_feat.shape)

pickle_out = open('test_feat','wb')
pickle.dump(test_feat, pickle_out)
pickle_out.close()

pickle_out = open('test_label','wb')
pickle.dump(test_label, pickle_out)
pickle_out.close()