import numpy.random
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorboard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, cv2, random, datetime
from tqdm import tqdm
import pickle

'''load files'''
pickle_in = open('test_feat', 'rb')
test_features = pickle.load(pickle_in)

pickle_in2 = open('test_label', 'rb')
test_labels = pickle.load(pickle_in2)


# '''model update'''
# MODEL_DIR = './model/'
# if not os.path.exists(MODEL_DIR):
# 	os.mkdir(MODEL_DIR)
#
# modelpath = "./model/{epoch:02d}-{val_loss:.4f}.h5"

# checkpointer = ModelCheckpoint(filepath=modelpath,
#                                monitor='val_loss',
#                                verbose=1,
#                                save_best_only=True)

'''normalize'''
test_features = (test_features / 255) - 0.5
print('Test Feature Shape', test_features.shape)
print('Length of Labels:', len(test_labels))

num_filters = 8
filter_size = 3
pool_size = 2

# docs = ['mask_weared_incorrect', 'with_mask', 'without_mask']
# labels = np.array([0,1,2])
# vocab_size = 50
# encoded_docs = [tf.one_hot(x, vocab_size) for x in docs]

'''model init and layers'''
model = Sequential([
	Conv2D(num_filters, filter_size, input_shape=(128, 128, 1)),  # input layer
	MaxPooling2D(pool_size=pool_size),
	Dropout(0.5),
	Flatten(),
	Dense(4, activation='relu'),
	Dense(3, activation='softmax')  # output layer
])

'''model compile'''
optim = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optim,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

'''logging and graphing'''
log_dir = "logs\\fit\\"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)

model.fit(test_features, to_categorical(test_labels),
          epochs=10,
          batch_size=10,
          callbacks=[tensorboard_callback])

'''save model'''
model.save('cnn_full_model_facemask')

# model.add(Conv2D(256, (3, 3), input_shape=test_features.shape[0]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
#
# model.add(Conv2D(256, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Flatten())
# model.add(Dense)
