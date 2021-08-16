import tensorflow as tf
import cv2 as cv
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorboard import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, random, datetime, glob, cv2
from PIL import Image
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array

print(tf.keras.__version__)


# # Get Images from folder and convert to GRAYSCALE, 128X128
# files = glob.glob(PATH)
# for f1 in files:
# 	img = cv2.resize(cv2.imread(f1, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
# 	testing_data.append(img)
#
# testing_data = np.array(testing_data)
# testing_data = testing_data.reshape((-1, IMG_SIZE, IMG_SIZE, 1))
#
# # Check to see if data went through
# print('Testing Data Shape: ', np.shape(testing_data))
# print(testing_data)

# model = tf.keras.models.load_model('cnn_full_model_facemask')
#
# predictions = model.predict(testing_data[:])
# answers = (np.argmax(predictions, axis=1))
# for i in answers:
# 	if i == 0:
# 		print('Person is wearing the mask in a wrong way')
# 	if i == 1:
# 		print('Person is wearing a mask')
# 	if i == 2:
# 		print('Person is not wearing a mask')
#
# from matplotlib import pyplot
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
#
# labels = {0: 'Half Mask', 1: 'Mask', 2: 'No Mask'}
#
# for i in range(len(testing_data)):
# 	subplot = fig.add_subplot(3, 5, i + 1)  #row, col, index
# 	subplot.set_xticks([])
# 	subplot.set_yticks([])
# 	subplot.set_title(f'{labels[answers[i]]}')
# 	subplot.imshow((testing_data[i]), cmap='gray')
#
# plt.show()





# def draw_border(img, pt1, pt2, color, thickness, r, d):
# 	x1,y1 = pt1
# 	x2,y2 = pt2
#
# 	# Top left
# 	cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
# 	cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
# 	cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
#
# 	# Top right
# 	cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
# 	cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
# 	cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
#
# 	# Bottom left
# 	cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
# 	cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
# 	cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
#
# 	# Bottom right
# 	cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
# 	cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
# 	cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

# ============================================================================

# img = np.zeros((256,256,3), dtype=np.uint8)

model = tf.keras.models.load_model('model.savedmodel', compile=False)

'''Start Video Capture'''
capture = cv.VideoCapture(0)

if capture.isOpened() == False:
	print('No Source')

while True:
	detector = cv2.CascadeClassifier('./haar-cascade-files-master/haarcascade_frontalface_alt2.xml')
	ret, frame = capture.read()

	# '''Brightness Adjusting'''
	# cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)

	'''To Gray'''
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_2 = frame.copy()

	'''Face Detection Cascade'''
	faces = detector.detectMultiScale(gray, 1.1, 4)

	'''Globals'''
	color = (127,255,255)
	thickness = 1
	r = 10
	d = 20
	font = cv2.FONT_HERSHEY_SIMPLEX
	IMG_SIZE = 224


	'''Prediction and Face Rectangle Drawing'''
	for (x, y, w, h) in faces:
		# testing_frame_roi = frame[x:x+w, y+y+h]
		# testing_frame_roi = gray[x:x+w, y+y+h]
		# testing_frame_roi = np.array(gray)
		# _roi = frame[x: x + w, y: y + h]
		# print(gray_roi)

		# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

		testing_frame_roi = cv2.resize(frame_2, (IMG_SIZE, IMG_SIZE))  #Resize
		testing_frame_roi = (testing_frame_roi / 224.0) - 0.5 #NORMALIZE
		testing_frame_roi = testing_frame_roi.reshape((-1, IMG_SIZE, IMG_SIZE, 3))  #Make Rank4 Tensor

		prediction = model.predict(testing_frame_roi[:])
		prediction_list = prediction.flatten()
		# print(prediction_list)
		max_index = np.argmax(prediction, axis=1)

		dict_ = {0: 'Mask', 1: 'No Mask', 2: 'Partial Mask'}

		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
		cv2.putText(frame, f'{dict_[max_index[0]]}', (x + w, y + h), font, 1, (200, 255, 155))


	cv.imshow('Frame', frame)

	if cv.waitKey(1) == ord('q'):
		break






# CHECK IF IMAGES EXIST
# use_samples = list(range(0,14))
# for sample use_samples:
#   # Generate a plot
#   reshaped_image = testing_data[sample].reshape((IMG_SIZE, IMG_SIZE))
#   plt.imshow(reshaped_image, cmap='gray')
#   plt.show()
