import tensorflow as tf
import cv2 as cv
from tensorboard import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, random, datetime, glob, cv2
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array

print(tf.keras.__version__)

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
