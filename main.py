import tensorflow as tf
import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#load image
file = input('Enter image file:')

image_vec = []
src = cv2.imread(os.getcwd() + '/' + file,0)
if src is not None:
    dst = cv2.resize(src, (128,128))
    image_vec.append(dst)

image_vec[0] = image_vec[0].astype(float) / 255.0

#load model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(128,128)),
    keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.003), activation=tf.nn.relu),
    keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.003), activation=tf.nn.relu),
    keras.layers.Dense(9, activation=tf.nn.softmax)])

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.load_weights('age_detection_model.ckpt')

#make prediction
predictions = model.predict(np.array(image_vec))

ageBracket = np.argmax(predictions[0])

if ageBracket == 0:
    print('0-5 years old')
if ageBracket == 1:
    print('5-10 years old')
if ageBracket == 2:
    print('10-15 years old')
if ageBracket == 3:
    print('15-20 years old')
if ageBracket == 4:
    print('20-30 years old')
if ageBracket == 5:
    print('30-40 years old')
if ageBracket == 6:
    print('40-50 years old')
if ageBracket == 7:
    print('50-70 years old')
if ageBracket == 8:
    print('>70 years old')

#draw bar graph
test_labels = ('0-5', '5-10', '10-15', '15-20', '20-30', '30-40', '40-50', '50-70', '70+')

y_pos = np.arange(len(test_labels))

plt.bar(y_pos, predictions[0], align='center', alpha=0.5)
plt.xticks(y_pos, test_labels)
plt.yticks([])
plt.title('Confidence value of predicted age bracket')
         
plt.show()

