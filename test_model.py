import tensorflow as tf
import numpy as np
import os
import keras

testing_vecs = np.load('testing_vecs.npy')

image_vec = []
age_label = []

for img in testing_vecs:
    image_vec.append(img[0].astype(float) / 255.0)
    age = img[1]
    if age < 5 :
        age_label.append(0)
    elif age < 10:
        age_label.append(1)
    elif age < 15:
        age_label.append(2)
    elif age < 20:
        age_label.append(3)
    elif age < 30:
        age_label.append(4)
    elif age < 40:
        age_label.append(5)
    elif age < 50:
        age_label.append(6)
    elif age < 70:
        age_label.append(7)
    else:
        age_label.append(8)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(128,128)),
    keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.003), activation=tf.nn.relu),
    keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.003), activation=tf.nn.relu),
    keras.layers.Dense(9, activation=tf.nn.softmax)])

model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.load_weights('age_detection_model.ckpt')

predictions = model.predict(np.array(image_vec))

#aggregate difference between pred and expected
aggregate = 0

for i in range(0,len(predictions)):
    aggregate = aggregate + np.argmax(predictions[i]) - age_label[i]

print('Average error: ', aggregate/len(predictions))
