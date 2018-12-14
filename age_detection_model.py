import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import os

checkpoint_path = 'age_detection_model.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=100)
training_vecs = np.load('training_vecs.npy')

age_label = []
image_vec = []

#labels:   
#0-5:0
#5-10:1
#10-15: 2
#15-20: 3
#20-30: 4
#30-40: 5
#40-50: 6
#50-70: 7
#70+ : 8

#normalize values, add to image vecs
for img in training_vecs:
    image_vec.append(img[0].astype(float) / 255.0 )
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

model.fit(np.array(image_vec), np.array(age_label), batch_size=256, epochs=100, callbacks=[cp_callback])

