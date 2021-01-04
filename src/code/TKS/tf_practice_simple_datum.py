import tensorflow as tf
import numpy as np
import cv2
import DBM

#from tensorflow.keras import layers




db = DBM.DB_man()
x_train, y_train, x_test, y_test = db.load()

print(len(x_test))

x_train, y_train = shuffle_in_unison(x_train, y_train)


#build the model by stacking multiple keras layers
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(48, 64, 3)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(26)
])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


# fit the model to minimize the loss
model.fit(x_train, y_train, epochs=100)

model.evaluate(x_test,  y_test, verbose=2)
print(len(x_train))
print(len(x_test))