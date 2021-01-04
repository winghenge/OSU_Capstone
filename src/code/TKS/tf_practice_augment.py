import tensorflow as tf
import numpy as np
import cv2
import DBM

from tensorflow.keras import layers

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


db = DBM.DB_man()
x_train, y_train, x_test, y_test = db.load()

print(len(x_test))

x_train, y_train = shuffle_in_unison(x_train, y_train)


#build the model by stacking multiple keras layers
model = tf.keras.models.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  #layers.experimental.preprocessing.RandomRotation(0.2),
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