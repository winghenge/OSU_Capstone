import tensorflow as tf
import numpy as np
import cv2
import DBM


# from tensorflow.keras import layers


db = DBM.DB_man()
x_train, y_train, x_test, y_test = db.load()

x_train = [[[[1-x_train[i][j][k][l] for l in range(3)] for k in range(64)] for j in range(48)] for i in range(len(x_train))]
x_test = [[[[1-x_test[i][j][k][l] for l in range(3)] for k in range(64)] for j in range(48)] for i in range(len(x_test))]

x_train = np.array(x_train)
x_test = np.array(x_test)

#build the model by stacking multiple keras layers
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(48, 64, 3)),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(26)
])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


# fit the model to minimize the loss
model.fit(x_train, y_train,
          epochs=128,
          shuffle=True,
          validation_data=(x_test, y_test))

model.evaluate(x_test,  y_test, verbose=2)

predictions = model.predict(x_test)

print(np.argmax(predictions[0]))
print(y_test[0])