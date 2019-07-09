import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
def CNN():
    cnn_model = tf.keras.models.Sequential([
        keras.layers.Conv2D(filters=16, kernel_size=(5, 5), 
                            padding='same',input_shape=(28, 28, 1),
                            activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=36, kernel_size=(5, 5), 
                            padding='same',input_shape=(28, 28, 1),
                            activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])

    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
    #cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

    return cnn_model

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:10000]
test_labels = test_labels[:1000]

train_images = train_images[:10000].reshape(-1, 28, 28, 1).astype('float32') / 255.0#reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28, 28, 1).astype('float32') / 255.0#reshape(-1, 28 * 28) / 255.0

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "train/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=0, save_weights_only=True,
    save_freq=5)

model = CNN()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
          epochs = 10, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
         # validation_split=0.2, 
          batch_size=300, verbose=1)  


latest = tf.train.latest_checkpoint(checkpoint_dir)
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')


# Save entire model to a HDF5 file
model.save('my_model.h5')
# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


