import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#import visualize
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

# genres = ['8Bit Chiptune','Classical','Dance','Dancehall','Deep House',
# 'Drum And Bass','Ethnic','Funk','Fusion','Glitch',
# 'Hardcore','Hardstyle','Heavy Metal', 'Reggae', 
# 'Techno', 'Trip Hop','Weird']

genres=['Ambient', 'Blues', 'Chill Out','Cinematic','Classical',
'Dance','Drum And Bass','Dubstep','Electro','Ethnic','Funk', 
'Pop','Rap','Techno','Weird']
#print(len(genres))#15


version = input('Enter Version Name:')
size = 200
batch_size = 64
epochs = 100
def CNN():
    genres_num = len(genres)
    cnn_model = keras.models.Sequential([
    
        keras.layers.Conv2D(filters=32, kernel_size=(5, 5), 
                            input_shape=(size, size, 3),
                            activation='relu'),
        keras.layers.Conv2D(filters=32, kernel_size=(1, 1), 
                            activation='relu'),          
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), 
                            activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=(1, 1), 
                            activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), 
                            activation='relu'),
        keras.layers.Conv2D(filters=128, kernel_size=(1, 1), 
                            activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(genres_num, activation='softmax')
    ])

    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(),#tf.train.AdamOptimizer(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

    return cnn_model

def load_data(inf=''):
    data = []
    names = [n.strip() for n in open(os.path.join(inf, 'filename.txt'),'r').readlines()]
    data.append(np.array(names))
    data.append(np.load(os.path.join(inf, 'spec.npy')))
    lbls = [n.strip() for n in open(os.path.join(inf, 'label.txt'),'r').readlines()]
    data.append(np.array(lbls))
    return data

# def _load_data(inf=''):
#     dataset = []
#     for m in ['train', 'test']:
#         data = []
#         names = [n.strip() for n in open(os.path.join(inf, 'filename_'+m+'.txt'),'r').readlines()]
#         data.append(np.array(names))
#         data.append(np.load(os.path.join(inf, 'spec_'+m+'.npy')))
#         lbls = [n.strip() for n in open(os.path.join(inf, 'label_'+m+'.txt'),'r').readlines()]
#         data.append(np.array(lbls))
#         dataset.append(data)
#     return dataset[0], dataset[1]

# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
#(train_name, train_images, train_labels), (test_name, test_images, test_labels) = _load_data()
#print(train_name.shape, train_images.shape, train_labels.shape)
#print(test_name.shape, test_images.shape, test_labels.shape)

names, imgs, labels = load_data()
train_names, test_names, train_images, test_images, train_labels, test_labels = train_test_split(names, imgs, labels, test_size=0.3, random_state=44)

train_images = train_images.reshape(-1, size, size, 3).astype('float32') / 255.0
test_images = test_images.reshape(-1, size, size, 3).astype('float32') / 255.0 

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "train/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=0, save_weights_only=True, period=5)

model = CNN()
model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(train_images, train_labels,
          epochs = epochs, callbacks = [cp_callback],
          #validation_data = (test_images,test_labels),
          validation_split=0.2, 
          batch_size=batch_size, verbose=1)  


#latest = tf.train.latest_checkpoint(checkpoint_dir)
# Save the weights
weight_file = './checkpoints/my_checkpoint'
model.save_weights(weight_file)
# Save entire model to a HDF5 file
model_name = 'loop_genres_classifier.h5'
model.save(model_name)
print('Model saved.')
print('------------------------')

# Plot
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('{}_loss.png'.format(version), bbox_inches='tight', pad_inches=0.0)
plt.close()
#plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('{}_acc.png'.format(version), bbox_inches='tight', pad_inches=0.0)
plt.close()
#plt.show()

# Load Saved Weights
new_model = CNN()
new_model.load_weights(weight_file)
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print('==================')

# # Recreate the exact same model, including weights and optimizer.
# new_model = keras.models.load_model(model_name)
# new_model.summary()
# new_model.compile(optimizer='adam',
#                 loss=tf.keras.losses.sparse_categorical_crossentropy,
#                 metrics=['accuracy'])
# test_images, test_labels = 
# loss, acc = new_model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print("...........")                

#visualize.visualize_layer(new_model, 'conv2d_1')

