import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import seaborn as sn
import CNN_model
import visualize

#from sklearn.metrics import confusion_matrix
import pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

# genres = ['8Bit Chiptune','Classical','Dance','Dancehall','Deep House',
# 'Drum And Bass','Ethnic','Funk','Fusion','Glitch',
# 'Hardcore','Hardstyle','Heavy Metal', 'Reggae', 
# 'Techno', 'Trip Hop','Weird']

#genres=['Ambient', 'Blues', 'Chill Out','Cinematic','Classical',
#'Dance','Drum And Bass','Dubstep','Electro','Ethnic','Funk', 
#'Pop','Rap','Techno','Weird']
#print(len(genres))#15
genres=['Blues','Dance','Drum And Bass','Electro','Funk']

version = input('Enter Version Name:')
size = 200
batch_size = 32
epochs = 30

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

names, imgs, labels = load_data()
train_names, test_names, train_images, test_images, train_labels, test_labels = train_test_split(names, imgs, labels, test_size=0.3, random_state=42, stratify=labels)

train_images = train_images.reshape(-1, size, size, 3).astype('float32') / 255.0
test_images = test_images.reshape(-1, size, size, 3).astype('float32') / 255.0 

# One hot encoding
train_labels = np_utils.to_categorical(train_labels, num_classes=len(genres))
test_labels = np_utils.to_categorical(test_labels, num_classes=len(genres))

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = version+"_train/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=0, save_weights_only=True, period=5)

model = CNN_model.LetNet(size, size, 3, len(genres))
model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(train_images, train_labels,
          epochs = epochs, callbacks = [cp_callback],
          #validation_data = (train_images, train_labels),
          validation_data = (test_images,test_labels),
          #validation_split=0.2, 
          shuffle=True,
          batch_size=batch_size, verbose=1)  


#latest = tf.train.latest_checkpoint(checkpoint_dir)
# Save the weights
weight_dir = '{}_checkpoints/'.format(version)
weight_file = os.path.join(weight_dir, 'my_checkpoint')
if not os.path.exists(weight_dir): os.makedirs(weight_dir)
model.save_weights(weight_file)
# Save entire model to a HDF5 file
model_name = version+'_loop_genres_classifier.h5'
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
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('{}_acc.png'.format(version), bbox_inches='tight', pad_inches=0.0)
plt.close()
#plt.show()

# Load Saved Weights
new_model = CNN_model.LetNet(size, size, 3, len(genres))
new_model.summary()
new_model.load_weights(weight_file)
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print('==================')

'''
model_name = './history/'+version+'/loop_genres_classifier.h5'
new_model = keras.models.load_model(model_name)
new_model.summary()
new_model.compile(optimizer='adam',
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# # Recreate the exact same model, including weights and optimizer.
# new_model = keras.models.load_model(model_name)
# new_model.summary()
# new_model.compile(optimizer='adam',
#                 loss=tf.keras.losses.sparse_categorical_crossentropy,
#                 metrics=['accuracy'])
# loss, acc = new_model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# print("...........")               
'''
### Visualize

####### Plot Confusion Matrix
#y_pred = np.array([np.argmax(y) for y in new_model.predict(test_images)])
y_pred = new_model.predict(test_images)
visualize.plot_confusion_matrix(y_pred, test_labels, classes=genres, version=version)

####### Plot Feature Maps
ind = random.randint(0, len(test_images))
img = test_images[ind]
print('Test_image:'+ test_names[ind])
layer_outputs = [layer.output for layer in new_model.layers]
activation_model = keras.models.Model(inputs=new_model.input, outputs=layer_outputs)
activations = activation_model.predict(img.reshape(1,size,size,3))
'''
for i in range(len(activations)):
    if new_model.layers[i].name.startswith('flatten') or new_model.layers[i].name.startswith('dense'):
        continue
    save_activations(activations, 8, 8, i, new_model.layers[i].name, test_names[ind])
'''
print('--- Plot Feature Map')
layer_names = [layer.name for layer in new_model.layers]
visualize.tiled_save_activations(layer_names, activations, version=version)

####### Plot model structure
print('--- Plot model structure')
visualize.plot_model(new_model, to_file='{}_model.png'.format(version), show_shapes=True, show_layer_names=True)

