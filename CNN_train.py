import tensorflow as tf
#from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model
import seaborn as sn
import CNN_model
import visualize
import pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

version = input('Enter Version Name:')
width, height, depth = 640, 480, 3
batch_size = 32
epochs = 30

# 64 genres
genres=['8Bit Chiptune', 'Acid', 'Acoustic', 'Ambient', 'Big Room', 'Blues', 'Boom Bap', 'Breakbeat', 'Chill Out', 
    'Cinematic', 'Classical', 'Comedy', 'Country', 'Crunk', 'Dance', 'Dancehall', 'Deep House', 'Dirty', 'Disco', 'Drum And Bass',
    'Dub', 'Dubstep', 'EDM', 'Electro', 'Electronic', 'Ethnic', 'Folk', 'Funk', 'Fusion', 'Garage', 'Glitch', 'Grime', 'Grunge', 
    'Hardcore', 'Hardstyle', 'Heavy Metal', 'Hip Hop', 'House', 'Indie', 'Industrial', 'Jazz', 'Jungle', 'Lo-Fi', 'Moombahton', 
    'Orchestral', 'Pop', 'Psychedelic', 'Punk', 'Rap', 'Rave', 'Reggae', 'Reggaeton', 'Religious', 'RnB', 'Rock', 'Samba', 'Ska',
    'Soul', 'Spoken Word', 'Techno', 'Trance', 'Trap', 'Trip Hop', 'Weird']

# 40 cates
categories=['Accordion', 'Arpeggio', 'Bagpipe', 'Banjo', 'Bass', 'Bass Guitar', 'Bass Synth', 'Bass Wobble', 
    'Beatbox', 'Bells', 'Brass', 'Choir', 'Clarinet', 'Didgeridoo', 'Drum', 'Flute', 'Fx', 'Groove', 'Guitar Acoustic', 
    'Guitar Electric', 'Harmonica', 'Harp', 'Harpsichord', 'Mandolin', 'Orchestral', 'Organ', 'Pad', 'Percussion', 
    'Piano', 'Rhodes Piano', 'Scratch', 'Sitar', 'Soundscapes', 'Strings', 'Synth', 'Tabla', 'Ukulele', 'Violin', 'Vocal', 'Woodwind']

def load_data(inf=''):
    data = []
    names = [n.strip() for n in open(os.path.join(inf, 'filename.txt'),'r').readlines()]
    data.append(np.array(names))
    data.append(np.load(os.path.join(inf, 'spec.npy')))
    lbls = [n.strip() for n in open(os.path.join(inf, 'label.txt'),'r').readlines()]
    data.append(np.array(lbls))
    return data

names, imgs, labels = load_data()
train_names, test_names, train_images, test_images, train_labels, test_labels = train_test_split(names, imgs, labels, test_size=0.3, random_state=42, stratify=labels)

train_images = train_images.reshape(-1, width, height, depth).astype('float32') / 255.0
test_images = test_images.reshape(-1, width, height, depth).astype('float32') / 255.0 

# One hot encoding
train_labels = np_utils.to_categorical(train_labels, num_classes=len(genres))
test_labels = np_utils.to_categorical(test_labels, num_classes=len(genres))

# include the epoch in the file name. (uses `str.format`)
root_dir = os.path.join('history/', version)
if not os.path.exists(root_dir): os.makedirs(root_dir)

checkpoint_dir = os.path.join(root_dir, version + '_train') #os.path.dirname(checkpoint_path)
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt')


cp_callback = [tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=0, save_weights_only=True, period=5),
    tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_acc")]

model, model_archi = CNN_model.CNN(width, height, depth, len(genres))
model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(train_images, train_labels,
          epochs = epochs, callbacks = cp_callback,
          validation_data = (test_images,test_labels),
          #validation_split=0.2, 
          shuffle=True,
          batch_size=batch_size, verbose=1)  


#latest = tf.train.latest_checkpoint(checkpoint_dir)
# Save the weights
weight_dir = os.path.join(root_dir, '{}_checkpoints/'.format(version))
if not os.path.exists(weight_dir): os.makedirs(weight_dir)
weight_file = os.path.join(weight_dir, 'my_checkpoint')
model.save_weights(weight_file)
# Save entire model to a HDF5 file
model_name = os.path.join(root_dir, version+'_loop_genres_classifier.h5')
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
plt.savefig(os.path.join(root_dir, '{}_{}_loss.png'.format(version, model_archi)), bbox_inches='tight', pad_inches=0.0)
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
plt.savefig(os.path.join(root_dir, '{}_{}_acc.png'.format(version, model_archi)), bbox_inches='tight', pad_inches=0.0)
plt.close()
#plt.show()

# Load Saved Weights
new_model = CNN_model.CNN(width, height, depth, len(genres))
new_model.summary()
new_model.load_weights(weight_file)
loss, acc = new_model.evaluate(test_images, test_labels)
print("{} {}\tRestored model, accuracy: {:5.2f}%".format(version, model_archi, 100*acc))
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
print('--- Plot Confusion Matrix')
#y_pred = np.array([np.argmax(y) for y in new_model.predict(test_images)])
y_pred = new_model.predict(test_images)
visualize.plot_confusion_matrix(y_pred, test_labels, to_file=os.path.join(root_dir,'{}_{}_Confusion_Matrix.png'.format(version, model_archi)), classes=genres, version=version)

####### Plot model structure
print('--- Plot model structure')
visualize.plot_model(new_model, to_file=os.path.join(root_dir,'{}_{}_model.png'.format(version, model_archi)), show_shapes=True, show_layer_names=True)
'''
####### Plot Feature Maps
ind = random.randint(0, len(test_images))
img = test_images[ind]
print('Test_image:'+ test_names[ind])
layer_outputs = [layer.output for layer in new_model.layers]
activation_model = Model(inputs=new_model.input, outputs=layer_outputs)
activations = activation_model.predict(img.reshape(1, width, height, depth))

#for i in range(len(activations)):
#    if new_model.layers[i].name.startswith('flatten') or new_model.layers[i].name.startswith('dense'):
#        continue
#    save_activations(activations, 8, 8, i, new_model.layers[i].name, test_names[ind])

print('--- Plot Feature Map')
layer_names = [layer.name for layer in new_model.layers]
visualize.tiled_save_activations(layer_names, activations, version=version, save_dir=root_dir)
#visualize.save_activations(layer_names, activations, version=version, save_dir=root_dir)
'''
