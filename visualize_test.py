from tensorflow import keras
import tensorflow as tf
# We preprocess the image into a 4D tensor
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os
import CNN_model
import visualize
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils

#from sklearn.metrics import confusion_matrix
import pandas as pd
#pd.set_option('display.float_format', lambda x: '%.3f' % x)
import seaborn as sn
#from matplotlib.ticker import funcformatter
#np.set_printoptions(suppress=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'


#genres=['Ambient', 'Blues', 'Chill Out','Cinematic','Classical',
#'Dance','Drum And Bass','Dubstep','Electro','Ethnic','Funk', 
#'Pop','Rap','Techno','Weird']
 
genres=['Blues','Dance','Drum And Bass','Electro','Funk']
size=200

def load_data(inf=''):
    data = []
    names = [n.strip() for n in open(os.path.join(inf, 'filename.txt'),'r').readlines()]
    data.append(np.array(names))
    data.append(np.load(os.path.join(inf, 'spec.npy')))
    lbls = [n.strip() for n in open(os.path.join(inf, 'label.txt'),'r').readlines()]
    data.append(np.array(lbls))
    return data

version=input('enter ver.:')
ep = int(input('load lastest checkpoint epoch(0 for final_checkpoint):'))
checkpoint_path = version+"_train/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir) if ep > 0 else 'history/{}/{}_checkpoints/my_checkpoint'.format(version,version)
#latest =  "train_"+version+"/cp-{04d}.ckpt".format(ep)#version+"_train/cp-{04d}.ckpt".format(ep)
'''
m=keras.models.load_model('history/'+version+'/'+version+'_loop_genres_classifier.h5')
'''
m = CNN_model.LetNet(size, size, 3, len(genres))
m.summary()
m.load_weights(latest)
print('load done')
size=200
'''
img_path = 'spectrogram_000556.png'

img = image.load_img(img_path, target_size=(size, size))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)
layer_names = []
for layer in m.layers: layer_names.append(layer.name)

layer_outputs = [layer.output for layer in m.layers]
activation_model = Model(inputs=m.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor.reshape(1,size,size,3))

#new_display(layer_names, activations)
'''

'''
plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.show()
plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
plt.show()
'''
#display_activation(activations, 6, 6, 0) # 0 = layer1
#display_activation(activations, 6, 6, 1) # 1 = layer2
#display_activation(activations, 8, 8, 6) # 6 = layer5

#x_file, x_img, x_label = load_data()
#test = load_data('3rd_data')
names, imgs, labels = load_data()
train_names, test_names, train_images, test_images, train_labels, test_labels = train_test_split(names, imgs, labels, test_size=0.3, random_state=42)

# One hot encoding
train_labels = np_utils.to_categorical(train_labels, num_classes=len(genres))
test_labels = np_utils.to_categorical(test_labels, num_classes=len(genres))

loss, acc = m.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print('==================')
visualize.plot_model(m, to_file='history/{}/{}_model.png'.format(version,version), show_shapes=True, show_layer_names=True)

print('Plot')
#plot_confusion_matrix(np.array([np.argmax(y) for y in m.predict(test_images)]), test_labels, classes=genres)
visualize.plot_confusion_matrix(m.predict(test_images), test_labels, classes=genres,version=version)


import random
new_model = m
ind = random.randint(0, len(test_images))
img = test_images[ind]
print('Test_image:'+ test_names[ind])
layer_outputs = [layer.output for layer in new_model.layers]
activation_model = keras.models.Model(inputs=new_model.input, outputs=layer_outputs)
activations = activation_model.predict(img.reshape(1,size,size,3))

print('--- Plot Feature Map')
layer_names = [layer.name for layer in new_model.layers]
visualize.tiled_save_activations(layer_names, activations, version=version)
