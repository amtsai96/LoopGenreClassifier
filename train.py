import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
#from sklearn.metrics import confusion_matrix
import pandas as pd
from keras.utils.vis_utils import plot_model
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
epochs = 100
def CNN():
    genres_num = len(genres)
    cnn_model = keras.models.Sequential([
    
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), 
                            padding='same',
                            input_shape=(size, size, 3),
                            activation='relu'),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), 
                            activation='relu'),          
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), 
                            padding='same',
                            activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), 
                            activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), 
                            padding='same',
                            activation='relu'),
        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), 
                            activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
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

names, imgs, labels = load_data()
train_names, test_names, train_images, test_images, train_labels, test_labels = train_test_split(names, imgs, labels, test_size=0.3, random_state=44)

train_images = train_images.reshape(-1, size, size, 3).astype('float32') / 255.0
test_images = test_images.reshape(-1, size, size, 3).astype('float32') / 255.0 


# include the epoch in the file name. (uses `str.format`)
checkpoint_path = version+"_train/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=0, save_weights_only=True, period=5)

model = CNN()
model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(train_images, train_labels,
          epochs = epochs, callbacks = [cp_callback],
          #validation_data = (train_images, train_labels),
          validation_data = (test_images,test_labels),
          #validation_split=0.2, 
          batch_size=batch_size, verbose=1)  


#latest = tf.train.latest_checkpoint(checkpoint_dir)
# Save the weights
weight_file = './{}checkpoints/my_checkpoint'.format(version)
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
new_model = CNN()
new_model.summary()
new_model.load_weights(weight_file)
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print('==================')
plot_model(new_model, to_file='{}_model.png'.format(version), show_shapes=True, show_layer_names=True)

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
def save_activations(activations, col_size, row_size, act_index, name, img_name): 
    activation = activations[act_index]
    print(name, activation.shape)
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            if activation_index < activation.shape[-1] and len(activation.shape) == 4:
                ax[row][col].imshow(activation[0, :, :, activation_index])
            else:
                ax[row][col].axis('off')
            activation_index += 1
    fig.suptitle('{} -- layer{}:{} with img:{}'.format(version, act_index+1, name, img_name))
    plt.savefig('{}_layer{}_{}.png'.format(version, act_index+1, name), bbox_inches='tight', pad_inches=0.0)
    plt.close()

def tiled_save_activations(layer_names, activations, images_per_row = 16):
    i=0
    for layer_name, layer_activation in zip(layer_names, activations):
        i+=1
        if len(layer_activation.shape) < 4: continue
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]
        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]
        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                            row * size : (row + 1) * size] = channel_image

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title('{} -- layer_{}: {}'.format(version, i, layer_name))
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.savefig('{}_layer{}_{}.png'.format(version, i, layer_name), bbox_inches='tight', pad_inches=0.0)
    plt.close()

def plot_confusion_matrix(y_actu, y_pred, classes, title='Confusion_Matrix', cmap=plt.get_cmap('nipy_spectral')):
    #confusion_matrix(y_actu, y_pred)
    #if normalize: title = 'Normalize_'+title
    y_actu = pd.Series([classes[int(y)] for y in y_actu], name='Actual')
    y_pred = pd.Series([classes[int(y)] for y in y_pred], name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, margins=True)

    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.savefig(title+'.png', bbox_inches='tight', pad_inches=0.0)
    #plt.show()
    plt.close()

####### Plot Confusion Matrix
y_pred = np.array([np.argmax(y) for y in new_model.predict(test_images)])
plot_confusion_matrix(y_pred, test_labels, classes=genres)

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
layer_names = [layer.name for layer in new_model.layers]
tiled_save_activations(layer_names, activations)


