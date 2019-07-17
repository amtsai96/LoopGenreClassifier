import numpy as np
import matplotlib.pyplot as plt
import os
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import pandas as pd
import seaborn as sn
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

def save_activations(activations, col_size, row_size, act_index, name, img_name, version): 
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

def tiled_save_activations(layer_names, activations, version, images_per_row = 16):
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

def plot_confusion_matrix(y_actu, y_pred, classes, version, title='Confusion_Matrix'):
    y_actu = pd.Series([classes[int(np.argmax(y))] for y in y_actu], name='Actual')
    y_pred = pd.Series([classes[int(np.argmax(y))] for y in y_pred], name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, margins=True)

    plt.figure(figsize = (10,7))
    plt.title(version+' '+title)
    cmap = sn.cubehelix_palette(light=1, as_cmap=True)
    sn.set(font_scale=0.9)
    sn.heatmap(df_confusion, cmap=cmap, annot=True, fmt='d',annot_kws={"size": 14})

    #plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.colorbar()
    #tick_marks = np.arange(len(df_confusion.columns))
    #plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    #plt.yticks(tick_marks, df_confusion.index)
    #plt.ylabel(df_confusion.index.name)
    #plt.xlabel(df_confusion.columns.name)
    plt.savefig(version+'_'+title+'.png', bbox_inches='tight', pad_inches=0.0)
    #plt.show()
    plt.close()

