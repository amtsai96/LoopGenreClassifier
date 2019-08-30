import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, ZeroPadding2D, Flatten, Dropout, add, concatenate, Input, InputSpec
from tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

def CNN(width, height, depth, classes_num):
    return CNN_V4(width, height, depth, classes_num), CNN_V4.__name__
    #0722b#return ResNet50(width, height, depth, classes_num), ResNet50.__name__
    #0722a#return ResNet34(width, height, depth, classes_num), ResNet34.__name__
    #return newResNet50(width, height, depth, classes_num)
    ###return _ResNet50(width, height, depth, classes_num)
    #return ResNext50(width, height, depth, classes_num)
    #return CNN_V3(width, height, depth, classes_num)
    #0721c#return CNN_V2(width, height, depth, classes_num)
    ##return ResNet34(width, height, depth, classes_num)
    #return InceptionV1(width, height, depth, classes_num)
    #return VGG16Net(width, height, depth, classes_num)
    #return AlexNet(width, height, depth, classes_num)
    #return LeNet(width, height, depth, classes_num)
#################################################
def CNN_V4(width, height, depth, classes_num):
    inpt = Input(shape=(width,height,depth))
    x = BatchNormalization()(inpt)
    x = Conv2D(64, 2, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, 2, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, 2, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(512, 2, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dense(1024, activation='softmax')(x)
    x = Dropout(rate=0.5)(x)
    x = BatchNormalization()(Dense(128, activation='relu')(x))
    x = BatchNormalization()(Dense(128, activation='relu')(x))
    x = Dense(classes_num, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)

    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), 
                    loss='categorical_crossentropy', metrics=['accuracy'])
    return model
'''
def newResNet50(width, height, depth, classes_num):
    from tensorflow.python.keras.applications.resnet50 import ResNet50 as R
    net = R(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(width, height, depth))
    x = net.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(classes_num, activation='softmax', name='softmax')(x)
    model = Model(inputs=net.input, outputs=output_layer)
    model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])
    return model
'''
#Define Residual Block for ResNet50(3 convolution layers)
def Residual_Blocks(input_model,nb_filters,kernel_sizes=[(1,1),(3,3),(1,1)],strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(input_model,nb_filter=nb_filters[0],kernel_size=kernel_sizes[0],strides=strides)
    x = Activation('relu')(x)
    x = Conv2d_BN(x, nb_filter=nb_filters[1], kernel_size=kernel_sizes[1],padding='same')
    x = Activation('relu')(x)
    x = Conv2d_BN(x, nb_filter=nb_filters[2], kernel_size=kernel_sizes[2])
    #need convolution on shortcut for add different channel
    if with_conv_shortcut:
        shortcut = Conv2d_BN(input_model,nb_filter=nb_filters[2],kernel_size=kernel_sizes[2],strides=strides)
        x = add([x,shortcut])
    else: x = add([x,input_model])
    x = Activation('relu')(x)
    return x
        
def ResNet50(width, height, depth, classes_num):
    inpt = Input(shape=(width,height,depth))
    x = Conv2d_BN(inpt,64,(7,7),strides=(2,2),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)

    #Residual conv2_x ouput 56x56x256
    x = Residual_Blocks(x,nb_filters=[64,64,256],with_conv_shortcut=True)
    x = Residual_Blocks(x,nb_filters=[64,64,256])
    x = Residual_Blocks(x,nb_filters=[64,64,256])

    #Residual conv3_x ouput 28x28x512
    x = Residual_Blocks(x,nb_filters=[128,128,512],strides=(2,2),with_conv_shortcut=True)# need do convolution to add different channel
    x = Residual_Blocks(x,nb_filters=[128,128,512])
    x = Residual_Blocks(x,nb_filters=[128,128,512])
    x = Residual_Blocks(x,nb_filters=[128,128,512])

    #Residual conv4_x ouput 14x14x1024
    x = Residual_Blocks(x,nb_filters=[256,256,1024],strides=(2,2),with_conv_shortcut=True)# need do convolution to add different channel
    x = Residual_Blocks(x,nb_filters=[256,256,1024] )
    x = Residual_Blocks(x,nb_filters=[256,256,1024] )
    x = Residual_Blocks(x,nb_filters=[256,256,1024] )
    x = Residual_Blocks(x,nb_filters=[256,256,1024] )
    x = Residual_Blocks(x,nb_filters=[256,256,1024] )

    #Residual conv5_x ouput 7x7x2048
    x = Residual_Blocks(x,nb_filters=[512,512,2048] ,strides=(2,2),with_conv_shortcut=True)
    x = Residual_Blocks(x,nb_filters=[512,512,2048] )
    x = Residual_Blocks(x,nb_filters=[512,512,2048] )

    #Using AveragePooling replace flatten
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    #x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(classes_num,activation='softmax')(x)
    
    model=Model(inputs=inpt,outputs=x)
    model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])
    return model

def CNN_V2(width, height, depth, classes_num):
    inpt = Input(shape=(width,height,depth))
    x = BatchNormalization()(inpt)
    x = Conv2D(16, kernel_size=(3, 7), activation='relu')(x)
    x = Conv2D(16, kernel_size=(3, 7), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 7))(x)
    x = Dropout(rate=0.1)(x)
    x = Conv2D(32, kernel_size=3, activation='relu')(x)
    x = Conv2D(32, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(rate=0.1)(x)
    x = Conv2D(128, kernel_size=3, activation='relu')(x)
    x = GlobalMaxPooling2D()(x)
    x = Dropout(rate=0.1)(x)

    x = BatchNormalization()(Dense(128, activation='relu')(x))
    x = BatchNormalization()(Dense(128, activation='relu')(x))
    x = Dense(classes_num, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)

    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), 
                    loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#Define convolution with batchnormalization
def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x

#################################################
#Define Residual Block for ResNet34(2 convolution layers)
def Residual_Block(input_model,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(input_model,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Activation('relu')(x)
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    
    #need convolution on shortcut for add different channel
    if with_conv_shortcut:
        shortcut = Conv2d_BN(input_model,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides)
        x = add([x,shortcut])
    else: x = add([x,input_model])
    return x
    
def ResNet34(width, height, depth, classes_num):
    Img = Input(shape=(width,height,depth))
    
    x = Conv2d_BN(Img,64,(7,7),strides=(2,2),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)

    #Residual conv2_x ouput 56x56x64 
    x = Residual_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=64,kernel_size=(3,3))
    
    #Residual conv3_x ouput 28x28x128 
    x = Residual_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)# need do convolution to add different channel
    x = Residual_Block(x,nb_filter=128,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=128,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=128,kernel_size=(3,3))
    
    #Residual conv4_x ouput 14x14x256
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)# need do convolution to add different channel
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    
    #Residual conv5_x ouput 7x7x512
    x = Residual_Block(x,nb_filter=512,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Residual_Block(x,nb_filter=512,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=512,kernel_size=(3,3))

    #Using AveragePooling replace flatten
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes_num,activation='softmax')(x)
    
    model=Model(inputs=Img,outputs=x)
    model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])

    return model

#################################################
#Define Inception structure
def Inception(x,nb_filter_para):
    (branch1,branch2,branch3,branch4)= nb_filter_para
    branch1x1 = Conv2D(branch1[0],(1,1), padding='same',strides=(1,1),name=None)(x)

    branch3x3 = Conv2D(branch2[0],(1,1), padding='same',strides=(1,1),name=None)(x)
    branch3x3 = Conv2D(branch2[1],(3,3), padding='same',strides=(1,1),name=None)(branch3x3)

    branch5x5 = Conv2D(branch3[0],(1,1), padding='same',strides=(1,1),name=None)(x)
    branch5x5 = Conv2D(branch3[1],(1,1), padding='same',strides=(1,1),name=None)(branch5x5)

    branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    branchpool = Conv2D(branch4[0],(1,1),padding='same',strides=(1,1),name=None)(branchpool)

    x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)

    return x
  
#Build InceptionV1 model
def InceptionV1(width, height, depth, classes):
    inpt = Input(shape=(width,height,depth))

    x = Conv2d_BN(inpt,64,(7,7),strides=(2,2),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Conv2d_BN(x,192,(3,3),strides=(1,1),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)

    x = Inception(x,[(64,),(96,128),(16,32),(32,)]) #Inception 3a 28x28x256
    x = Inception(x,[(128,),(128,192),(32,96),(64,)]) #Inception 3b 28x28x480
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x) #14x14x480

    x = Inception(x,[(192,),(96,208),(16,48),(64,)]) #Inception 4a 14x14x512
    x = Inception(x,[(160,),(112,224),(24,64),(64,)]) #Inception 4a 14x14x512
    x = Inception(x,[(128,),(128,256),(24,64),(64,)]) #Inception 4a 14x14x512
    x = Inception(x,[(112,),(144,288),(32,64),(64,)]) #Inception 4a 14x14x528
    x = Inception(x,[(256,),(160,320),(32,128),(128,)]) #Inception 4a 14x14x832
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x) #7x7x832

    x = Inception(x,[(256,),(160,320),(32,128),(128,)]) #Inception 5a 7x7x832
    x = Inception(x,[(384,),(192,384),(48,128),(128,)]) #Inception 5b 7x7x1024

    #Using AveragePooling replace flatten
    x = AveragePooling2D(pool_size=(7,7),strides=(7,7),padding='same')(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1000,activation='relu')(x)
    x = Dense(classes,activation='softmax')(x)
    
    model=Model(inputs=inpt,outputs=x)
    model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])

    return model
#################################################

def AlexNet(width, height, depth, classes):
    model = Sequential()
    
    #First Convolution and Pooling layer
    model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(width,height,depth),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    
    #Second Convolution and Pooling layer
    model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    
    #Three Convolution layer and Pooling Layer
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    
    #Fully connection layer
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000,activation='relu'))
    model.add(Dropout(0.5))
    
    #Classfication layer
    model.add(Dense(classes,activation='softmax'))

    model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                    loss = 'categorical_crossentropy',metrics=['accuracy'])

    return model


def LeNet(width, height, depth, classes_num):
    # initialize the model
    model = Sequential()

    # first layer, convolution and pooling
    model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(5, 5), filters=6, strides=(1,1), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second layer, convolution and pooling
    model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(5, 5), filters=16, strides=(1,1), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fully connection layer
    model.add(Flatten())
    model.add(Dense(120,activation = 'tanh'))
    model.add(Dense(84,activation = 'tanh'))

    # softmax classifier
    model.add(Dense(classes_num))
    model.add(Activation("softmax"))

    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model

def CNN_V1(width, height, depth, classes_num):
    cnn_model = Sequential([
    
        Conv2D(filters=32, kernel_size=(3, 3), 
                            padding='same',
                            input_shape=(width, height, depth),
                            activation='relu'),
        Conv2D(filters=32, kernel_size=(3, 3), 
                            activation='relu'),          
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(filters=64, kernel_size=(3, 3), 
                            padding='same',
                            activation='relu'),
        Conv2D(filters=64, kernel_size=(3, 3), 
                            activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(filters=128, kernel_size=(3, 3), 
                            padding='same',
                            activation='relu'),
        Conv2D(filters=128, kernel_size=(3, 3), 
                            activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(classes_num, activation='softmax')
    ])

    cnn_model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),#tf.keras.optimizers.Adam(),#tf.train.AdamOptimizer(),
                loss='categorical_crossentropy',#tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

    return cnn_model
   
def CNN_V3(width, height, depth, classes_num):
    inpt = Input(shape=(width,height,depth))
    x = BatchNormalization()(inpt)
    x = Conv2D(16, kernel_size=9, activation='relu')(x)
    x = Conv2D(16, kernel_size=9, activation='relu')(x)
    x = MaxPooling2D(pool_size=9)(x)
    x = Dropout(rate=0.1)(x)
    x = Conv2D(32, kernel_size=3, activation='relu')(x)
    x = Conv2D(32, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(rate=0.1)(x)
    x = Conv2D(128, kernel_size=3, activation='relu')(x)
    x = GlobalMaxPooling2D()(x)
    x = Dropout(rate=0.1)(x)

    x = BatchNormalization()(Dense(128, activation='relu')(x))
    x = BatchNormalization()(Dense(128, activation='relu')(x))
    x = Dense(classes_num, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), 
                    loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def VGG16Net(width, height, depth, classes):
    model = Sequential()
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(width,height,depth),padding='same',activation='relu'))
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes,activation='softmax'))
    
    model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])

    return model
