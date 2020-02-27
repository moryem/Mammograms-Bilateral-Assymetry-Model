# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 20:56:50 2018

@author: Mor
"""

from keras.models import Model 
from keras.layers import Dropout, BatchNormalization, LeakyReLU, Dense
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Lambda
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.utils import plot_model
from keras import backend as K
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt
from utils_mammo import visualize
from IPython.display import SVG
import json
import os
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
import numpy as np
from imutils import build_montages
import cv2
from glob import glob

class registration_net:
    
    def __init__(self, dp, bs, ep, lr, y_tr, y_val, y_test, right_train, right_val, right_test,
                 left_train, left_val, left_test, im_to_vis, vis, plt_loss, plt_mat, plt_model):
# =============================================================================
#       Initialize parameters
# =============================================================================
        self.dp = dp                                # dropout probability
        self.bs = bs                                # batch size
        self.ep = ep                                # number of epochs
        self.lr = lr                                # learning rate
        self.y_tr = y_tr                            # train targets
        self.y_val = y_val                          # validation targets
        self.y_test = y_test                        # test targets
        self.right_train = right_train              # right net train data
        self.right_val = right_val                  # right net validation data
        self.right_test = right_test                # right net test data
        self.left_train = left_train                # left net train data
        self.left_val = left_val                    # left net validation data
        self.left_test = left_test                  # left net test data
        self.plt_loss = plt_loss                    # boolean - plot loss or not
        self.vis = vis                              # boolean - visualize example or not
        self.im_to_vis = im_to_vis                  # images to visualize
        self.plt_mat = plt_mat                      # boolean - plot confusion matrix or not
        self.plt_model = plt_model                  # boolean - plot model or not
        self.inp_shape = right_train.shape[1:]   # input shape

        
    def twin(self):
# =============================================================================
#       Model architecture        
# =============================================================================
        
        inp = Input(self.inp_shape)
        
        # conv and pooling 1
        x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', 
                   input_shape=self.inp_shape)(inp)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        # conv and pooling 2
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        # conv and pooling 3
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(self.dp)(x)

        # conv and pooling 4
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = BatchNormalization()(x)

        # unpooling and conv 1
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(self.dp)(x)

        # unpooling and conv 2
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        
        # unpooling and conv 3
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(1, (3, 3), strides=(1, 1), padding='same', activation='sigmoid')(x)
        
        # define the twin network
        self.twin_net = Model(inp,x) 

    def net(self):
# =============================================================================
#       merged twin networks                 
# =============================================================================

        # Use the same model - weight sharing
        self.twin()

        in_l = Input(shape=self.inp_shape, name='left_input')
        out_l = self.twin_net(in_l)  

        in_r = Input(shape=self.inp_shape, name='right_input')
        out_r = self.twin_net(in_r)
        
        # Merge the outputs with L2 distance norm
        L2 = Lambda(lambda x: K.sqrt(K.sum(K.sum(K.pow(x[0]-x[1],2), axis=1),axis=1)))
        dist = L2([out_r, out_l])
        out = Dense(1, activation='sigmoid')(dist)

        net = Model([in_r, in_l], out)

        adam = optimizers.Adam(lr=self.lr) #,beta_1=0.9,beta_2=0.95
        net.compile(loss='mse', optimizer=adam, metrics=['mse'])
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-6, 
                                patience=5, verbose=1, mode='min');

        return net, monitor
     
    def train(self):
# =============================================================================
#       Train the model
# =============================================================================
        
        self.reg_model, monitor = self.net()
        self.history = self.reg_model.fit(x=[self.right_train, self.left_train], y=self.y_tr, validation_split=0.3,                            
                          # validation_data=([self.right_val,self.left_val], self.y_val),
                            callbacks=[monitor], batch_size=self.bs, epochs=self.ep, verbose=1); 
        
        # plot loss                                
        if self.plt_loss:
            self.plot_history_loss()
            
        # plot model
        if self.plt_model:
            SVG(model_to_dot(self.reg_model).create(prog='dot', format='svg'))
        
        #save model and twin model
        self.save_model(self.reg_model, self.history.history, 'siam')
        self.save_model(self.twin_net, [], 'twin')
                
    def save_model(self, model, history, name):
# =============================================================================
#       Save model to disk
# =============================================================================
        
        # serialize model to JSON
        model_json = model.to_json()
        with open('model_' + name + '.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize history to JSON
        if name == 'siam':
            with open('history_' + name + '.json', 'w') as f:
                json.dump(history, f)       
        # serialize weights to HDF5
        model.save_weights('weights_' + name + '.h5')
        print('Saved model to disk')
       
    def load_model(self, model_name, weights_name):
# =============================================================================
#       Load model from disk
# =============================================================================
        
        # load json and create model
        json_file = open(model_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(weights_name)
        print('Loaded model from disk')
        
        return loaded_model
        
    def test(self):
# =============================================================================
#         Test the model performance
# =============================================================================

        # Predict models performance on test set
        if os.path.isfile('./model_siam.json'):         
            self.loaded_siam = self.load_model('model_siam.json','weights_siam.h5')
            self.y_pred = self.loaded_siam.predict(x=[self.right_test, self.left_test],
                                                   batch_size=self.bs, verbose=1)
            
            # plot loss                                
            if self.plt_loss:
                self.plot_history_loss()
            # plot confusion matrix
            if self.plt_mat:
                self.plot_conf_mat()
            # visualize middle layers
            if self.vis:
                self.plot_montage()
            # plot model
            if self.plt_model:
                plot_model(self.loaded_siam, to_file='model.png', show_shapes=False, show_layer_names=False)
        else:
            print('Train a model first!')

    def plot_montage(self):
# =============================================================================
#     plot matching maps
# =============================================================================

        # if images don't exist - create them    
        if not glob(os.path.join('.','vis_*')): 
                
            # load model
            loaded_twin = self.load_model('model_twin.json','weights_twin.h5')
    
            # create each image
            for i in self.im_to_vis:            
                right_img = self.right_test[i,:,:,:]
                left_img = self.left_test[i,:,:,:]
                visualize(loaded_twin, right_img, left_img, i)
    
        # get all images in the directory
        imagePaths = glob(os.path.join('.','vis_*'))
        
        # initialize the list of images
        images = []
        
        # loop over the list of image paths
        for imagePath in imagePaths:
        	# load the image and update the list of images
        	image = cv2.imread(imagePath)
        	images.append(image)
         
        # construct the montages for the images
        montages = build_montages(images, (image.shape[1], image.shape[0]), (2, int(len(imagePaths)/2)))

        # loop over the montages and display each of them
        for montage in montages:
            cv2.imwrite('montage.png',montage)

    def plot_conf_mat(self):
# =============================================================================
#       Plot confusion matrix
# =============================================================================
        
        cm = confusion_matrix(self.y_test, np.around(self.y_pred)) 
        labels = ['similar','different']
        plot_confusion_matrix(cm,labels,title='Confusion Matrix - Test',normalize=True)
        plt.savefig('conf_mat_test')
        plt.close()
               
    def plot_history_loss(self):
# =============================================================================
#       Plot loss history        
# =============================================================================
        
        try:
            with open('history_siam.json') as history: 
                hist = json.load(history)
        except:
            print('Creating new loss graph')
            hist = self.history.history
                    
        plt.plot(hist['loss'])
        plt.plot(hist['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('loss')
        plt.show(); plt.close()
        
        
        
        
        
        
        
        
        