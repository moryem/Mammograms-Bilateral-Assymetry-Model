"""
Created on Tue Oct 23 13:11:45 2018

@author: Mor
"""

import os
import numpy as np
import cv2
from sklearn.utils import shuffle 
import matplotlib.pyplot as plt
from imutils import rotate

def load_Soroka_DB():
# =============================================================================
#       load Soroka Database for training
# =============================================================================
    
    # load all data
    X = np.load('./mammos.npy')
    X = np.squeeze(X)
    X = np.divide(X,np.max(X))
    
    # split to train and validation
    # take amount of images that are devided by 3
    Xtr = X[:int(0.8*X.shape[0])+1,:,:].astype(float)
    Xtr = Xtr[:Xtr.shape[0]-Xtr.shape[0]%3,:,:]
    Xval = X[int(0.8*X.shape[0]):,:,:].astype(float)
    Xval = Xval[:Xval.shape[0]-Xval.shape[0]%3,:,:]
   
    return Xtr, Xval

def circle_rotate(image, x, y, radius):
# =============================================================================
#   circle rotate a patch of the image
# =============================================================================
    
    crop = image[x-radius:x+radius+1,y-radius:y+radius+1] 
    
    # build the cirle mask
    mask = np.zeros(crop.shape) 
    for i in range(crop.shape[0]):
        for j in range(crop.shape[1]):
            if (i-radius)**2 + (j-radius)**2 <= radius**2:
                mask[i,j] = 1
                
    # create the new circular image
    sub_img = np.empty(crop.shape ,dtype='uint8')
    sub_img = mask * crop  
    angle = np.random.randint(40,125) # random angle between 40 to 125 degrees
    M = cv2.getRotationMatrix2D((crop.shape[0]/2,crop.shape[1]/2),angle,1)
    dst = cv2.warpAffine(sub_img,M,(crop.shape[0],crop.shape[1]))  

    # return the whole image after distortion
    i2 = image.copy()
    i2[x-radius:x+radius+1,y-radius:y+radius+1] = crop * (1-mask)
    i2[x-radius:x+radius+1,y-radius:y+radius+1] += dst
    
    return i2    

def translate(image, x, y, length):
# =============================================================================
#   translate a patch in the image    
# =============================================================================

    crop = image[x:x+length,y:y+length]
                
    # translate
    rand_x = int(np.random.randint(0,crop.shape[0]/2)) # random translation on x axis
    rand_y = int(np.random.randint(0,crop.shape[1]/2)) # random translation on y axis
    M = np.float32([[1,0,rand_x],[0,1,rand_y]])
    dst = cv2.warpAffine(crop,M,(crop.shape[0],crop.shape[1]))
    
    # create the mask
    mask = np.zeros(crop.shape,'int')
    ind_zero = np.where(dst==0)
    mask[ind_zero] = 1
    
    # return the whole image after distortion
    i2 = image.copy()
    i2[x:x+length,y:y+length] = crop * mask
    i2[x:x+length,y:y+length] += dst
    
    return i2

def affine(image, x, y, length):
# =============================================================================
#   affine transformation of a patch in the image    
# =============================================================================

    crop = image[x:x+length,y:y+length]
                
    # affine transformation
    s10 = (0,0)
    s11 = (crop.shape[1],0)
    s12 = (0,crop.shape[0])
    pts1 = np.float32([s10,s11,s12])
    s20 = (crop.shape[0] * 0.0,crop.shape[1] * 0.33)
    s21 = (crop.shape[0] * 0.85,crop.shape[1] * 0.25)
    s22 = (crop.shape[0] * 0.15,crop.shape[1] * 0.7)
    pts2 = np.float32([s20,s21,s22])
    M = cv2.getAffineTransform(pts1,pts2)    
    dst = cv2.warpAffine(crop,M,(crop.shape[0],crop.shape[1]))
    
    # create the mask
    mask = np.zeros(crop.shape,'int')
    ind_zero = np.where(dst==0)
    mask[ind_zero] = 1
    
    # return the whole image after distortion
    i2 = image.copy()
    i2[x:x+length,y:y+length] = crop * mask
    i2[x:x+length,y:y+length] += dst
    
    return i2

def brightness(image, x, y, length):
# =============================================================================
#   change the brightness of the image
# =============================================================================
    
    crop = image[x:x+length,y:y+length]
    
    # change brightness
    brightness = np.random.uniform(0.2,0.5)   
    factor = np.random.uniform(-brightness,brightness)
    dst = crop + factor

    # create the mask
    mask = np.zeros(crop.shape,'int')
    ind_zero = np.where(dst==0)
    mask[ind_zero] = 1
    
    # return the whole image after distortion
    i2 = image.copy()
    i2[x:x+length,y:y+length] = crop * mask
    i2[x:x+length,y:y+length] += dst
    
    return i2

def contrast(image, x, y, length):
# =============================================================================
#   change the contrast of the image
# =============================================================================
    
    crop = image[x:x+length,y:y+length]
    
    # change contrast
    contrast = np.random.uniform(0.4,0.8) 
    image_mean = np.mean(np.mean(crop,axis=0),axis=0)
    dst = (crop - image_mean) * contrast + image_mean

    # create the mask
    mask = np.zeros(crop.shape,'int')
    ind_zero = np.where(dst==0)
    mask[ind_zero] = 1
    
    # return the whole image after distortion
    i2 = image.copy()
    i2[x:x+length,y:y+length] = crop * mask
    i2[x:x+length,y:y+length] += dst

    return i2

def flip(image, x, y, length):
# =============================================================================
#     flip on x - axis
# =============================================================================

    crop = image[x:x+length,y:y+length]

    # flip on x axis
    dst = cv2.flip(crop, 1)
    
    # create the mask
    mask = np.zeros(crop.shape,'int')
    ind_zero = np.where(dst==0)
    mask[ind_zero] = 1
    
    # return the whole image after distortion
    i2 = image.copy()
    i2[x:x+length,y:y+length] = crop * mask
    i2[x:x+length,y:y+length] += dst

    return i2
   
def dist_part(img, perc, func):
# =============================================================================
#     send to one of the distortions
# =============================================================================

    if func.__name__ == 'circle_rotate':
        # radius length
        radius = int(np.round(np.sqrt(perc) * img.shape[0])/2-1)
        # center location
        rand_x = np.random.randint(low = radius, high = img.shape[0]-radius)
        rand_y = np.random.randint(low = radius, high = img.shape[1]-radius)
        # distort
        dst = func(img, rand_x, rand_y, radius)
    else:       
        # size of patch
        x = int(np.round(np.sqrt(perc) * img.shape[0]))  
        y = int(np.round(np.sqrt(perc) * img.shape[1])) 
        
        # generate random locations
        rand_ind_x = np.random.randint(low = 0, high = img.shape[0]-x)
        rand_ind_y = np.random.randint(low = 0, high = img.shape[1]-y)
        
        # distort the patch
        dst = func(img, rand_ind_x, rand_ind_y, x)
        
    return dst
    
def matched_pair_perc(X):
# =============================================================================
#   create the parallel set for the siamese net and the labels
# =============================================================================
        
    # similar pairs
    Cs = X[:int(X.shape[0]/3),:,:]
    
    # totaly different pairs
    Cdf = Cs
    
    # level of distortion
    y_s = np.concatenate((np.zeros((Cs.shape[0],1),dtype=int),np.ones((Cdf.shape[0],1),dtype=int)))    

    # one distorted image in a pair
    dist_X =  X[int(2*X.shape[0]/3):,:,:]
    p = np.linspace(0,1,30)
    p = p[1:len(p)-1] # skip 0 and 1
    funcs = [brightness] #[circle_rotate, translate, affine, brightness, contrast, flip] 
    Cds = []
    for x in dist_X:
        i = np.random.randint(0,len(p))
        j =  np.random.randint(0,len(funcs))
        ds_x = dist_part(x, p[i] ,funcs[j])
        Cds.append(ds_x)
        y_s = np.append(y_s,p[i])
    Cds = np.asarray(Cds)

    # total pairs
    C = np.concatenate((Cs,Cdf,Cds),axis=0)    

    # # rotate to get variability
    # angle = np.random.randint(1,8)
    # C = np.asarray([rotate(C[i,:,:],angle) for i in range(C.shape[0])])

    # create the labels - 1 if totaly different and 0 otherwise
    y = np.concatenate((np.zeros((Cs.shape[0],1),dtype=int),np.ones((Cdf.shape[0],1),
                                 dtype=int),np.zeros((Cds.shape[0],1),dtype=int)))    
    # shuffle the data
    X, C, y, y_s = shuffle(X, C, y, y_s, random_state=52)
    
    return X, C, y, y_s

def data_prep_perc():
# =============================================================================
#     prepare the data for train, validation and test
# =============================================================================

    Xtr, Xval = load_Soroka_DB()
    
    Xtr, Ctr, y_tr, l_tr = matched_pair_perc(Xtr)
    Xval, Cval, y_val, l_val = matched_pair_perc(Xval)

    #expand dims
    Xtr = Xtr[:,:,:,np.newaxis]
    Ctr = Ctr[:,:,:,np.newaxis]
    Xval = Xval[:,:,:,np.newaxis]
    Cval = Cval[:,:,:,np.newaxis]
    
    np.save('right_train',Xtr)
    np.save('right_val',Xval)
    np.save('left_train',Ctr)
    np.save('left_val',Cval)
    np.save('y_tr',y_tr)
    np.save('y_val',y_val)
    np.save('l_tr',l_tr)
    np.save('l_val',l_val)

    return Xtr, Xval, Ctr, Cval, y_tr, y_val, l_tr, l_val
    
def load_train():
# =============================================================================
#   Load train data    
# =============================================================================
    if not os.path.exists('right_train.npy'):
        right_train, right_val, left_train, left_val, y_tr, y_val, l_tr, l_val = data_prep_perc()
    else:
        right_train = np.load('right_train.npy')
        right_val = np.load('right_val.npy')
        left_train = np.load('left_train.npy')
        left_val = np.load('left_val.npy')
        y_tr = np.load('y_tr.npy')
        y_val = np.load('y_val.npy')
        l_tr = np.load('l_tr.npy')
        l_val = np.load('l_val.npy')   
    
    return right_train, right_val, left_train, left_val, y_tr, y_val, l_tr, l_val

def load_test():
# =============================================================================
#   Load test data
# =============================================================================
    if os.path.exists('right_test.npy'):
        Xte = np.load(os.path.join('.','right_test.npy'))
#        Xte = np.divide(Xte,np.max(Xte))
        Cte = np.load(os.path.join('.','left_test.npy'))
#        Cte = np.divide(Cte,np.max(Cte))
        yte = np.load(os.path.join('.','y_te.npy'))
            
    else:
        print('Files do not exists')
        Xte, Cte, yte = 0,0,0
   
    return Xte, Cte, yte

def visualize(twin_net, img, ref, im_num):
# =============================================================================
#   Visualize the outputs of the twins net    
# =============================================================================
    
    # Input images
    img_tensor = np.expand_dims(img,axis=0)
    ref_tensor = np.expand_dims(ref,axis=0)
    
    # Right twin's output
    right_output = twin_net.predict(img_tensor)
    left_output = twin_net.predict(ref_tensor)

    # plot the output image and it's refernce image
    plt.subplot(2,2,1)
    plt.imshow(np.squeeze(img)); plt.axis('off') 
    plt.title('Right twin input')
    plt.subplot(2,2,2)
    plt.imshow(np.squeeze(ref)); plt.axis('off')
    plt.title('Left twin input')
    plt.subplot(2,2,3)
    plt.imshow(np.squeeze(right_output)); plt.axis('off')
    plt.title('Right twin output')
    plt.subplot(2,2,4)
    plt.imshow(np.squeeze(left_output)); plt.axis('off')
    plt.title('Left twin output')
    plt.savefig('vis_examp ' + str(im_num))
    plt.show(); plt.close()

