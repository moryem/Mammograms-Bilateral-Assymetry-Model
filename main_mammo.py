"""
Created on Wed Oct 24 14:00:25 2018

@author: Mor
"""

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from utils_mammo import load_train, load_test
from DeepModel import registration_net
import matplotlib.pyplot as plt
import numpy as np

# load the data
right_train, right_val, left_train, left_val, y_tr, y_val, l_tr, l_val = load_train()
right_test, left_test, y_te = load_test()

#%%
# Define hyper-parameters
bs = 64; ep = 100; dp = 0.55; lr = 6e-6
imgs = np.where(y_te == 1)[0][:8]

reg = registration_net(dp, bs, ep, lr, l_tr, l_val, y_te, right_train, right_val, right_test,
                 left_train, left_val, left_test, im_to_vis = imgs, vis = 1, plt_loss = 1, plt_mat=0, plt_model=0)

#%%
# train the model
reg.train()

#%%
# predict and visualize on test set
reg.test()
y_pred = reg.y_pred

# different color for positive and negative
zeros_idx = np.where(y_te == 0)
ones_idx = np.where(y_te == 1)

# plot dependencies
plt.figure(figsize=(10,8))
plt.scatter(y_pred[zeros_idx], y_te[zeros_idx], s=20, facecolors='none', 
            edgecolors='black',label='Negative Cases')
plt.scatter(y_pred[ones_idx], y_te[ones_idx], s=20, facecolors='none', 
            edgecolors='red',label='Positive Cases')
plt.xlabel('Predicted Target')
plt.ylabel('True Target')
plt.legend(loc = 'center right')
plt.savefig('Mammos Evaluation')
plt.show(); plt.close()

#%%
# predict on the validation set

# Define hyper-parameters
bs = 64; ep = 100; dp = 0.65; lr = 2e-2
imgs = [1, 2, 7, 27, 28, 31, 42, 53]

reg = registration_net(dp, bs, ep, lr, l_tr, l_val, l_val, right_train, right_val, right_val,
                 left_train, left_val, left_val, im_to_vis = imgs, vis = 1, plt_loss = 0, plt_mat=0, plt_model=0)

#%%
reg.test()
y_pred = reg.y_pred

# mae
mae_value = mae(l_val, y_pred)
print('mae = ' + str(mae_value))
# mse
mse_value = mse(l_val, y_pred)
print('mse = ' + str(mse_value))
# R2
r2 = r2_score(l_val, y_pred)
print('R2 = ' + str(r2))

# plot dependencies
plt.scatter(l_val, y_pred, s=10, facecolors='none', edgecolors='black')
plt.xlabel('True Target')
plt.ylabel('Predicted Target')
plt.legend(['$R^2$ = ' + str(np.round(r2,decimals=2))])
plt.savefig('Evaluation')
plt.show(); plt.close()
