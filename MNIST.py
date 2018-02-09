
# coding: utf-8

# In[10]:

#!python2

# setting up compute backend 

from neon.backends import gen_backend

batch_size = 32

be = gen_backend(batch_size = batch_size)


# In[11]:

# getting our data

from neon.data import MNIST

mnist = MNIST(path='data/')


# In[12]:

# splitting into train and test
train_set = mnist.train_iter
valid_set = mnist.valid_iter


# In[13]:

# initializing weights
# Gaussian distribution with mean = 0 and S.Deviation = 0.01

from neon.initializers import Gaussian

init_norm = Gaussian(loc=0.0, scale=0.01)


# In[14]:

# model architecture - multi layer perceptron with fully connected layers
# Affine - Fully connected Layer
# Rectlin - Rectified Linear Activation Function
# Softmax - ensure sum(outputs) = 1 and outputs within range [0,1]

from neon.layers import Affine
from neon.transforms import Rectlin, Softmax

layers = []
layers.append(Affine(nout=10, init=init_norm, activation=Rectlin()))
layers.append(Affine(nout=10, init=init_norm,
                     activation=Softmax()))


# In[15]:

# initialize model object

from neon.models import Model

mlp = Model(layers=layers)


# In[16]:

# cost function - Cross Entropy Loss and Generalized Cost Layer

from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyMulti

cost = GeneralizedCost(costfunc=CrossEntropyMulti())


# In[17]:

# learning rules - stochastic gradient descent 
# learning rate = 0.1 , momentum coefficient = 0.9

from neon.optimizers import GradientDescentMomentum

optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9)


# In[18]:

# API Callbacks - calls operations during model fit

from neon.callbacks.callbacks import Callbacks

callbacks = Callbacks(mlp, eval_set=valid_set)


# In[19]:

# training our model
# GRADED FUNCTION
mlp.fit(train_set, optimizer=optimizer, num_epochs=10, cost=cost, callbacks=callbacks)


# In[20]:

# getting our outputs

results = mlp.get_outputs(valid_set)


# In[21]:

# evaluating model performance using misclassification rate

from neon.transforms import Misclassification

error = mlp.eval(valid_set, metric=Misclassification())*100
print('Misclassification error = %.1f%%' % error)


# In[22]:

# inference
# new digit image

import urllib.request
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# download image
url = "http://datawrangling.s3.amazonaws.com/sample_digit.png"
urllib.request.urlretrieve(url, filename="data/digit.jpg")

# scale to 28x28 pixels
img = Image.open("data/digit.jpg")
img.thumbnail((28, 28))

digit = np.asarray(img, dtype=np.float32)[:, :, 0]

# reshape to a single feature vector
digit = digit.reshape(784, 1)

# store digit into a GPU tensor
x_new = be.zeros((28*28, batch_size), dtype=np.float32)
x_new[:, 0] = digit


# In[23]:

# testing our model
# forward pass through the model
outputs = mlp.fprop(x_new)
outputs = outputs.get()[:, 0]

# examine the output of the model for this image
print("Model final layer was: {}".format(outputs))
print("The most probable guess is digit: {}".format(np.argmax(outputs)))
plt.figure(2)
plt.imshow(img)


# In[ ]:



