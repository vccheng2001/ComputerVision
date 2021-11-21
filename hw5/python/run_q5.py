import numpy as np
import scipy.io
from nn import *
from collections import Counter
from util import *
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################


params = {}
initialize_weights(1024,32,params,'layer1')
initialize_weights(32,32,params,'layer2')
initialize_weights(32,32,params,'layer3')
initialize_weights(32,1024,params,'output')


layers = ['output', 'layer3', 'layer2', 'layer1']
for layer in layers:
    params["MW"+layer] = 0
    params["Mb"+layer] = 0


train_data = scipy.io.loadmat('../data/nist36_train.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
batches, _ = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20


# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        h1 = forward(xb,params,name='layer1',activation=relu)
        h2 = forward(h1,params,name='layer2',activation=relu)
        h3 = forward(h2,params,name='layer3',activation=relu)
        image_out = forward(h3,params,name='output',activation=sigmoid)

        loss = np.sum((xb-image_out)**2)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss 

        # backward
        delta1 = 2*(xb - image_out)
        delta2 = backwards(delta1,params,'output',sigmoid_deriv)
        delta3 = backwards(delta2,params,'layer3',relu_deriv)
        delta4 = backwards(delta3,params,'layer2',relu_deriv)
        backwards(delta4,params,'layer1',relu_deriv)

        # apply gradient
        for layer in layers:
            # calc momentum
            params["MW"+layer] = 0.9*params["MW"+layer] - learning_rate * params["grad_W"+layer]
            params["Mb"+layer] = 0.9*params["Mb"+layer] - learning_rate * params["grad_b"+layer]

            # update weights 
            params["W"+layer] = params["W"+layer] + params["MW"+layer]
            params["b"+layer] = params["b"+layer] + params["Mb"+layer]

   
    total_loss /= batch_num

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
##########################


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################
