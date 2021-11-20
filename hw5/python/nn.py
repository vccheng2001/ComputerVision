import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):

    ##########################
    ##### your code here #####
    ##########################
    res = 1 / (1+np.e**-x)

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    ##########################
    ##### your code here #####
    ##########################

    # linear combination 
    pre_act  = W @ X + b

    # activation
    if activation == "sigmoid":
        post_act = sigmoid(pre_act)
    else:
        post_act = softmax(pre_act)


    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################

    exp = np.exp(x)
    res = exp / np.sum(exp, axis=1, keepdims=True)

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is one hot, size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################

    n, _ = y.shape

    gt_classes = np.argmax(y, axis=1)
    pred_classes = np.argmax(probs, axis=1)

    loss = np.sum(y * -np.log(probs))
    correct = np.count_nonzero(gt_classes == pred_classes)
    acc = correct / n

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    ##########################

    # calculate dW and db only under training mode
    da               =  df2 @ self.W[1].T                       # dE/da
    self.dW[1]       =  df2.T @ self.a                          # dE/dW1
    df1 = self.db[0] =  da @ activation_deriv # dE/df1
    self.dW[0]       =  df1.T @ self.activations[0].x  


    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):

    # x : N x M
    # y : N x 1 
    N,M = x.shape

    # number of batches 
    num_chunks = N // batch_size 
    
    # nd arrays only shuffled along first axis
    x_shuff = np.random.shuffle(x)

    # randomize examples 
    shuffler = np.random.permutation(N)
    x_shuff = x[shuffler]
    y_shuff = y[shuffler]
    
    batches = []
    i = 0

    for j in range(num_chunks):
        # each batch should be x:bxM, y:bx1
        batches.append(x_shuff[i:i+batch_size], y_shuff[i:i+batch_size])
        i += batch_size 


    ##########################
    ##### your code here #####
    ##########################
    return batches
