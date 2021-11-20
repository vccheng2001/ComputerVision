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
    print('**** Initialize weights ***')

    W, b = None, None

    W = np.zeros((in_size, out_size))
    b = np.zeros((out_size,))

    params['W' + name] = W
    params['b' + name] = b


    

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    print('**** Sigmoid ****')

    ##########################
    ##### your code here #####
    ##########################
    res = 1 / (1+np.exp(-x))
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
    print('**** Forward ****')

    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]




    ##########################
    ##### your code here #####
    ##########################

    # linear combination 

    N, M = X.shape
    M, D = W.shape
    # print(f'N={N}, M={M}, D={D}')
    # print('W', W.shape) # M,D
    # print('X', X.shape) # N,M
    # print('b', b.shape) # D,1
    # print((X @ W).shape)
    # print(np.expand_dims(b, 0).shape)
    pre_act  = X @ W + np.expand_dims(b, 0) # (NxD)+(1xD)

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
    print('**** Softmax ****')
    res = None

    ##########################
    ##### your code here #####
    ##########################

    exp = np.exp(x-np.max(x))
    res = exp / np.sum(exp, axis=1, keepdims=True)

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is one hot, size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    print('**** Compute loss, acc ****')

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
    print('**** Sigmoid deriv ****')
    print('inp', post_act.shape)

    res = post_act*(1.0-post_act)
    print('out', res.shape)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    print(f'****Backwards, name={name}, activ={activation_deriv}****')
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """

    '''
    a = Wx+b
    o = sigmoid(a)

    W:25,4 25-->4
    X in: 40,25
    Wx+b=preact=40,4
    postact:40,4
    '''
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    print("X", X.shape)

    print('pre_act', pre_act.shape)
    print('post_act', post_act.shape)
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    ##########################
    # z = Wx+b
    #  o = σ(z) = o(Wx+b)
    #  σ'(x) =  σ(x)[1- σ(x)]
    print('W', W.shape) # M=25, 4
    print('b', b.shape) # 4
    print('delta',delta.shape)
    print

    if activation_deriv == sigmoid_deriv:
        print('aa', delta.shape, sigmoid_deriv(post_act).shape)
        # (40x40) @ (25x40) = (40x25)
        dL_dz = delta @ sigmoid_deriv(post_act)
    elif activation_deriv == linear_deriv:
        dL_dz = delta.T
    print('dL_dz', dL_dz.shape)



    dz_dX = W.T
    dz_dW = X
    dz_db = 1
    print('dz_dx', dz_dX.shape)
    print('dz_dW', dz_dW.shape)

    #dL/dW = dL/do @ do/dz @ dz/dW
    # (40,25) @ (40,2) = (25,2).T = (2,25)
    grad_W = (dL_dz @ dz_dW).T
    assert grad_W.shape == W.shape #output:(25,4), layer1: (2,25)

    # (40,25) @ (25,2) = (40,2)
    grad_X = dL_dz.T @ dz_dX
    # assert(grad_X.shape == (40,25)) # output


    dz_db = np.ones((40,1))
    # (40,25 ) @ (40, 1)  = (25,1)
    grad_b = dL_dz @ dz_db 
    grad_b = grad_b.squeeze()
    assert grad_b.shape == b.shape #output:(4,), layer1:(25,)

    print(f'grad_W={grad_W.shape}, grad_X={grad_X.shape}, grad_b={grad_b.shape}')

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
