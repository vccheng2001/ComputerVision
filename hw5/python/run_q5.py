import numpy as np
import scipy.io
from nn import *
from collections import Counter
from util import *
train_data = scipy.io.loadmat('../data/nist36_train.mat')

# we don't need labels now!
train_x = train_data['train_data']

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

initialize_weights(1024,32,params,'layer1')
initialize_weights(32,32,params,'layer2')
initialize_weights(32,32,params,'layer3')
initialize_weights(32,1024,params,'output')


layers = ['output', 'layer3', 'layer2', 'layer1']
for layer in layers:
    params["MW"+layer] = 0
    params["Mb"+layer] = 0



train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

batches, _ = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)


n = train_x.shape[0]
losses = []

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
        # print('h1', h1.shape)
        h2 = forward(h1,params,name='layer2',activation=relu)
        # print('h2', h2.shape)
        h3 = forward(h2,params,name='layer3',activation=relu)
        # print('h3', h3.shape)
        image_out = forward(h3,params,name='output',activation=sigmoid)
        assert image_out.shape == xb.shape

        loss = np.sum((xb-image_out)**2) / n

        # loss
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss 

        

        # backward
        delta1 = 2*(image_out - xb)
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
    losses.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9



# run on validation set and report accuracy! should be above 75%

print("***** VALIDATION *********")

val_batches, _ = get_random_batches(valid_x,valid_y,1)#batch_size)
val_batch_num = len(val_batches)
n, m = valid_x.shape
val_accs = []
val_losses = []
for val_xb,val_yb in val_batches:
    # print(f'xb={xb},yb={yb}')

    val_h1 = forward(val_xb,params,name='layer1',activation=relu)
    val_h2 = forward(val_h1,params,name='layer2',activation=relu)
    val_h3 = forward(val_h2,params,name='layer3',activation=relu)
    val_image_out = forward(val_h3,params,name='output',activation=sigmoid)
    assert val_image_out.shape == val_xb.shape

    val_loss = np.sum((val_xb-val_image_out)**2) / n

    val_losses.append(val_loss)
import matplotlib.pyplot as plt

plt.figure()
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(val_xb.reshape(32,32).T)
axarr[0,1].imshow(val_image_out.reshape(32,32).T)
f.show()


# ##########################
# ##### your code here #####
# ##########################

# if True: # view the data
#     for crop in val_xb:
#         import matplotlib.pyplot as plt
#         plt.imshow(crop.reshape(32,32).T)
#         plt.show()





# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
plt.plot(range(max_iters), losses, '-r', label='loss')
plt.show()

# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################
# peak_signal_noise_ratio