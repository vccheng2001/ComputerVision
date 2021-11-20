import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

print('train x', train_x.shape)
print('train_y', train_y.shape)
max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 1e-3
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################
# initialize a layer
# N = 10800 examples
# M = 1024 feature dim
# H = 64 hidden 
# C = 36 classes
initialize_weights(1024,64,params,'layer1')
initialize_weights(64,36,params,'output')

# with default settings, you should get loss < 150 and accuracy > 80%
n = train_x.shape[0]
print('num total train examples: ', n)
batches = get_random_batches(train_x,train_y,5)


print("******STARTING TRAINING LOOP********\n\n")
accs = []
losses = []
# with default settings, you should get loss < 35 and accuracy > 75%
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    for xb,yb in batches:
        # print(f'xb={xb},yb={yb}')

        # forward
        # print('xb', xb.shape) # (5,2)
        # print('yb', yb.shape) # (5,4)
        h1= forward(xb,params,name='layer1',activation=sigmoid)
        probs= forward(h1, params,name='output',activation=softmax)
        # print('Probs', probs.shape)
        loss, acc = compute_loss_and_acc(yb, probs)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss 
        avg_acc += acc

        y_idx = np.argmax(yb, axis=1) # one hot to indices
        # backward
        delta1 = probs
        delta1[np.arange(probs.shape[0]),y_idx] -= 1
        delta2 = backwards(delta1,params,'output',linear_deriv)

        # apply gradient
        params["Woutput"] = params["Woutput"] + (learning_rate*params["grad_Woutput"])
        params["boutput"] = params["boutput"] + (learning_rate*params["grad_boutput"])
   
        backwards(delta2,params,'layer1',sigmoid_deriv)

        params["Wlayer1"] = params["Wlayer1"] + (learning_rate*params["grad_Wlayer1"])
        params["blayer1"] = params["blayer1"] + (learning_rate*params["grad_blayer1"])

    avg_acc /= batch_num # average acc across all batches 
    avg_loss = total_loss / batch_num
    accs.append(avg_acc)
    losses.append(avg_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))

import matplotlib.pyplot as plt


epochs = range(max_iters)
plt.plot(epochs, accs, '-b', label="accuracy")
plt.plot(epochs, losses, '-r', label='loss')

plt.show()

# run on validation set and report accuracy! should be above 75%
valid_acc = None
##########################
##### your code here #####
##########################

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################
##### your code here #####
##########################

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
##########################

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()