import numpy as np
import scipy.io
from nn import *

# train_data = scipy.io.loadmat('../data/nist36_train.mat')
# valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# train_x, train_y = train_data['train_data'], train_data['train_labels']
# valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

# print('train x', train_x.shape)
# print('train_y', train_y.shape)
# max_iters = 50
# # pick a batch size, learning rate
# batch_size = 32
# learning_rate = 1e-3
# hidden_size = 64
# ##########################
# ##### your code here #####
# ##########################

# batches, _ = get_random_batches(train_x,train_y,batch_size)
# batch_num = len(batches)

# params = {}

# # initialize layers here
# ##########################
# ##### your code here #####
# ##########################
# # initialize a layer
# # N = 10800 examples
# # M = 1024 feature dim
# # H = 64 hidden 
# # C = 36 classes
# initialize_weights(1024,64,params,'layer1')
# initialize_weights(64,36,params,'output')

# # with default settings, you should get loss < 150 and accuracy > 80%
# n = train_x.shape[0]
# print('num total train examples: ', n)


# print("******STARTING TRAINING LOOP********\n\n")
# accs = []
# losses = []
# avg_accs = []
# # with default settings, you should get loss < 35 and accuracy > 75%
# for itr in range(max_iters):
#     print(f"******STARTING ITER {itr}***\n")
#     total_loss = 0
#     accs = []
#     for xb,yb in batches:
#         # print(f'xb={xb},yb={yb}')

#         # forward
#         # print('xb', xb.shape) # (5,2)
#         # print('yb', yb.shape) # (5,4)
#         h1 = forward(xb,params,name='layer1',activation=sigmoid)
#         probs= forward(h1, params,name='output',activation=softmax)


#         # print('Probs', probs.shape)
#         loss, acc = compute_loss_and_acc(yb, probs)

#         # if itr % 10 == 0:
#             # print(f'curr batch loss ={loss}, acc={acc}')

#         # loss
#         # be sure to add loss and accuracy to epoch totals 
#         total_loss += loss 
#         accs.append(acc)

#         y_idx = np.argmax(yb, axis=1) # one hot to indices
        

#         # backward derivative of SCE 
#         delta1 = probs
#         delta1[np.arange(probs.shape[0]),y_idx] -= 1

#         delta2 = backwards(delta1,params,'output',linear_deriv)
#         backwards(delta2,params,'layer1',sigmoid_deriv)

#         # apply gradient
#         params["Woutput"] = params["Woutput"] - (learning_rate*params["grad_Woutput"])
#         params["boutput"] = params["boutput"] - (learning_rate*params["grad_boutput"])
   

#         params["Wlayer1"] = params["Wlayer1"] - (learning_rate*params["grad_Wlayer1"])
#         params["blayer1"] = params["blayer1"] - (learning_rate*params["grad_blayer1"])

#     avg_acc = np.mean(accs)
#     avg_accs.append(avg_acc)
#     total_loss /= batch_num
#     losses.append(total_loss)
    
#     if itr % 1 == 0:
#         print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))


# import matplotlib.pyplot as plt


# plt.plot(range(max_iters), avg_accs, '-b', label="accuracy")
# plt.plot(range(max_iters), losses, '-r', label='loss')

# plt.show()


# # run on validation set and report accuracy! should be above 75%

# val_batches, _ = get_random_batches(valid_x,valid_y,1)#batch_size)
# val_batch_num = len(batches)

# val_accs = []
# val_losses = []
# for val_xb,val_yb in val_batches:
#     # print(f'xb={xb},yb={yb}')

#     val_h1 = forward(val_xb,params,name='layer1',activation=sigmoid)
#     val_probs= forward(val_h1, params,name='output',activation=softmax)


#     # print('Probs', probs.shape)
#     val_loss, val_acc = compute_loss_and_acc(val_yb, val_probs)
#     print(f'val_loss={val_loss},val_acc={val_acc}')

#     val_losses.append(val_loss)
#     val_accs.append(val_acc)




# ##########################
# ##### your code here #####
# ##########################

# print('Validation accuracy: ',val_acc)
# if True: # view the data
#     for crop in xb:
#         import matplotlib.pyplot as plt
#         plt.imshow(crop.reshape(32,32).T)
#         # plt.show()
# import pickle
# saved_params = {k:v for k,v in params.items() if '_' not in k}
# with open('q3_weights.pickle', 'wb') as handle:
#     pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

weights = np.load('../out/q3_weights.pickle', allow_pickle=True)
print(weights)
# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here


def squarify(M,val):
    (a,b)=M.shape
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    return np.pad(M,padding,mode='constant',constant_values=val)

first_layer_weights = []
for k, v in weights.items():
    if not 'layer1' in k: continue
    first_layer_weights += [[k,v]]
    
for i in range(len(first_layer_weights)):
    fig = plt.figure(1, (4,4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(1,1),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    k, v = first_layer_weights[i]
    print('param: ', k)
    try:
        x,y = v.shape
    except:
        v = np.expand_dims(v, 1)
        x,y = v.shape
    

    im = v.flatten()
    sq= int(np.sqrt(len(im)))
        
    im.shape = sq, sq
    grid[0].imshow(im)  # The AxesGrid object work as a list of axes.

    plt.show()


# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute confusion matrix here
##########################
##### your code here #####
##########################

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()