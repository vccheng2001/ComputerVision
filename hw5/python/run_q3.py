import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle
import string


# one hot to idx 
def one_hot_to_idx(y):
    return np.argmax(y, axis=1)


# plot weights of NN as a (32x32xN) image
def plot_weights(W):
    print(f"**** PLOTTING WEIGHTS ****")
    fig = plt.figure(1, (4,4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(1,1),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    W = W.reshape((32,32,64))
    W = W.flatten()
    sq= int(np.sqrt(len(W)))
    W.shape = sq, sq
    grid[0].imshow(W)  # The AxesGrid object work as a list of axes.
    plt.show()

# load trian, valid data
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

print('train x', train_x.shape)
print('train_y', train_y.shape)



max_iters = 100
batch_size = 64
learning_rate = 1e-3
hidden_size = 64

# train loader
batches, _ = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

# val loader 
val_batches, _ = get_random_batches(valid_x,valid_y,batch_size)
val_batch_num = len(val_batches)



params = {}

# initialize a layer
# N = 10800 examples
# M = 1024 feature dim
# H = 64 hidden 
# C = 36 classes
initialize_weights(1024,64,params,'layer1')
initialize_weights(64,36,params,'output')


print("PLOTTING INITIAL FIRST LAYER WEIGHTS")
plot_weights(params['Wlayer1'])

# # with default settings, you should get loss < 150 and accuracy > 80%
print("******STARTING TRAINING LOOP********\n\n")
accs = []
losses = []
avg_accs = []

val_accs = []
val_losses = []
val_avg_accs = []


# with default settings, you should get loss < 35 and accuracy > 75%
for itr in range(max_iters):
    total_loss = 0
    val_total_loss = 0
    accs = []
    val_accs = []

    # Train
    for xb,yb in batches:
       
        h1 = forward(xb,params,name='layer1',activation=sigmoid)
        probs= forward(h1, params,name='output',activation=softmax)


        # print('Probs', probs.shape)
        loss, acc = compute_loss_and_acc(yb, probs)

        # if itr % 10 == 0:
            # print(f'curr batch loss ={loss}, acc={acc}')

        # loss
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss 
        accs.append(acc)

        y_idx = np.argmax(yb, axis=1) # one hot to indices
        
        # backward derivative of SCE 
        delta1 = probs
        delta1[np.arange(probs.shape[0]),y_idx] -= 1

        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)

        # apply gradient
        params["Woutput"] = params["Woutput"] - (learning_rate*params["grad_Woutput"])
        params["boutput"] = params["boutput"] - (learning_rate*params["grad_boutput"])
   

        params["Wlayer1"] = params["Wlayer1"] - (learning_rate*params["grad_Wlayer1"])
        params["blayer1"] = params["blayer1"] - (learning_rate*params["grad_blayer1"])

    avg_acc = np.mean(accs)
    avg_accs.append(avg_acc)
    total_loss /= batch_num
    losses.append(total_loss)


    # Validation
    for val_xb,val_yb in val_batches:

        val_h1 = forward(val_xb,params,name='layer1',activation=sigmoid)
        val_probs= forward(val_h1, params,name='output',activation=softmax)

        val_loss, val_acc = compute_loss_and_acc(val_yb, val_probs)

        val_total_loss += val_loss
        val_accs.append(val_acc)

    val_avg_acc = np.mean(val_accs)
    val_avg_accs.append(val_avg_acc)
    val_total_loss /= val_batch_num
    val_losses.append(val_total_loss)
    if itr % 1 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f} \t val_loss: {:.2f} \t val_acc : {:.2f}".format(itr,total_loss,avg_acc, val_total_loss, val_avg_acc))




# run on validation set and report accuracy! should be above 75%
print('FINAL VALID ACC', np.mean(val_avg_accs))
# print
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
axes[0].plot(range(max_iters), losses, '-b', label="train_loss")
axes[1].plot(range(max_iters), val_losses, '-r', label="val loss")
plt.legend()

fig.tight_layout()
plt.show()

# print losses 
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
axes[0].plot(range(max_iters), avg_accs, '-b', label="train_acc")
axes[1].plot(range(max_iters), val_avg_accs, '-r', label="val acc")
plt.legend()
fig.tight_layout()
plt.show()

# save_fig = f"../out/q3_iters{max_iters}_bs{batch_size}_lr{learning_rate}_hs{hidden_size}.jpg"
# plt.savefig(save_fig, format="jpg") 




# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
# axes[0].plot(range(len(avg_accs)), avg_accs, '-b', label="accuracy")
# axes[1].plot(range(len(val_accs)), val_accs, '-r', label='val accuracy')
# plt.legend()
# fig.tight_layout()
# plt.show()


# ##########################
# ##### your code here #####
# ##########################

if True: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        # plt.show()

saved_params = {k:v for k,v in params.items() if '_' not in k}
save_file = f'../out/q3_weights_iters{max_iters}_bs{batch_size}_lr{learning_rate}_hs{hidden_size}.pickle'

with open(save_file, 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

weights = np.load(save_file, allow_pickle=True)

test_data = scipy.io.loadmat('../data/nist36_test.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']

print("***** TESTING *********")
test_batches, _ = get_random_batches(test_x,test_y,batch_size)
v_batch_num = len(test_batches)
test_params = weights
test_accs = []
test_losses = []
confusion_matrix = np.zeros((test_y.shape[1],test_y.shape[1]))

for test_xb,test_yb in test_batches:
    # print(f'xb={xb},yb={yb}')

    test_h1 = forward(test_xb, test_params,name='layer1',activation=sigmoid)
    test_probs= forward(test_h1, test_params,name='output',activation=softmax)

    # fill in conf matrix (gt: rows, preds: cols)
    testy_preds = np.argmax(test_probs, axis=1) # o
    testy_gt = np.argmax(test_yb, axis=1)
    confusion_matrix[testy_gt[0]][testy_preds[0]] += 1


    # print('Probs', probs.shape)
    test_loss, test_acc = compute_loss_and_acc(test_yb, test_probs)
    print(f'test_loss={test_loss},test_acc={test_acc}')

    test_losses.append(test_loss)
    test_accs.append(test_acc)


print('FINAL TEST ACC', np.mean(test_acc))




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

print("**** PLOTTING FIRST LAYER WEIGHTS ****")
plot_weights(weights['Wlayer1'])



# Q3.4
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()