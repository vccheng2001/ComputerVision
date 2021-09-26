import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image
import util
import visual_words
from multiprocessing import Process, Queue, Pool


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    hist, bins = np.histogram(wordmap, bins=K)

    # l1 norm to sum histogram to 1
    norm_hist = hist / np.linalg.norm(hist, ord=1) 
    
    return norm_hist


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def get_feature_from_wordmap_SPM(opts, wordmap):
    
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)
ui
    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L

    hist_all = None

    size = min(wordmap.shape)
    wordmap = wordmap[:size, :size]
    H, W = wordmap.shape
    assert(wordmap.shape[0] == wordmap.shape[1])

    for l in range(L): # loop through layer

        # weight each layer
        if l == 0 or l == 1:
            weight = 2**(-L)

        else:
            weight = 2**(l-L-1)


        block_h = H // (2**l) # pixels across block height
        block_w = W // (2**l) # pixels across block width

        # number of subblocks 
        num_subblocks = (2**l) * (2**l)
        # print('num subblocks', num_subblocks)
      
        # split into blocks
        wm = wordmap[:H-(H%(block_h)), :W-(W%(block_w))]
        wm = split(wm, block_h, block_w)   
        
        # get histogram for each subblock (treat as an image)
        for subblock in wm:
            hist, _ = np.histogram(subblock, bins=K)
            # normalize by number of features
            hist = hist/sum(hist)
            # weight
            hist = hist*weight
            # concat
            hist_all = hist if hist_all is None else np.concatenate((hist_all, hist), axis=0)

    # l1 norm to sum histogram to 1
    hist_all = hist_all / np.linalg.norm(hist_all, ord=1) 

    return hist_all

def get_image_feature(args): 
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    opts, train_file, dictionary = args
    
    # load image
    img_path = join(opts.data_dir, train_file)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255 # normalize

    # extract wordmap from image using dictionary
    wordmap = visual_words.get_visual_words(opts, img, dictionary)


    # computes SPM from wordmap
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    feature = np.expand_dims(feature, axis=0)
    
    return feature



def build_recognition_system(opts, wordmap, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    n_worker = util.get_num_CPU()


    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines() # (T)
    print('2.4 num_train', len(train_files))
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)   # (T)
    print('2.4 train_labels', train_labels.shape)
    dictionary = np.load(join(out_dir, 'dictionary.npy')) # (K, 3F)
    print('2.4 dictionary', dictionary.shape)

    all_features = None

    # multiprocessing for getting each image's feature
    p = Pool(n_worker)
    print(f'Using Pool with {n_worker} workers')
    args = [[opts, tf, dictionary] for tf in train_files]
    imap = p.imap(get_image_feature, args)

    for features in imap: 
        all_features = features if all_features is None else np.concatenate((all_features, features), axis=0)

    print('2.4 features', all_features.shape) # (T, ...)

    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=all_features, # save histograms corresponding to all training images (T, K*.../3)
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    # take smaller value for each bin
    # sum up across all bins
    min_vals = np.minimum(word_hist, histograms)
    min_sum = np.sum(min_vals, axis=-1) # larger means nearer (similarity)
    return 1 - min_sum # distance measure

    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    # confusion matrix
    conf = np.zeros((8,8))

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    train_labels =  np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)

    labels = ['aquarium', 'desert', 'highway', 'kitchen', 'laundromat', 'park', 'waterfall', 'windmill']
    for i in range(len(test_files)):
        # ground truth label
        test_file = test_files[i]
        label = test_labels[i]

        # compare test to training set 
        args = opts, test_file, dictionary 
        test_histogram = get_image_feature(args)

        distances = distance_to_set(test_histogram, trained_system['features'])
        # smallest distance means most similar
        predicted_img = np.argmin(distances) # most similar training img
        predicted_label = train_labels[predicted_img]
        # if predicted_label != label:
        #     print('Actual:', label, 'Predicted:', predicted_label)
        # else:
        #     print('Correct', label)

        conf[label][predicted_label] += 1

    # acc: num accurate predictions / all predictions
    accuracy = np.trace(conf) / np.sum(conf)
    print('Conf:', conf)
    print('Accuracy:', accuracy)
    return conf, accuracy


