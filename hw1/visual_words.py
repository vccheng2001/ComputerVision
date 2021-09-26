import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_laplace
import skimage.color
from matplotlib import pyplot as plt
import sklearn
from sklearn import cluster
import scipy
from multiprocessing import Process, Queue, Pool

def extract_filter_responses(opts, img):
    
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    # # show original img
    # plt.imshow(img)
    # plt.title('original img')
    # plt.show()
    # checks
    assert(((0 <= img) & (img <= 1)).all()), "pixels must be in range [0,1]"

    # if grayscale, change to rgb
    if img.ndim == 3 and img.shape[-1] == 1:
        print('changing grayscale to RGB')
        img = np.repeat(img, repeats=3, axis=2)
    # if two dimensional 
    elif img.ndim == 2:
        print("changing (H,W) to (H,W,C)")
        img = np.expand_dims(img, 2)
        img = np.repeat(img, repeats=3, axis=2)
    
    H, W, C = img.shape

    # rgb to lab
    img = skimage.color.rgb2lab(img)

    filter_responses = None

    for sigma in opts.filter_scales: # filter scales 
        # gaussian filter
        gaussian = gaussian_filter(img, sigma=sigma, order=0)
    
        # first apply Gaussian, then take Lapplacian 
        log = gaussian_laplace(img, sigma)

        # derivative of gaussian in x-dir
        dog_x = gaussian_filter(img, sigma=sigma,order=[1,0,0])
       
        # derivative of gaussian in y-dir
        dog_y = gaussian_filter(img, sigma=sigma,order=[0,1,0])
      
        # concatenate all filter responses for this scale
        concat_response = np.concatenate((gaussian, log, dog_x, dog_y), axis=2)

        # 3 filter scales * 3 channels * 4 filter responses = 36
        filter_responses = concat_response if filter_responses is None else np.concatenate((filter_responses, concat_response), axis=2)

    return filter_responses # (H, W, 3F)

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    train_file, opts = args 
    img_path = join(opts.data_dir, train_file)


    # open image 
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255 # normalize
    H, W, C = img.shape

    # get filter response from image
    response  = extract_filter_responses(opts, img)

    filter_responses = None

    for i in range(opts.alpha):

        # choose random pixel at loc (i,j) and get its response
        i = np.random.randint(0, H)
        j = np.random.randint(0, W)
        pixel_response = np.expand_dims(response[i, j], 0)

        # add to filter responses
        filter_responses = pixel_response if filter_responses is None else np.concatenate((filter_responses, pixel_response), axis=0)

    return filter_responses
    

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha

    # gather train files 
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    T = len(train_files) 

    # pass in opts as well
    args = [[tf, opts] for tf in train_files]

    # multiprocessing 
    images = []
    p = Pool(n_worker)
    print(f'Using Pool with {n_worker} workers')
    imap = p.imap(compute_dictionary_one_image, args)

    # append each image's filter_responses
    filter_responses = None
    for img_response in imap: 
        filter_responses = img_response if filter_responses is None else np.concatenate((filter_responses, img_response), axis=0)

    print('Q1.2 filter responses', filter_responses.shape)

    # cluster into K centroids
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_ # (K, 36)
    print('Q1.2 Dictionary', dictionary.shape) 

    # save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary) 

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    if img.ndim == 3:
        H, W, _ = img.shape
    else:
        H, W = img.shape

    # extract filte rresponse for image 
    filter_responses = extract_filter_responses(opts, img)
    # reshape 
    filter_responses = np.reshape(filter_responses, (H*W,4*int(len(opts.filter_scales))*3))
    # compare 36-dim vectors for each pixel to each word in dictionary
    idx = scipy.spatial.distance.cdist(filter_responses, dictionary)
    
    # generate wordmap based on minimum distance visual word
    # each pixel should map to the index of the closest color in dictionary
    wordmap = idx.argmin(axis=-1)
    wordmap = np.reshape(wordmap, (H, W))

    return wordmap # (H, W)

