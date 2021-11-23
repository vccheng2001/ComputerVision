import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import scipy.io

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)






for img in os.listdir('../images')[:1]:
    print('Processing img: ', img)
    im = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, img_bw = findLetters(im)

    plt.imshow(img_bw, cmap='gray')
    # plt.show()
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################



    # Crop
    processed_images = []
    for bbox in bboxes:


      
        x1, y1, x2, y2 = bbox
        # print(x1,x2, '   ', y1,y2)

        cropped = img_bw[x1:x2, y1:y2] # crop from full image 
        
        # padded = squarify(cropped, 0)
        resized = cv2.resize(src=cropped, dsize=(24,24), interpolation = cv2.INTER_AREA)
        padded = np.pad(resized, (4,4), 'constant', constant_values=(1,1))
       

        kernel = np.ones((5,5),np.uint8)
        eroded = cv2.erode(padded,kernel,iterations = 1)
        eroded = eroded * 255

        thresh =  160
        eroded = cv2.threshold(eroded, thresh, 255, cv2.THRESH_BINARY)[1]

        eroded /= 255

        cv2.imshow("processed", eroded)
        cv2.waitKey(0)

        inp = eroded.T.flatten()
        processed_images.append(np.expand_dims(inp,0))
        




    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    test_params = pickle.load(open('./q3_weights.pickle','rb'))
    test_x = processed_images
    


    # (N, M=1024) (N, K=36)

    print("***** TESTING *********")
    test_accs = []
    test_losses = []
    for test_xb in test_x:
        # print(f'xb={xb},yb={yb}')
        
        # image = cv2.imread("./B.jpg")
        # img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # thresh = 127
        # img_gray = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
        # kernel = np.ones((5,5),np.uint8)

        # img_gray = cv2.erode(img_gray,kernel,iterations = 1)
        # cv2.imshow("Cropped and resized", img_gray)
        # cv2.waitKey(0)


        # test_xb = cv2.resize(img_gray, (32,32), interpolation = cv2.INTER_AREA)
        # test_xb = test_xb.T.flatten()
        # test_xb = np.expand_dims(test_xb, 0)



        test_h1 = forward(test_xb, test_params,name='layer1',activation=sigmoid)
        test_probs= forward(test_h1, test_params,name='output',activation=softmax)

        # fill in conf matrix (gt: rows, preds: cols)
        testy_preds = np.argmax(test_probs, axis=1) 
        print(letters[testy_preds[0]], end=' ')

        predicted = letters[testy_preds[0]]
        cropped = test_xb.reshape(32,32)
        # cv2.imshow(predicted, cropped.T)
        # cv2.waitKey(0)




    