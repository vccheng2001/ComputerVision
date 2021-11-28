import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import sort

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
from skimage.morphology import square

import skimage.segmentation
import cv2
from skimage.color import label2rgb

from skimage.measure import label
import copy 


# sort contours so that letters are read left to right 
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    # sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # sort against y or x coord
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # sort
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)
    


############################  Q4.2 ############################ 
# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    image = np.array(image * 255, dtype = np.uint8)


    # convert to grayscale
    img_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = 100
    img_bw = cv2.threshold(img_bw, thresh, 255, cv2.THRESH_BINARY)[1]

    img_gray = copy.copy(img_bw)

    # Set the adaptive thresholding (gasussian) parameters:
    windowSize = 15
    windowConstant = -1
    # adaptive threshold 
    img_binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, windowSize, windowConstant)
    
    # cv2.imshow('img_bin', img_binary)
    # cv2.waitKey(0)


    # kernel = np.ones((5,5),np.uint8)
    # img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)


    # img_binary  = skimage.morphology.opening(img_binary, square(8))
    # cv2.imshow('img_open', img_binary)
    # cv2.waitKey(0)



    # # Creating kernel
    # kernel = np.ones((3, 3), np.uint8)
    # # Using cv2.erode() method 
    # img_binary= cv2.erode(img_binary, kernel, iterations=3)
    # # Displaying the image 
    # cv2.imshow('img eroded', img_binary)
    # cv2.waitKey(0)


    
    componentsNumber, labeledImage, componentStats, componentCentroids = \
    cv2.connectedComponentsWithStats(img_binary, connectivity=4)

    minArea = 1500

    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

    # filter labeled pixels 
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    # Set kernel (structuring element) size:
    kernel_size = 3

    struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # # closing: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    img_closing = cv2.morphologyEx(filteredImage, cv2.MORPH_CLOSE, struct_elem,  None, None, 1, cv2.BORDER_REFLECT101)
    # cv2.imshow('imgclose', img_closing)
    # cv2.waitKey(0)

    # find bounding boxes
    contours, hierarchy = cv2.findContours(filteredImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    

    contours_poly = [None] * len(contours)
    cnts = []
    bboxes = []

    for i, c in enumerate(contours):

        if hierarchy[0][i][3] == -1:
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            cnts.append(contours_poly[i])
            x,y,w,h = cv2.boundingRect(contours_poly[i])
            bbox = y,x,y+h,x+w
            bboxes.append(bbox)

        # image = cv2.drawContours(img_gray, cnts, -1, (0, 255, 0), 2)

        # show the image with the drawn contours
        # plt.imshow(image)
        # plt.show()


    # bboxes = sorted(bboxes , key=lambda k: [k[1], k[0]])
    # print(bboxes)



    # reverse =True# change vertical (top->down vs down->top)
    # i = 0
    # (cnts, bboxes) = zip(*sorted(zip(cnts, bboxes),
	# 	key=lambda b:b[1][i], reverse=reverse))



    bboxes.sort(key=lambda x:get_contour_precedence(x, img_gray.shape[1]))

    for b in bboxes:
        y1,x1,y2,x2 = b
        img_with_rect = cv2.rectangle(img_gray, (x1,y1),(x2,y2), (0,255,0), 2)
        cv2.imshow('hi', img_with_rect)
        cv2.waitKey(0)

    # exit(-1)

    cv2.imshow('GRAYi', img_bw)
    cv2.waitKey(0)


    # bboxes: [y1,x1,y2,x2]
    # img_gray: floating point 0->1
   
    return bboxes, img_bw / 255



def get_contour_precedence(bbox, cols):
    tolerance_factor = 190
    y1,x1,y2,x2 = bbox
    return ((y1 // tolerance_factor) * tolerance_factor) * cols + x1

