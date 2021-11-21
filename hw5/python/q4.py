import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import sort

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import cv2
from skimage.color import label2rgb

from skimage.measure import label


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)
    
# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    image = np.array(image * 255, dtype = np.uint8)


    # convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


    # Set the adaptive thresholding (gasussian) parameters:
    windowSize = 31
    windowConstant = -1
    # adaptive threshold 
    img_binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, windowSize, windowConstant)
 
    # area filter 0o
    componentsNumber, labeledImage, componentStats, componentCentroids = \
    cv2.connectedComponentsWithStats(img_binary, connectivity=4)

    minArea = 1800

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    # Set kernel (structuring element) size:
    kernel_size = 3

    struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # closing: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    img_closing = cv2.morphologyEx(filteredImage, cv2.MORPH_CLOSE, struct_elem,  None, None, 1, cv2.BORDER_REFLECT101)

    # find bounding boxes
    contours, hierarchy = cv2.findContours(img_closing, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    


    contours_poly = [None] * len(contours)
    cnts = []

    bboxes = []
    for i, c in enumerate(contours):

        if hierarchy[0][i][3] == -1:
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            cnts.append(contours_poly[i])
            x,y,w,h = cv2.boundingRect(contours_poly[i])
            bbox = y,x,y+h,x+w#x,y,x+w,y+h
            bboxes.append(bbox)


    reverse = False
    i = 0
    (cnts, bboxes) = zip(*sorted(zip(cnts, bboxes),
		key=lambda b:b[1][i], reverse=reverse))


   
    return bboxes, img_gray /255
