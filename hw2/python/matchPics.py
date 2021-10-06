import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from helper import plotMatches



def swap(x):
    return x[:, [1,0]]

def matchPics(I1, I2, opts, plot=False):
	#I1, I2 : Images to match
	#opts: input opts
	ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
	sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
	

	#Convert Images to GrayScale

	I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

	cv2.imwrite("I1_gray.jpg", I1_gray)

	I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

	cv2.imwrite("I2_gray.jpg", I2_gray)

	
	#Detect Corner Peaks using FAST algorithm
	
	locs1 = corner_detection(I1_gray, sigma)
	locs2 = corner_detection(I2_gray, sigma)

	# Obtain descriptors for the computed feature locations
	# desc: BRIEF features 
	desc1, locs1 = computeBrief(I1_gray, locs1)
	desc2, locs2 = computeBrief(I2_gray, locs2)


	#Match features using the descriptors
	matches = briefMatch(desc1, desc2, ratio)

	if plot:
		plotMatches(I1_gray, I2_gray,matches,locs1,locs2)
	

	locs1 = swap(locs1)
	locs2 = swap(locs2)
	# unordered locs1, unordered locs2
	return matches, locs1, locs2
