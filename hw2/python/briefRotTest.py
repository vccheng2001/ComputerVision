import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
from scipy import ndimage, misc
from matplotlib import pyplot as plt 

opts = get_opts()

orig_img = cv2.imread('../data/cv_cover.jpg')

img = orig_img
rot_arr, hist_arr = [], []


for i in range(36):
	# Rotate Image 10 degrees at a time

	img = ndimage.rotate(img, 10, reshape=False)
	
	# Compute features, descriptors and Match features
	deg = (i+1)*10 # degree of roation from original image
	matches, locs1, locs2 = matchPics(img, orig_img, opts, plot=False)
	num_matches = len(matches)

	hist_arr.append(num_matches)
	rot_arr.append(deg)

#Update histogram
plt.bar(rot_arr, hist_arr)
plt.title("Orientation vs Number of matches")
plt.show()