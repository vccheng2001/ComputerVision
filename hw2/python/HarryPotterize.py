import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import computeH
#Import necessary functions



#Write script for Q2.2.4
opts = get_opts()

# reads images
cv_cover = cv2.imread("../data/cv_cover.jpg")
cv_desk = cv2.imread("../data/cv_desk.png")
hp_cover = cv2.imread("../data/hp_cover.jpg")

# computes homography
matches, locs1, locs2 = matchPics(hp_cover, cv_desk, opts)
# locs1, locs2: Nx2 matrices containing x, y coords of matched point pairs
H2to1 = computeH(locs1, locs2)
print('h', H2to1)
# warp 
warped = cv2.warpPerspective(hp_cover, M=H2to1, dsize=(cv_desk.shape[0], cv_desk.shape[1]))
cv2.imshow("warped", warped)
# modify hp_cover.jpg? 

