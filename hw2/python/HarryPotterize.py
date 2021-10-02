import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac
#Import necessary functions



#Write script for Q2.2.4
opts = get_opts()

# reads images
cv_cover = cv2.imread("../data/cv_cover.jpg")
cv_desk = cv2.imread("../data/cv_desk.png")
hp_cover = cv2.imread("../data/hp_cover.jpg")

# computes homography
matches, locs1, locs2 = matchPics(hp_cover, hp_cover, opts, plot=True)

# locs1, locs2: Nx2 matrices containing x, y coords of matched point pairs
bestH, bestInliers = computeH_ransac(locs1, locs2, opts)
print('BEST H', bestH)
print('BEST INLIERS', bestInliers)
# warp 

warped = cv2.warpPerspective(src=hp_cover.astype(np.float64),
                             M=bestH.astype(np.float64),
                             dsize=(hp_cover.shape[0], hp_cover.shape[1]))
# cv2.imshow("warped", warped)
cv2.imwrite("warpedhpcvhpcvdesk.png", warped)
print('doneeeeeee')
# modify hp_cover.jpg? 

