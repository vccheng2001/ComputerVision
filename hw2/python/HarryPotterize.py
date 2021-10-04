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
print('opts', opts)

# reads images
cv_cover = cv2.imread("../data/cv_cover.jpg")
cv_desk = cv2.imread("../data/cv_desk.png")
hp_cover = cv2.imread("../data/hp_cover.jpg")

# computes homography
# [x1,y1]   [x2,y2]
matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts, plot=True)
print('matches', matches.shape)

# print('matches0', matches[0]) # index: 722, 431
# print(locs1[matches[0][0]]) #xy=281,13
# print('\n')
# print(locs2[matches[0][1]]) #xy=334,521
locs1 = locs1[matches[:, 0], :] # 0th column is the indices
locs2 = locs2[matches[:, 1], :] # 1st column is the indices
# print('locs1', locs1[0])
# print('locs2', locs2[0])

# locs1, locs2: Nx2 matrices containing x, y coords of matched point pairs
bestH, bestInliers = computeH_ransac(locs1, locs2, opts)
# print('BEST H', bestH)
# print('BEST INLIERS', bestInliers)
# warp 

warped = cv2.warpPerspective(src=hp_cover.astype(np.float64),
                             M=bestH.astype(np.float64),
                             dsize=(cv_desk.shape[0], cv_desk.shape[1]))
cv2.imshow("warped", warped)
cv2.imwrite("warpedhpcvhpcvdesk.png", warped)
# modify hp_cover.jpg? 

