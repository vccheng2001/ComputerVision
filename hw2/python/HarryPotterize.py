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
hp_desk = cv2.imread("../data/hp_desk.png")


hp_cover = cv2.resize(hp_cover, dsize=(cv_cover.shape[1], cv_cover.shape[0]))

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


# dsize: (width, height)
warped = cv2.warpPerspective(src=hp_cover.astype(np.float64),
                             M=bestH.astype(np.float64),
                             dsize=(cv_desk.shape[1], cv_desk.shape[0]))
# resize full image to overlay on cv desk
warped = cv2.resize(warped, dsize=(cv_desk.shape[1], cv_desk.shape[0]))
cv2.imwrite("warped.png", warped)
warped = warped.astype(np.uint8)


# If the element at position x,y in the mask
# is 0, no operation is performed, 
# and the pixel in the resulting array is 0,0,0 (black)

# any pixel value at 0 (pitch black) would be 'false'. 
# Since false & anything = false, any other pixel with value above 0 would become 0.

# fg (book cover) white
mask = cv2.threshold(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV)[1]
mask = mask.astype(np.uint8)
cv2.imwrite('mask.png',mask)

# bg (desk) white
mask_inv = cv2.bitwise_not(mask).astype(np.uint8)
cv2.imwrite('maskinv.png',mask_inv)

cv_desk = cv_desk.astype(np.uint8)

# take book 
# warped = warped.astype(np.uint8)
fg = cv2.bitwise_and(warped, warped, mask=mask_inv)
cv2.imwrite('fg.jpg',fg)

# take desk

bg = cv2.bitwise_and(cv_desk,cv_desk, mask=mask)
cv2.imwrite('bg.jpg',bg)

res = cv2.add(bg, fg)

cv2.imwrite("res.jpg", res)
