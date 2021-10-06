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


def main():
    # reads images
    cv_cover = cv2.imread("../data/cv_cover.jpg")
    cv_desk = cv2.imread("../data/cv_desk.png")
    hp_cover = cv2.imread("../data/hp_cover.jpg")
    hp_desk = cv2.imread("../data/hp_desk.png")


    hp_cover = cv2.resize(hp_cover, dsize=(cv_cover.shape[1], cv_cover.shape[0]))
    # matches 
    matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts, plot=True)
    # reorder matches
    locs1 = locs1[matches[:, 0], :] # 0th column is the indices
    locs2 = locs2[matches[:, 1], :] # 1st column is the indices

    # compute homography
    bestH, bestInliers = computeH_ransac(locs1, locs2, opts)
    compositeH(bestH,cv_desk,hp_cover)



def compositeH(H, template, img, write=True):
   
    img = cv2.warpPerspective(src=img.astype(np.float64),
                             M= H.astype(np.float64),
                             dsize=(template.shape[1], template.shape[0]))
    # cv2.imwrite("IMG.jpg", img)

    # # resize full image to overlay on bg
    # img = cv2.resize(img, dsize=(template.shape[1], template.shape[0]))
    img = img.astype(np.uint8)
    # to extract bg
    mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV)[1]
   
    # cv2.imwrite("mask_ar.jpg", mask)
    mask = mask.astype(np.uint8)

    # to extract fg
    mask_inv = cv2.bitwise_not(mask).astype(np.uint8)
    # cv2.imwrite("mask_inv_ar.jpg", mask_inv)

    fg = cv2.bitwise_and(img, img, mask=mask_inv)
    # cv2.imwrite("fg_ar.jpg", fg)

    bg = cv2.bitwise_and(template, template, mask=mask)
    # cv2.imwrite("bg_ar.jpg", bg)

    # add 
    composite_img = cv2.add(bg, fg)

    if write:
        cv2.imwrite("composite_img.jpg", composite_img)
    return composite_img


if __name__ == "__main__":
    main()