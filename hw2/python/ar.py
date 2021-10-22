from inspect import FrameInfo
import numpy as np
import cv2
import os
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac
from HarryPotterize import compositeH
#Import necessary functions
#Write script for Q2.2.4
opts = get_opts()




# Returns aspect ratio (Height / Width)
def get_ar(x):
    return x.shape[0]/x.shape[1] 


def main():

    count = 0

    cap1 = cv2.VideoCapture("../data/ar_source.mov") # src
    cap2 =cv2.VideoCapture("../data/book.mov")       # dest
    book_cover = cv2.imread("../data/cv_cover.jpg")
    


    if (cap1.isOpened()== False): 
        print("Error opening video stream or file")
    if (cap2.isOpened()== False): 
        print("Error opening video stream or file")

    while(cap1.isOpened() or cap2.isOpened()):
        count += 1
        # Capture frame-by-frame
        ret1, src_frame = cap1.read() # kungfu panda
        ret2, dst_frame = cap2.read() # books

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # make src frame same AR as book cover
        book_cover_ar = get_ar(book_cover) # H/W desired aspect ratio
        h = src_frame.shape[0] - 100 # crop off black borders
        w = int(1/book_cover_ar * h) # (w/h) * h

        centerx = src_frame.shape[1] / 2
        centery = src_frame.shape[0] / 2
        x = int(centerx - w/2)
        y = int(centery - h/2)

        # crop 
        src_frame = src_frame[y:y+h, x:x+w, :]

        
        # resize src to be same as book cover so that warp works out
        src_frame = cv2.resize(src_frame, dsize=(book_cover.shape[1], book_cover.shape[0]))
    
        # match book cover to the book in dest vid
        matches, locs1, locs2 = matchPics(dst_frame,book_cover, opts, plot=False)
        # reorder matches
        locs1 = locs1[matches[:, 0], :] 
        locs2 = locs2[matches[:, 1], :] 

        # compute homography
        bestH, bestInliers = computeH_ransac(locs1, locs2, opts)

        # overlay
        if count % 2 == 0:
            composite_img = compositeH(bestH,dst_frame, src_frame, write=False)
            cv2.imwrite(f"frames{count}.jpg", composite_img)

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()