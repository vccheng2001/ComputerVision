import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import *
import cv2
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=0.001, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

# (h=240,w=320,indices=415)
# first frame: imshow(frames[:,:,0])
seq = np.load("../data/carseq.npy")


# cv2.imshow('frame0', seq[:,:,0])
# cv2.waitKey()

# x1,y1,x2,y2
rect = [59, 116, 145, 151]

h, w, num_frames = seq.shape
print('hwf', h,w,num_frames) # 415 frames 
rects = []

# store initial rect


for i in range(1, num_frames-1):

    

    print(f'Frame {i}')

    #  seq: (240,320,415)
    # track optical flow from frame to frame
    template = seq[:,:,i-1]*255
    img = seq[:,:,i]*255

    # run LK on each frame pair
    p = LucasKanade(template, img, rect, threshold, num_iters, p0=np.zeros(2))
    rect[0]  += p[0] # !!!! add to original rect!
    rect[1]  +=  p[1]
    rect[2]  += p[0]
    rect[3]  += + p[1]

    rects.append(rect)

    if i == 1 or i % 100 == 0:

        


        x1,y1,x2,y2 = map(int, rect)

        img_with_rect = cv2.rectangle(img=img.copy(),
                                     pt1=(x1,y1), 
                                     pt2=(x2,y2), 
                                     color=(255,0,0), 
                                     thickness=2)
        # cv2.imshow('img_with_rect',img)
        cv2.imwrite(f'car_frame{i}.jpg', img_with_rect)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


# save all rects as Nx4 matrix
out = 'carseqrects.npy'
with open('carseqrects.npy', 'wb') as f:
    print('Saving.....')
    np.save(f, rects)



# load rects only
# rects = np.load('carseqrects.npy')

# for i in range(num_frames-1):
#     if i % 10 == 0:

#         # track optical flow from frame to frame
#         img = seq[:,:,i+1] # image at t+1
#         x1,y1,x2,y2 = map(int, rects[i])
#         img_with_rect= cv2.rectangle(img=img.copy(), 
#                                         pt1=(x1,y1), 
#                                         pt2=(x2,y2), 
#                                         color=(255,0,0), 
#                                         thickness=3)
#         cv2.imshow('img_with_rect',img_with_rect)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

