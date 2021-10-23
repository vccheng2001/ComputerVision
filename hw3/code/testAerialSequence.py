import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from  SubtractDominantMotion import *
import time
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()

# NOTE: USE THESE params for inv comp lk
# parser.add_argument('--num_iters', type=int, default=5000, help='number of iterations of Lucas-Kanade')
# parser.add_argument('--threshold', type=float, default=1, help='dp threshold of Lucas-Kanade for terminating optimization')
# parser.add_argument('--tolerance', type=float, default=0.25, help='binary threshold of intensity difference when computing the mask')

# NOTE: USE THESE params for forward lk
parser.add_argument('--num_iters', type=int, default=1000, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=10, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.15, help='binary threshold of intensity difference when computing the mask')

args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')


h, w, num_frames = seq.shape

for i in range(1, num_frames-1):
    
    start = time.time()


    # track optical flow from frame to frame
    template = seq[:,:,i-1]
    img = seq[:,:,i]# image at t+1
    # result = SubtractDominantMotion(template, img, threshold, num_iters, tolerance)    

    end = time.time()
    print(f'frame {i}, time:', end - start)

    if i == 1 or i % 30 == 0:
        result = SubtractDominantMotion(template, img, threshold, num_iters, tolerance)    
        cv2.imwrite(f'aerial-frame{i}.jpg', result)
