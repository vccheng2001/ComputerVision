import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from  SubtractDominantMotion import *
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')





h, w, num_frames = seq.shape





h, w, num_frames = seq.shape


for i in range(num_frames-1):
    print(f'Frame {i}')

    # track optical flow from frame to frame
    template = seq[:,:,i]
    img = seq[:,:,i+1]# image at t+1
    # result = SubtractDominantMotion(template, img, threshold, num_iters, tolerance)    

    if i % 30 == 0:
        result = SubtractDominantMotion(template, img, threshold, num_iters, tolerance)    
        cv2.imwrite(f'aerial-frame{i}.jpg', result)