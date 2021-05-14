import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import os
import cv2
import time
import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':
    target_path = '../result/'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    video = np.load('../data/aerialseq.npy')
    frame = video[:, :, 0]
    t = time.time()

    for i in range(1, video.shape[2]):

        next_frame = video[:, :, i]
        mask = SubtractDominantMotion.SubtractDominantMotion(frame, next_frame)

        img = np.zeros((next_frame.shape[0], next_frame.shape[1], 3))
        img[:, :, 0] = next_frame
        img[:, :, 1] = next_frame
        img[:, :, 2] = next_frame
        img[:, :, 0][mask] = 1
        cv2.imshow('image', img)
        cv2.waitKey(1)

        if i in [30, 60, 90, 120]:
            cv2.imwrite(os.path.join(target_path, 'q3_3_{}.jpg'.format(i)), img*255)
        
        frame = next_frame
        
    rects = np.array(rects)
    #print(rect_list)
    np.save(os.path.join('carseqrects.npy'), rects)
    print(time.time() - t)
    input("Press Enter to Exit.")