import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import os
import cv2

import LucasKanadeBasis
import LucasKanade

if __name__ == '__main__':
    target_path = '../result/'

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    video = np.load('../data/sylvseq.npy')
    bases = np.load('../data/sylvbases.npy')
    #print(video.shape)
    frame = video[:, :, 0]

    rects = []
    rect_origin = np.array([101, 61, 155, 107])
    rect = np.array([101, 61, 155, 107])
    rects.append(rect)

    for i in range(1, video.shape[2]):
        next_frame = video[:, :, i]

        p_origin = LucasKanade.LucasKanade(frame, next_frame, rect_origin)
        p = LucasKanadeBasis.LucasKanadeBasis(frame, next_frame, rect, bases)
        rect_origin = [rect_origin[0]+p_origin[0], rect_origin[1]+p_origin[1], rect_origin[2]+p_origin[0], rect_origin[3]+p_origin[1]]
        rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
        # print('p',p)
        # print('p_o', p_origin)

        # print('rect_o', rect_origin)
        # print('rect', rect)

        rects.append(rect)

        img = np.zeros((next_frame.shape[0], next_frame.shape[1], 3))
        img[:, :, 0] = next_frame
        img[:, :, 1] = next_frame
        img[:, :, 2] = next_frame
        cv2.rectangle(img, (int(round(rect_origin[0])), int(round(rect_origin[1]))), (int(round(rect_origin[2])), int(round(rect_origin[3]))), color=(0,255,0), thickness=2)
        cv2.rectangle(img, (int(round(rect[0])), int(round(rect[1]))), (int(round(rect[2])), int(round(rect[3]))), color=(0,255,255), thickness=1)
        cv2.imshow('image', img)
        cv2.waitKey(1)

        if i in [1, 200, 300, 350, 400]:
            cv2.imwrite(os.path.join(target_path, 'q2_3_{}.jpg'.format(i)), img*255)

        frame = next_frame

        # break

    rects = np.array(rects)
    #print(rect_list)
    np.save(os.path.join('sylvseqrects.npy'), rects)