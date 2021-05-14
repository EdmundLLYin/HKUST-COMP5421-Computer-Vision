import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib.patches as patches
import scipy.ndimage
import cv2
import os

img = np.load('lena.npy')

# template cornes in image space [[x1, x2, x3, x4], [y1, y2, y3, y4]]
pts = np.array([[248, 292, 248, 292],
                [252, 252, 280, 280]])

# size of the template (h, w)
dsize = np.array([pts[1, 3] - pts[1, 0] + 1,
                  pts[0, 1] - pts[0, 0] + 1])

# set template corners
tmplt_pts = np.array([[0, dsize[1]-1, 0, dsize[1], -1],
                      [0, 0, dsize[0] - 1, dsize[0] - 1]])


# apply warp p to template region of img
def imwarp(p):
    global img, dsize
    return img[p[1]:(p[1]+dsize[0]), p[0]:(p[0]+dsize[1])]


# get positive example
gnd_p = np.array([252, 248])  # ground truth warp
x = imwarp(gnd_p)  # the template

# stet up figure
fig, axarr = plt.subplots(1, 3)
axarr[0].imshow(img, cmap=plt.get_cmap('gray'))
patch = patches.Rectangle((gnd_p[0], gnd_p[1]), dsize[1], dsize[0],
                          linewidth=1, edgecolor='r', facecolor='none')
axarr[0].add_patch(patch)
axarr[0].set_title('Image')

cropax = axarr[1].imshow(x, cmap=plt.get_cmap('gray'))
axarr[1].set_title('Cropped Image')

dx = np.arange(-np.floor(dsize[1]/2), np.floor(dsize[1]/2)+1, dtype=int)
dy = np.arange(-np.floor(dsize[0]/2), np.floor(dsize[0]/2)+1, dtype=int)
[dpx, dpy] = np.meshgrid(dx, dy)
dpx = dpx.reshape(-1, 1)
dpy = dpy.reshape(-1, 1)
dp = np.hstack((dpx, dpy))
N = dpx.size

all_patches = np.ones((N*dsize[0], dsize[1]))
all_patchax = axarr[2].imshow(all_patches, cmap=plt.get_cmap('gray'),
                              aspect='auto', norm=colors.NoNorm())
axarr[2].set_title('Concatenation of Sub-Images (X)')

X = np.zeros((N, N))
Y = np.zeros((N, 1))

sigma = 5

def im_show_and_save(image, path):
    cv2.imshow('filter response', image)
    cv2.waitKey(0)  # press any key to exit
    cv2.destroyAllWindows()
    cv2.imwrite(path, image*255)

def init():
    return [cropax, patch, all_patchax]


def animate(i):
    global X, Y, dp, gnd_p, sigma, all_patches, patch, cropax, all_patchax, N

    if i < N:  # If the animation is still running
        xn = imwarp(dp[i, :] + gnd_p)
        X[:, i] = xn.reshape(-1)
        Y[i] = np.exp(-np.dot(dp[i, :], dp[i, :])/sigma)
        all_patches[(i*dsize[0]):((i+1)*dsize[0]), :] = xn
        cropax.set_data(xn)
        all_patchax.set_data(all_patches.copy())
        all_patchax.autoscale()
        patch.set_xy(dp[i, :] + gnd_p)
        return [cropax, patch, all_patchax]
    else:  # Stuff to do after the animation ends
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.plot_surface(dpx.reshape(dsize), dpy.reshape(dsize),
                          Y.reshape(dsize), cmap=plt.get_cmap('coolwarm'))

        # Place your solution code for question 4.3 here
        target_path = '../../result/'
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        lam0 = 0
        S_ = np.dot(X, np.transpose(X) + lam0 * np.eye(X.shape[0]))
        S_ = np.linalg.inv(S_)
        S_ = np.dot(S_, X)
        g0 = np.dot(S_, Y)
        g0 = np.reshape(g0, (29, 45))
        plt.matshow(g0)
        plt.show()

        out = scipy.ndimage.correlate(img, g0)
        im_show_and_save(out, os.path.join(target_path, 'correlate0.jpg'))

        lam1 = 1
        g1 = np.dot(np.dot(np.linalg.inv(np.dot(X, np.transpose(X)) + lam1 * np.eye(X.shape[0])), X), Y)
        g1 = np.reshape(g1, (29, 45))
        plt.matshow(g1)
        plt.show()

        out = scipy.ndimage.correlate(img, g1)
        im_show_and_save(out, os.path.join(target_path, 'correlate1.jpg'))
        out = scipy.ndimage.convolve(img, g0)
        im_show_and_save(out, os.path.join(target_path, 'convolve0.jpg'))
        out = scipy.ndimage.convolve(img, g1)
        im_show_and_save(out, os.path.join(target_path, 'convolve1.jpg'))
        out = scipy.ndimage.convolve(img, g0[::-1, ::-1])
        im_show_and_save(out, os.path.join(target_path, 'convolve2.jpg'))
        out = scipy.ndimage.convolve(img, g1[::-1, ::-1])
        im_show_and_save(out, os.path.join(target_path, 'convolve3.jpg'))

        return []
        


# Start the animation
ani = animation.FuncAnimation(fig, animate, frames=N+1,
                              init_func=init, blit=True,
                              repeat=False, interval=10)
plt.show()
