import cv2
import numpy as np
from planarH import computeH
import matplotlib.pyplot as plt


W = np.array([  [0.0, 18.2, 18.2, 0.0], 
                [0.0, 0.0, 26.0, 26.0]])

D = np.array([  [483, 1704, 2175, 67], 
                [810, 781, 2217, 2286]])

K = np.array([[3043.72, 0.0,      1196.00], 
              [0.0,     3043.72,  1604.00], 
              [0.0,     0.0,      1.0]])

H = computeH(D, W)

target = np.array([[833],[1642],[1]])

def compute_extrinsics(K, H):
    Kinv = np.linalg.inv(K)
    KH = np.dot(Kinv, H)
    u, s, vh = np.linalg.svd(KH[:, :2])
    R = np.dot(np.dot(u, np.array([[1, 0], [0, 1], [0, 0]])), vh)
    Rt = np.cross(R[:, 0], R[:, 1], axis=0)
    Rt /= np.sum(Rt**2)
    R = np.hstack((R, np.transpose([Rt])))
    t = KH[:, 2]
    t /= np.sum(KH[:, :2]/R[:, :2])/6
    t = np.transpose([t])
    return R, t

def project_extrinsics(K, W, R, t):
    X = np.dot(np.dot(K, np.hstack((R, t))), W)
    X = X/X[-1, :]
    Wz = W[2, :]
    min_Wz = np.transpose([X[:, np.where(Wz == np.min(Wz))[0][0]]])
    target_translation = target-min_Wz
    target_translation[2] = 1
    T = np.hstack((np.array([[1, 0], [0, 1], [0, 0]]), target_translation))
    X = np.dot(T, X)
    X /= X[-1, :]
    return X

if __name__ == '__main__':

    
    R, t = compute_extrinsics(K, H)
    print("R :", R)
    print("t :", t)

    W = []
    for i in open('../data/sphere.txt', 'r').readlines():
        coordinate = [a for a in i.strip('\n').split(' ') if a != '']
        coordinate = list(map(eval, coordinate))
        W.append(coordinate)
    W = np.vstack((W, np.ones((1, len(W[0])))))

    im = cv2.imread('../data/prince_book.jpeg')

    X = project_extrinsics(K, W, R, t)
    fig = plt.figure('Sphere Projection')
    plt.imshow(im)
    plt.plot(X[0, :], X[1, :], 'y.', linewidth=1, markersize=1)
    plt.draw()
    plt.savefig('../results/ar.jpg')
    plt.waitforbuttonpress(0)
    plt.close(fig)
