'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

M = max(max(im1.shape),max(im2.shape))

F8 = sub.eightpoint(data['pts1'], data['pts2'], M)

K = np.load('../data/intrinsics.npz')
E = sub.essentialMatrix(F8, K['K1'], K['K2'])

M2s = helper.camera2(E)

M1 = np.hstack((np.identity(3) , np.array([[0], [0], [0]])))
C1 = np.dot(K['K1'], M1)

M2, P = None, None
for ind in range(M2s.shape[-1]):
    tempM2 = M2s[:, :, ind]
    p, err = sub.triangulate(C1, data['pts1'], np.dot(K['K2'], tempM2), data['pts2'])
    if np.min(p[:, -1]) > 0:
        M2 = tempM2
        P = p
        break

C2 = np.dot(K['K2'], M2)
print('M2', M2)
print('C2', C2)
np.savez('../data/q3_3.npz', M2 = M2, C2 = C2, P = P)
