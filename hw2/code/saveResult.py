'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

M = max(max(im1.shape),max(im2.shape))
#print('M', M)
data = np.load('../data/some_corresp.npz')

#Q2.1
F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
#print('F8', F8)
# save matrix F8, scale M
np.savez('../data/q2_1', F8 = F8, M = M)

#Q2.2
F7 = sub.sevenpoint(data['pts1'][:7, :], data['pts2'][:7, :], M)
#print('F7', F7)
# save matrix F7, scale M, 2D points pts1, pts2
np.savez('../data/q2_2', F7 = F7, M = M, pts1 = data['pts1'][:7, :], pts2 = data['pts2'][:7, :])

#Q3.1
K = np.load('../data/intrinsics.npz')
E = sub.essentialMatrix(F8, K['K1'], K['K2'])
#print('E', E)
M2s = helper.camera2(E)
#print('M2s', M2s[:,:,0], M2s[:,:,1], M2s[:,:,2], M2s[:,:,3])

#Q3.2
M1 = np.hstack((np.identity(3) , np.array([[0], [0], [0]])))
C1 = np.dot(K['K1'], M1)
M2 = M2s[:,:,0]
C2 = np.dot(K['K2'], M2)
P, err = sub.triangulate(C1, data['pts1'], C2, data['pts2'])
print('err',err)

#Q4.1
#helper.epipolarMatchGUI(im1,im2,F8)
points = np.array([[[520, 231], [520, 201]], \
                    [[454, 118], [447, 122]], \
                    [[413, 206], [411, 188]], \
                    [[234, 255], [234, 252]], \
                    [[170, 333], [171, 339]], \
                    [[153, 133], [152, 136]], \
                    [[121, 210], [121, 182]], \
                    [[64, 137], [65, 124]], \
                    [[56, 184], [56, 177]]])

pts1 = points[:,0]
pts2 = points[:,1]
np.savez('../data/q4_1', F = F8, pts1 = pts1, pts2 = pts2)

#Q5
noisy = np.load('../data/some_corresp_noisy.npz')
#sub.ransacF(noisy['pts1'], noisy['pts2'], M)

M2, P2 = sub.bundleAdjustment(K['K1'], M1, data['pts1'], K['K2'], M2, data['pts2'], P)

C2 = np.dot(K['K2'], M2)
P3, err = sub.triangulate(C1, data['pts1'], C2, data['pts2'])
print('err', err)
# P, err = sub.triangulate(C1, data['pts1'], C2, data['pts2'])
# print('err',err)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='.')
#ax.scatter(P3[:, 0], P3[:, 1], P3[:, 2], c='g', marker='.')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()