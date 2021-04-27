'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
K = np.load('../data/intrinsics.npz')
selectedPoints = np.load('../data/templeCoords.npz')
M = max(max(im1.shape),max(im2.shape))

F = sub.eightpoint(data['pts1'], data['pts2'], M)

E = sub.essentialMatrix(F, K['K1'], K['K2'])

x1, y1 = selectedPoints['x1'][:, 0], selectedPoints['y1'][:, 0]
matchPoint, matchPointPrime = [], []
for i in range(x1.shape[0]):
    x2, y2 = sub.epipolarCorrespondence(im1, im2, F, x1[i], y1[i])
    matchPoint.append([x1[i], y1[i]])
    matchPointPrime.append([x2, y2])

M2s = helper.camera2(E)

M1 = np.hstack((np.identity(3) , np.array([[0], [0], [0]])))
C1 = np.dot(K['K1'], M1)

M2, P = None, None
for ind in range(M2s.shape[-1]):
    tempM2 = M2s[:, :, ind]
    p, err = sub.triangulate(C1, np.array(matchPoint), np.dot(K['K2'], tempM2), np.array(matchPointPrime))
    if np.min(p[:, -1]) > 0:
        M2 = tempM2
        P = p
        break

C2 = np.dot(K['K2'], M2)

print('F', F)
print('M1', M1)
print('M2', M2)
print('C1', C1)
print('C2', C2)
print(np.array(matchPoint).shape)
print(np.array(matchPointPrime).shape)
pt1 = np.array(matchPoint)
pt2 = np.array(matchPointPrime)


np.savez('../data/q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='.')

#M2, P2 = sub.bundleAdjustment(K['K1'], M1, pt1, K['K2'], M2, pt2, P)
#C2 = np.dot(K['K2'], M2)
#P3, err = sub.triangulate(C1, pt1, C2, pt2)
#print('err', err)
#ax.scatter(P2[:, 0], P2[:, 1], P2[:, 2], c='r', marker='.')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()