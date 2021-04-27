"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
import sympy as sp
import scipy

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation

    x, y, x_prime, y_prime = pts1[:, 0]/M, pts1[:, 1]/M, pts2[:, 0]/M, pts2[:, 1]/M

    A = np.transpose(np.vstack((x*x_prime, x*y_prime, x, y*x_prime, y*y_prime, y, x_prime, y_prime, np.ones(x.shape))))
    
    u, s, vh = np.linalg.svd(A)

    F = np.reshape(vh[-1, :], (3, 3))

    F = helper.refineF(F, pts1/M, pts2/M)

    T_prime = T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    F = np.dot(np.transpose(T_prime), np.dot(F, T))

    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation

    x_prime, y_prime, x, y = pts1[:, 0]/M, pts1[:, 1]/M, pts2[:, 0]/M, pts2[:, 1]/M

    A = np.transpose(np.vstack((x*x_prime, x*y_prime, x, y*x_prime, y*y_prime, y, x_prime, y_prime, np.ones(x.shape))))
    #print('A', A)
    u, s, vh = np.linalg.svd(A)
    w = np.reshape(vh[7, :], (3, 3))
    w = helper.refineF(w, pts1/M, pts2/M)
    v = np.reshape(vh[8, :], (3, 3))
    v = helper.refineF(v, pts1/M, pts2/M)

    detF = lambda alpha: np.linalg.det(alpha * v + (1 - alpha) * w)

    a0 = detF(0)
    a1 = (detF(1) - detF(-1))/3-(detF(2)-detF(-2))/12
    a2 = 0.5*detF(1) + 0.5*detF(-1) - detF(0)
    a3 = (detF(1) - detF(-1))/6 + (detF(2) - detF(-2))/12


    alpha = np.roots([a3, a2, a1, a0])

    T_prime = T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])

    Farray = [a*v+(1-a)*w for a in alpha]

    Farray = [helper.refineF(F, pts1/M, pts2/M) for F in Farray]

    Farray = [np.dot(np.transpose(T_prime), np.dot(F, T)) for F in Farray]

    return Farray


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    return np.dot(np.transpose(K2), np.dot(F, K1))


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    x, y, x_prime, y_prime = pts1[:, 0], pts1[:, 1], pts2[:, 0], pts2[:, 1]

    p1t, p2t, p3t = C1[0,:], C1[1,:], C1[2,:]
    p1_prime_t, p2_prime_t, p3_prime_t = C2[0,:], C2[1,:], C2[2,:]

    A1 = y * p3t[:,np.newaxis]- np.ones_like(x) * p2t[:,np.newaxis] 
    A2 = np.ones_like(x) * p1t[:,np.newaxis] - x * p3t[:,np.newaxis]
    A3 = y_prime * p3_prime_t[:,np.newaxis] -  np.ones_like(x) * p2_prime_t[:,np.newaxis]
    A4 = np.ones_like(x) * p1_prime_t[:,np.newaxis] - x_prime * p3_prime_t[:,np.newaxis]

    N = pts1.shape[0]
    P = np.zeros((N, 3))
    for ind in range(N):
        A = np.vstack((A1[:, ind], A2[:, ind], A3[:, ind], A4[:, ind]))
        u, s, vh = np.linalg.svd(A)
        p = vh[-1, :]
        P[ind, :] = p[:3]/p[-1]

    W = np.hstack((P, np.ones((N, 1))))
    err = 0
    for i in range(N):
        proj1 = np.dot(C1, np.transpose(W[i, :]))
        proj2 = np.dot(C2, np.transpose(W[i, :]))
        proj1 = np.transpose(proj1[:2]/proj1[-1])
        proj2 = np.transpose(proj2[:2]/proj2[-1])
        err += np.sum((proj1-pts1[i])**2 + (proj2-pts2[i])**2)

    return P, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    x1, y1 = int(round(x1)), int(round(y1))
    window_size = 11
    center = window_size//2
    sigma = 5
    search_range = 40

    GaussianMask = np.ones((window_size, window_size))*center
    GaussianMask = np.repeat(np.array([range(window_size)]), window_size, axis=0) - GaussianMask
    GaussianMask = np.sqrt(GaussianMask**2+np.transpose(GaussianMask)**2)
    GaussianWeight = np.exp(-0.5*(GaussianMask**2)/(sigma**2))
    GaussianWeight /= np.sum(GaussianWeight)
    GaussianWeight = np.repeat(np.expand_dims(GaussianWeight, axis=2), im1.shape[-1], axis=2)

    epipolarLine = np.dot(F, np.array([[x1], [y1], [1]]))

    patch1 = im1[y1-center:y1+center+1, x1-center:x1+center+1]

    Y = np.array(range(y1-search_range, y1+search_range))
    X = np.round(-(epipolarLine[1]*Y+epipolarLine[2])/epipolarLine[0]).astype(np.int)
    targets = (X >= center) & (X < im2.shape[1] - center) & (Y >= center) & (Y < im2.shape[0] - center)
    X, Y = X[targets], Y[targets]

    min_distance = None
    x2, y2 = None, None
    for i in range(len(X)):
        patch2 = im2[Y[i]-center:Y[i]+center+1, X[i]-center:X[i]+center+1]
        distance = np.sum((patch1-patch2)**2*GaussianWeight)
        if min_distance is None or distance < min_distance:
            min_distance = distance
            x2, y2 = X[i], Y[i]

    return x2, y2

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    N = pts1.shape[0]
    iter = 100
    threshold = 1
    max_inlier = 0
    F = None
    inliers = None

    for i in range(iter):
        indexs = np.random.randint(0, N, (7,))
        F7s = sevenpoint(pts1[indexs, :], pts2[indexs, :], M)

        for F7 in F7s:
            pts1_homo = np.vstack((np.transpose(pts1), np.ones((1, N))))
            epipolarLine = np.dot(F7, pts1_homo)
            epipolarLine = epipolarLine/np.sqrt(np.sum(epipolarLine[:2, :]**2, axis=0))
            pts2_homo = np.vstack((np.transpose(pts2), np.ones((1, N))))
            deviate = abs(np.sum(pts2_homo*epipolarLine, axis=0))

            tempInliers = np.transpose(deviate < threshold)

            if tempInliers[tempInliers].shape[0] > max_inlier:
                max_inlier = tempInliers[tempInliers].shape[0]
                F = F7
                inliers = tempInliers

    print(max_inlier/N)
    return F, inliers

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    theta = np.sqrt(np.sum(r**2))
    n = r/theta if theta != 0 else r
    n_cross = np.array([[0, -n[2, 0], n[1, 0]], [n[2, 0], 0, -n[0, 0]], [-n[1, 0], n[0, 0], 0]])
    n_cross_square = np.dot(n, np.transpose(n)) - np.sum(n**2)*np.identity(3)
    return np.identity(3) + np.sin(theta)*n_cross + (1-np.cos(theta))*n_cross_square

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    A = (R - np.transpose(R))/2
    p = np.array([[A[2, 1]], [A[0, 2]], [A[1, 0]]])
    s = np.sqrt(np.sum(p**2))
    c = (R[0, 0]+R[1, 1]+R[2, 2]-1)/2

    if s == 0. and c == 1.:
        r = np.zeros((3, 1))
        return r
    elif s == 0. and c == -1.:
        tmp = R + np.diag(np.array([1, 1, 1]))
        v = None
        for i in range(3):
            if np.sum(tmp[:, i]) != 0:
                v = tmp[:, i]
                break
        u = v/np.sqrt(np.sum(v**2))
        r = np.reshape(u*np.pi, (3, 1))
        if np.sqrt(np.sum(r**2)) == np.pi and ((r[0, 0] == 0. and r[1, 0] == 0. and r[2, 0] < 0)
                                               or (r[0, 0] == 0. and r[1, 0] < 0) or (r[0, 0] < 0)):
            r = -r
        return r
    else:
        u = p / s
        theta = np.arctan2(np.float(s), np.float(c))
        r = u*theta
        return r

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    P, r2, t2 = x[:-6], x[-6:-3], x[-3:]

    M2 = np.hstack((rodrigues(np.reshape(r2, (3, 1))), np.reshape(t2, (3, 1))))

    P = np.vstack((np.transpose(np.reshape(P, (P.shape[0]//3, 3))), np.ones((1, P.shape[0]//3))))
    p1_hat = np.dot(np.dot(K1, M1), P)
    p1_hat = np.transpose(p1_hat[:2, :]/p1_hat[2, :])
    p2_hat = np.dot(np.dot(K2, M2), P)
    p2_hat = np.transpose(p2_hat[:2, :]/p2_hat[2, :])

    return np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    R2_init, t2_init = M2_init[:, :3], M2_init[:, 3]
    r2_init = invRodrigues(R2_init).reshape([-1])
    x = np.concatenate([P_init.reshape([-1]), r2_init, t2_init])

    func = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x)**2).sum()
    x_update = scipy.optimize.minimize(func, x).x

    P, r2, t2 = x_update[:-6], x_update[-6:-3], x_update[-3:]

    P2 = np.reshape(P, (P.shape[0] // 3, 3))
    M2 = np.hstack((rodrigues(np.reshape(r2, (3, 1))), np.reshape(t2, (3, 1))))
    return M2, P2
