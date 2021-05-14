import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    th = 0.0001
    B = []
    for i in range(bases.shape[2]):
        B.append(bases[:, :, i].flatten())
    B = np.transpose(np.array(B))
    I = np.diag([1]*B.shape[0])
    B_null = I - np.dot(B, np.transpose(B))
    x_min, y_min, x_max, y_max = rect[0], rect[1], rect[2], rect[3]
    delta_p = np.array([3211, 3211])
    p = np.zeros(2)

    spline_It1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    spline_It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    while np.sum(delta_p ** 2) >= th:
        current_x_min, current_x_max = x_min + p[0], x_max + p[0]
        current_y_min, current_y_max = y_min + p[1], y_max + p[1]
        
        current_x_vector = np.arange(current_x_min, current_x_max+1e-9)
        current_y_vector = np.arange(current_y_min, current_y_max+1e-9)
        
        current_x_stack, current_y_stack = np.meshgrid(current_x_vector,current_y_vector)
        
        current_image_gradient_x = spline_It1.ev(current_y_stack, current_x_stack, dx=0, dy=1).flatten()
        current_image_gradient_y = spline_It1.ev(current_y_stack, current_x_stack, dx=1, dy=0).flatten()
        
        N = current_image_gradient_x.shape[0]
        A = np.zeros((N, 2))
        A[:, 0] = current_image_gradient_x
        A[:, 1] = current_image_gradient_y

        current_image_intensity = spline_It1.ev(current_y_stack, current_x_stack)

        template_x_vector = np.arange(x_min, x_max + 1e-9)
        template_y_vector = np.arange(y_min, y_max + 1e-9)
        template_x_stack, template_y_stack = np.meshgrid(template_x_vector, template_y_vector)

        template_image_intensity = spline_It.ev(template_y_stack, template_x_stack)

        A = np.dot(B_null, A)

        b = template_image_intensity.flatten() - current_image_intensity.flatten()
        b = np.expand_dims(b, axis=1)
        b_ = np.dot(B_null, b)
        inv_H = np.linalg.inv(np.dot(np.transpose(A), A))
        delta_p = np.dot(inv_H, np.dot(np.transpose(A), b_))

        p += delta_p.flatten()

    return p

    
