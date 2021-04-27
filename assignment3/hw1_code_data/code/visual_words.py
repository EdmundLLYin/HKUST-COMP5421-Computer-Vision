import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random
import shutil

tmp_dir = '../tmp'

def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	
	# ----- TODO -----
	h, w = image.shape[0], image.shape[1]

	if len(image.shape) == 2:
		image = np.matlib.repmat(image, 3, 1)
	else:
		if image.shape[-1] == 4:
			image = image[:, :, :-1]

	lab_color = skimage.color.rgb2lab(image)

	scales = [1, 2, 4, 8, 8*(2**0.5)]
	filter_responses = np.zeros((h, w, 3*4*len(scales)))
	for ind, scale in enumerate(scales):
		for i in range(3):
			filter_responses[:, :, ind*12+i] = scipy.ndimage.gaussian_filter(lab_color[:, :, i], scale)
			filter_responses[:, :, ind*12+3+i] = scipy.ndimage.gaussian_laplace(lab_color[:, :, i], scale)
			filter_responses[:, :, ind*12+6+i] = scipy.ndimage.gaussian_filter(lab_color[:, :, i], scale, [0, 1])
			filter_responses[:, :, ind*12+9+i] = scipy.ndimage.gaussian_filter(lab_color[:, :, i], scale, [1, 0])

	return filter_responses

def get_visual_words(image,dictionary):
	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	
	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''
	
	# ----- TODO -----
	filter_response = extract_filter_responses(image)

	wordmap = np.zeros(filter_response.shape[:2])
	for i in range(filter_response.shape[0]):
		for j in range(filter_response.shape[1]):
			pixel_response = filter_response[i, j, :]
			dist = scipy.spatial.distance.cdist(np.array([pixel_response]), dictionary, 'euclidean')
			[best] = np.where(dist == np.min(dist))[1]
			wordmap[i, j] = best

	return wordmap


def compute_dictionary_one_image(args):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* i: index of training image
	* alpha: number of random samples
	* image_path: path of image file
	* time_start: time stamp of start time

	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''

	i,alpha,image_path = args
	# ----- TODO -----
	image = (skimage.io.imread(image_path)).astype('float') / 255
	filter_response = extract_filter_responses(image)

	pixel_X = np.random.randint(0, filter_response.shape[0], size=alpha)
	pixel_Y = np.random.randint(0, filter_response.shape[1], size=alpha)
	sampled_response = np.zeros((alpha, filter_response.shape[2]))

	sampled_response[:, :] = filter_response[pixel_X[:], pixel_Y[:], :]

	np.save(os.path.join(tmp_dir, str(i)+'.npy'), sampled_response)


def compute_dictionary(num_workers=2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel
	
	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''

	train_data = np.load("../data/train_data.npz", allow_pickle = True)
	# ----- TODO -----
	K, alpha = 100, 150
	# acc 0.6026 K = 100, alpha = 200
	# acc 0.50625 K = 200 alpha = 250
	# acc 0.50625 K = 150 alpha = 150
	# acc 0.53125 K = 100 alpha = 250
	# acc 0.55625 K = 100 alpha = 150

	if os.path.exists(tmp_dir):
		shutil.rmtree(tmp_dir)
	os.makedirs(tmp_dir)

	image_paths = [os.path.join('../data', item[0]) for item in train_data['image_names']]

	pool = multiprocessing.Pool(processes=num_workers)
	args = zip(range(len(image_paths)), [alpha for _ in image_paths], image_paths)
	pool.map(compute_dictionary_one_image, args)

	filter_responses = np.array([])
	for file in os.listdir(tmp_dir):
		sampled_response = np.load(os.path.join(tmp_dir, file))
		filter_responses = np.array(np.append(filter_responses, sampled_response, axis=0)
									if filter_responses.shape[0] != 0 else sampled_response)

	kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=num_workers).fit(filter_responses)
	dictionary = kmeans.cluster_centers_

	np.save('dictionary.npy', dictionary)

	shutil.rmtree(tmp_dir)


