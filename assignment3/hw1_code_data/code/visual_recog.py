import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import multiprocessing
import skimage

def get_feature_mp(args):
	image_path, label, dictionary, SPM_layer_num = args
	feature = get_image_feature(image_path, dictionary, SPM_layer_num, dictionary.shape[0])
	return feature, label

def build_recognition_system(num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

			[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''
	train_data = np.load("../data/train_data.npz", allow_pickle=True)
	dictionary = np.load("dictionary.npy", allow_pickle=True)
	# ----- TODO -----
	SPM_layer_num = 2

	image_path = [os.path.join('../data', image_name[0]) for image_name in train_data['image_names']]
	args = zip(image_path, train_data['labels'], [dictionary for _ in image_path], [SPM_layer_num for _ in image_path])

	pool = multiprocessing.Pool(processes=num_workers)
	results = pool.map(get_feature_mp, args)

	features = np.array([result[0] for result in results])
	labels = np.array([result[1] for result in results])

	np.savez('trained_system.npz', features=features, labels=labels,
				dictionary=dictionary, SPM_layer_num=SPM_layer_num)

def evaluate_recognition_system(num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''
	test_data = np.load("../data/test_data.npz", allow_pickle=True)
	trained_system = np.load("trained_system.npz", allow_pickle=True)
	# ----- TODO -----

	test_image_path = [os.path.join('../data', item[0]) for item in test_data['image_names']]

	pool = multiprocessing.Pool(processes=num_workers)
	args = zip(test_image_path, test_data['labels'], [trained_system['dictionary'] for _ in test_image_path],
			[trained_system['SPM_layer_num'] for _ in test_image_path])
	test_result = pool.map(get_feature_mp, args)
	test_features = [result[0] for result in test_result]
	test_labels = [result[1] for result in test_result]

	class_num = max(len(set(test_labels)), len(set(trained_system['labels'])))
	conf = np.zeros((class_num, class_num))
	for i, feature in enumerate(test_features):
		sim = distance_to_set(feature, trained_system['features'])
		[index] = np.where(sim == np.max(sim))[0]
		predict_label = trained_system['labels'][index]
		true_label = test_labels[i]
		conf[true_label, predict_label] += 1

	accuracy = np.diag(conf).sum()/conf.sum()

	return conf, accuracy




def get_image_feature(file_path,dictionary,layer_num,K):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K)
	'''
	# ----- TODO -----
	image = (skimage.io.imread(file_path)).astype('float') / 255
	wordmap = visual_words.get_visual_words(image, dictionary)
	feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
	return feature



def distance_to_set(word_hist,histograms):
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''

	# ----- TODO -----
	intersection = np.minimum(word_hist, histograms)
	return np.sum(intersection, axis=1)



def get_feature_from_wordmap(wordmap,dict_size):
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''
	
	# ----- TODO -----
	hist, _ = np.histogram(wordmap.flatten(), bins=range(dict_size+1), density=True)
	return hist / np.sum(hist)



def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''
	
	# ----- TODO -----
	num_cell = int(2**layer_num)
	cell_h, cell_w = int(wordmap.shape[0]//num_cell), int(wordmap.shape[1]//num_cell)

	hist_all = np.empty((0,), np.float)

	weight = 1/2
	cell_hist_arr = np.zeros((num_cell, num_cell, dict_size))
	for row_id in range(num_cell):
		for col_id in range(num_cell):
			cell = wordmap[row_id*cell_h:(row_id+1)*cell_h, col_id*cell_w:(col_id+1)*cell_w]
			cell_hist_arr[row_id, col_id, :] = get_feature_from_wordmap(cell, dict_size)
	hist_all = np.append((cell_hist_arr*weight).flatten(), hist_all)

	pre_layer_hist = np.copy(cell_hist_arr)
	for l in range(layer_num-1, -1, -1):
		num_cell //= 2
		weight /= (2 if l != 0 else 1)
		layer_hist = np.zeros((num_cell, num_cell, dict_size))
		for row_id in range(num_cell):
			for col_id in range(num_cell):
				layer_hist[row_id, col_id, :] = np.sum(pre_layer_hist[row_id*2:(row_id+1)*2,
													col_id*2:(col_id+1)*2, :], axis=(0, 1))

		hist_all = np.append((layer_hist*weight).flatten(), hist_all)
		pre_layer_hist = layer_hist

	hist_all = hist_all/np.sum(hist_all)

	return hist_all






	

