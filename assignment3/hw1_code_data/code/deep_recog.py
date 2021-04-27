import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import torch
import torch.nn as nn
import torchvision
from scipy.spatial.distance import cdist
import shutil

tmp_dir = '../vgg_tmp'

def build_recognition_system(vgg16, num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''

	train_file = np.load("../data/train_data.npz", allow_pickle=True)

	if os.path.exists(tmp_dir):
		shutil.rmtree(tmp_dir)
	os.makedirs(tmp_dir)

	image_path = [os.path.join('../data', item[0]) for item in train_file['image_names']]

	pool = multiprocessing.Pool(processes=num_workers)
	args = zip(range(len(image_path)), image_path, [vgg16 for _ in image_path])
	pool.map(get_image_feature, args)

	features = [None]*len(image_path)
	for file in os.listdir(tmp_dir):
		feature = np.load(os.path.join(tmp_dir, file))
		index = int(file.split('.')[0])
		features[index] = feature
	features = np.array(features)

	features = np.asarray(features)
	np.savez('trained_system_deep1.npz', features=features, labels=train_file['labels'])

	shutil.rmtree(tmp_dir)


def evaluate_recognition_system(vgg16, num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	test_data = np.load("../data/test_data.npz", allow_pickle=True)
	trained_system = np.load("trained_system_deep1.npz", allow_pickle=True)

	if os.path.exists(tmp_dir):
		shutil.rmtree(tmp_dir)
	if not os.path.exists(tmp_dir):
		os.makedirs(tmp_dir)
	image_path = [os.path.join('../data', item[0]) for item in test_data['image_names']]

	pool = multiprocessing.Pool(processes=num_workers)
	args = zip(range(len(image_path)), image_path, [vgg16 for _ in image_path])
	pool.map(get_image_feature, args)

	features = [None]*len(image_path)
	for file in os.listdir(tmp_dir):
		feature = np.load(os.path.join(tmp_dir, file))
		index = int(file.split('.')[0])
		features[index] = feature
	features = np.array(features)
	
	labels = []
	for i, test_file in enumerate(test_data['image_names']):
		feature = features[i]
		dist = distance_to_set(feature, trained_system['features'])
		classifer = np.argmin(dist)
		labels.append(trained_system['labels'][classifer])

	labels = np.asarray(labels, dtype=int)
	np.save("labels_deep2.npy", labels)
	conf = np.zeros((8,8))

	for i, test in enumerate(test_data['labels']):
		conf[test, labels[i]] += 1

	accuracy = np.trace(conf)/conf.sum()
	np.savez("deep_results2.npz", confusion_matrix=conf, accuracy=accuracy)

	if os.path.exists(tmp_dir):
		shutil.rmtree(tmp_dir)
	return conf, accuracy


def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (1,3,H,W)
	'''
	# ----- TODO -----

	if image.ndim == 2:
		image = np.dstack((image,image,image))
	if image.shape[2] == 4:
		image = np.delete(image, -1, axis=2)

	image = skimage.transform.resize(image, (224,224,3))
	MEAN = [0.485, 0.456, 0.406]
	STD = [0.229, 0.224, 0.225]
	image[:,:,:] = (image[:,:,:] - MEAN[:]) / STD[:]
	image = np.asarray(image).astype(np.float32)
	image_processed = torch.unsqueeze((torch.from_numpy(image)).permute(2,0,1), 0)
	return image_processed


def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
 	[saved]
	* feat: evaluated deep feature
	'''
	i, image_path, vgg16 = args
	vgg16.classifier = nn.Sequential(*[vgg16.classifier[i] for i in range(5)])
	image = skimage.io.imread(image_path)
	image_tensor = preprocess_image(image)
	conv = vgg16.features(image_tensor).flatten()
	feat = vgg16.classifier(conv).detach().numpy()
	np.save(os.path.join(tmp_dir, str(i) + '.npy'), feat)


def distance_to_set(feature, train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''
	feature = feature.reshape(1, len(feature))
	dist = cdist(feature, train_features)[0]
	print(dist)
	return dist
