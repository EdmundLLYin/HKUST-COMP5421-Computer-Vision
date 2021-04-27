import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage.io
import network_layers
import torch
import multiprocessing

if __name__ == '__main__':

	num_cores = util.get_num_CPU()
	multiprocessing.set_start_method('spawn')

	# path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
	# image = skimage.io.imread(path_img)
	# image = image.astype('float')/255
	# filter_responses = visual_words.extract_filter_responses(image)
	# util.display_filter_responses(filter_responses)

	# visual_words.compute_dictionary(num_workers=num_cores)
	
	# dictionary = np.load('dictionary.npy')
	# wordmap = visual_words.get_visual_words(image,dictionary)
	# util.save_wordmap(wordmap, 'test_wordmap.jpg')
	# visual_recog.build_recognition_system(num_workers=num_cores)

	# conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
	# print(conf)
	# print(accuracy)
	import time
	t = time.time()
	vgg16 = torchvision.models.vgg16(pretrained=True)
	vgg16.eval()
	#deep_recog.build_recognition_system(vgg16,num_workers=num_cores)
	print(time.time() - t)
	conf, accuracy = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores)
	print(conf)
	print(accuracy)



