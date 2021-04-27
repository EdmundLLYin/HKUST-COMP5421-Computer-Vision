from os import walk
import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage.io
import glob
import os


if __name__ == '__main__':

    PATH = '../data'
    dictionary = np.load('dictionary.npy')

    for (dirpath, dirnames, filenames) in walk(PATH):
        for folder in dirnames:
            for file in glob.glob(PATH + '/' + folder + '/' +  '*.jpg'):
                image = skimage.io.imread(file)
                image = image.astype('float')/255
                
                wordmap = visual_words.get_visual_words(image,dictionary)
                util.save_wordmap(wordmap, '../wordmap/' + folder + '/' + os.path.basename(file))






