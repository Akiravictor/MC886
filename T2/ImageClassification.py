# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 21:46:05 2017

@author: Akira
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import pylab

image = Image.open("train/00000.png")

imageGray = image.convert('L')

imageArr = np.array(image)
imageGrayArr = np.array(imageGray)

pylab.figure()
pylab.gray()

pylab.contour(imageGray, origin='image')
pylab.axis('equal')
pylab.axis('off')

pylab.figure()

pylab.hist(imageGrayArr.flatten(),128)

activation = "tanh"

if activation == "tanh":
    print("tf.nn.tanh")
if activation == "relu":
    print("relu")
    
    
import tensorflow as tf

tf.nn.relu()

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../")


tf.contrib.learn.DNNClassifier()