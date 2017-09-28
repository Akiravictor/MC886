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
