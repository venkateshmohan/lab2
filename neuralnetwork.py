# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:11:28 2018

@author: Venkatesh T Mohan
"""

from keras.layers import Activation,Dense
from numpy import array
from numpy import argmax
import pandas as pd
from keras.utils import to_categorical
# define example
with open("optdigits.tra", "r") as f:
     for row in f:
        tmp=row.split()
        str1=''.join(tmp)
        str2=str1.split(',')
        encoded = to_categorical(str2)
        print(encoded)
# invert encoding
#inverted = argmax(encoded[0])
#print(inverted)
