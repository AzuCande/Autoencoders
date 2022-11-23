import numpy as np
import matplotlib.pyplot as plt
from font import *
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def print_letter(letter):
    arr = np.array(letter)
    test = np.array_split(letter, 5)
    aux = len(test)
    for line in range(0, aux):
        str = ''
        for i in range(0, len(test[0])):
            if test[line][i] > 0.5:
                str = str + '*'
            else:
                str = str + ' '
        print(str)

def graph_digits(original, output):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('AE result')
    ax1.imshow(np.array(original).reshape((7, 5)), cmap='gray')
    ax2.imshow(np.array(output).reshape((7, 5)), cmap='gray')
    fig.show()

def transform(t):
    to_ret = []
    for i in t:
        aux = []
        for num in i:
            a = format(num, "b").zfill(5)
            for j in a:
                if j == "0":
                    aux.append(-1)
                elif j == "1":
                    aux.append(1)
        to_ret.append(aux)
    return np.array(to_ret)

def get_input(font):
    return fonts

def get_header(font):
    return fonts_header

def get_output(font):
    return fonts_output

def noise(t):
    RAND = 1 / 35
    to_ret = []
    for i in t:
        aux = []
        for num in i:
            a = format(num, "b").zfill(5)
            for j in a:
                rand = random.uniform(0, 1)
                if j == "0":
                    if rand < RAND:
                        aux.append(1)
                    else:
                        aux.append(-1)
                elif j == "1":
                    if rand < RAND:
                        aux.append(-1)
                    else:
                        aux.append(1)
        to_ret.append(aux)
    return np.array(to_ret)

def makeLittleNoise(num, t):
    to_ret = []
    for i in t:
        bits= []
        for n in range(num): 
            x = round(random.uniform(0, 6))
            y = round(random.uniform(0, 4))
            bits.append([x,y])
        aux = []
        for num in range(len(i)):
            a = format(i[num], "b").zfill(5)
            for j in range(len(a)):
                if a[j] == "0":
                    if [num, j] in bits:
                        aux.append(1)
                    else:
                        aux.append(-1)
                elif a[j] == "1":
                    if [num, j] in bits:
                        aux.append(-1)
                    else:
                        aux.append(1)
        to_ret.append(aux)
    return np.array(to_ret)
# [a, b, c]
# [0x04, 0x02, 0x04...] -> len=7 0x04 -> 00011

def getSample(z_mean, z_log_var):
    z_mean, z_log_var = z_mean, z_log_var
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

