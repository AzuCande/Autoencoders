import numpy as np
import matplotlib.pyplot as plt
from layer import Layer
from multi_perceptron import *
from font import *
from utils import *

x = np.array(get_input(1))
x = [x[0], x[25], x[10], x[4], x[12]]
x = np.array(x)

x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)

x_noise = noise(x)
x_noise2 = noise(x)
x = transform(x)

# x_noise = makeLittleNoise(1, x)
# x_noise2 = makeLittleNoise(1, x)
# x = transform(x)
# for i in range(len(x)):
#     to_predict = x_noise2[i, :]
#     pepe = x_noise[i,:]
#     graph_digits(to_predict, pepe)

layer1 = Layer(20, 35, activation="tanh")
layer2 = Layer(10, activation="tanh")
layer2_bis = Layer(5, activation="tanh")
layer3 = Layer(2, activation="tanh")
layer4 = Layer(5, activation="tanh")
layer4_bis = Layer(10, activation="tanh")
layer5 = Layer(20, activation="tanh")
layer6 = Layer(35, activation="tanh")

layers = [layer1, layer2, layer2_bis, layer3, layer4, layer4_bis, layer5, layer6]

encoderDecoder = MultiPerceptron(layers, init_layers=True, momentum=True, eta=0.0005)

min_error, errors, epochs, training_accuracies = encoderDecoder.train(x_noise, x, iterations_qty=10000, adaptative_eta=False)
print(min_error)

encoder = MultiPerceptron(encoderDecoder.neuron_layers[0:int(len(layers)/2)], init_layers=False)

decoder = MultiPerceptron(encoderDecoder.neuron_layers[int(len(layers)/2):], init_layers=False)

print("Longitud de x: " + str(len(x)))

for i in range(len(x)):
    to_predict = x_noise2[i, :]
    encoded = encoder.predict(to_predict)
    decoded = decoder.predict(encoded)
    graph_digits(to_predict, decoded)

plt.show()