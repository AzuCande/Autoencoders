import numpy as np
import matplotlib.pyplot as plt
from layer import Layer
from multi_perceptron import *
from font import *
from utils import *
import time

start_time = time.time()

x1 = np.array(get_input(1))
x1 = [x1[8], x1[9], x1[10], x1[11]]
x1 = np.array(x1)

x = np.array(get_input(1))
x = [x[0], x[1], x[2], x[3], x[4], x[4], x[5],x[6],x[7]]
x = np.array(x)

text_names = get_header(1)

x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)

x = transform(x)
x1 = transform(x1)


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

min_error, errors, epochs, training_accuracies = encoderDecoder.train(x, x, iterations_qty=10000, adaptative_eta=False)
print(min_error)
print(time.time() - start_time)

encoder = MultiPerceptron(encoderDecoder.neuron_layers[0:int(len(layers)/2)], init_layers=False)

decoder = MultiPerceptron(encoderDecoder.neuron_layers[int(len(layers)/2):], init_layers=False)

# aux_x = []
# aux_y = []

# for i in range(len(x)):
#     to_predict = x[i, :]
#     encoded = encoder.predict(to_predict)
#     decoded = decoder.predict(encoded)
#     graph_digits(to_predict, decoded)
#     aux_x.append(encoded[0])
#     aux_y.append(encoded[1])

# a = [aux_x[1], aux_y[1]]
# b = [aux_x[2], aux_y[2]]

# m = (a[1] - b[1]) / (a[0] - b[0])

# # y-y1 = m(x-x1)

# x = []
# if a[0] < b[0]:
#     for i in range(11):
#         x.append(a[0] + i * abs((b[0] - a[0]) / 10))
# else:
#     for i in range(10):
#         x.append(b[0] + i * abs((b[0] - a[0]) / 10))


# y = []

# for i in x:
#     y.append(m * (i - a[0]) + a[1])


# input = []
# for i in range(len(x)):
#     input.append([x[i], y[i]])


# for i in range(len(x)):
#     decoded = decoder.predict(input[i])
#     graph_digits(decoded, decoded)

# plt.plot()


# plt.xlim([-1.1, 1.1])
# plt.ylim([-1.1, 1.1])
# for i, txt in enumerate(text_names):
#     plt.annotate(txt, (aux_x[i], aux_y[i]))
# plt.scatter(aux_x, aux_y)
# plt.show()

aux_1 = []
aux_2 = []

for i in range(len(x)):
    to_predict = x[i, :]
    encoded = encoder.predict(to_predict)
    decoded = decoder.predict(encoded)
    graph_digits(to_predict, decoded)
    aux_1.append(encoded[0])
    aux_2.append(encoded[1])
    
for i in range(len(x1)):
    to_predict = x1[i, :]
    encoded = encoder.predict(to_predict)
    decoded = decoder.predict(encoded)
    graph_digits(to_predict, decoded)


plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
# for i, txt in enumerate(text_names):
    # plt.annotate(txt, (aux_1[i], aux_2[i]))
plt.scatter(aux_1, aux_2)
plt.show()