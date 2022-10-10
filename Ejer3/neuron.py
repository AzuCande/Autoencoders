import numpy as np
from Ejer3.activation import *


class Neuron:
    def __init__(self, weight_num, activation: Activation, learn_rate):
        self.weights = np.random.uniform(-1, 1, weight_num + 1)
        self.output = 0
        self.delta = 0
        # self.output_dx = 0
        self.activation = activation
        self.learn_rate = learn_rate

    def calculate_output(self, inputs: np.ndarray):
        inputs = np.append(inputs, 1)
        e = np.inner(inputs, self.weights)
        self.output = self.activation.apply(e)
        aux = 0
        # self.output_dx = self.activation.apply_dx(e)

    def update_w(self, inputs: np.ndarray, expected: np.ndarray):
        inputs = np.append(inputs, 1)
        self.delta = self.output * (1 - self.output) * expected
        for i in range(0, len(self.weights)):
            self.weights[i] += self.learn_rate * self.delta * inputs[i]

    def __str__(self):
        print("{ Pesos: " + str(self.weights) + " Delta: " + str(self.delta) + " Output: " + str(self.output) + " }\n")
