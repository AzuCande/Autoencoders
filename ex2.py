# Import libraries
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import json
from perceptrons.linear_perceptron import LinearPerceptron
from perceptrons.non_linear_perceptron import NonLinearPerceptron

x = []
y = []
z = []


with open('Ejer2/TP2-ej2-conjunto.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    data = []
    expected_output = []

    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[2]))
        z.append(float(row[3]))
        data.append([float(row[0]), float(row[1]), float(row[2])])
        expected_output.append(float(row[3]))

    #pnt promedio
    x1 = sum(x)/len(x)
    y1 = sum(y) / len(y)
    z1 = sum(z) / len(z)

    colors = []
    for i in x:
        colors.append("green")
    colors.append('red')

    x.append(x1)
    y.append(y1)
    z.append(z1)

    # Creating figure
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(x, y, z, color=colors)
    plt.title("simple 3D scatter plot")

    # show plot
    plt.show()

with open('Ejer2/ex2_config.json') as file:
    jsonObject = json.load(file)
    file.close()

perceptron = str(jsonObject["perceptron"])
learning_rate = float(jsonObject["learning_rate"])
generation = int(jsonObject["generation"])

if (perceptron == "linear"):
    perceptron = LinearPerceptron(data, expected_output, learning_rate)
    errors, min_w = perceptron.train(generation)
    print("Errors: " + str(errors))
    print("Min w: " + str(min_w))
else:
    perceptron = NonLinearPerceptron(data, expected_output, learning_rate)
    errors, min_w = perceptron.train(generation)
    print("Errors: " + errors)
    print("Min w: " + min_w)
