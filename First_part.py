from numpy import exp, array, random, dot
import math

file = open('FirstData.txt', 'r')

# создание массивов с данными
str_data = [line.strip() for line in file]
x_data = str_data[0].split(' ')
y_data = str_data[1].split(' ')

for i in range(len(x_data)):
    x_data[i] = float(x_data[i])
    y_data[i] = float(y_data[i])

# Установка количества слоёв и изначальные параметры весов
sizes = [5, 3, 2, 1]
#biases = [random.randn(y, 1) for y in sizes[1:]]
#weights = [random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
neurons = [[0, 0, 0], [0, 0], [0]]
a = 0.001

def generate_edges():
    random.seed()

    edges = [
        [0, 3, random.uniform(-0.5, 0.5)],
        [0, 4, random.uniform(-0.5, 0.5)],
        [1, 3, random.uniform(-0.5, 0.5)],
        [1, 4, random.uniform(-0.5, 0.5)],
        [2, 3, random.uniform(-0.5, 0.5)],
        [2, 4, random.uniform(-0.5, 0.5)],
        [3, 5, random.uniform(-0.5, 0.5)],
        [4, 5, random.uniform(-0.5, 0.5)],
        [-1, 3, random.uniform(-0.5, 0.5)],  # нейроны смещения
        [-1, 4, random.uniform(-0.5, 0.5)],  # нейроны смещения
        [-1, 5, random.uniform(-0.5, 0.5)],  # нейроны смещения
    ]

    return edges

# Подсчет входных данных
def input(x):
    x1 = -3.439281 * x - 0.322819
    x2 = 3.441836 * x + 0.811632
    x3 = 3.067 * (x**2) + 1.012 * x + 1.19
#    x4, x5 = 0, 0
#    if x > 0.2:
#        x4 = 1
#    if x < -0.53:
#        x5 = 1
    input = array([x1, x2, x3])
    return input

# Логистическая функция активации
def sigmoid(x):
    return 1 / (1 + exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def new_neurons(x, edges):
    x1 = -3.439281 * x - 0.322819
    x2 = 3.441836 * x + 0.811632
    x3 = 3.067 * (x**2) + 1.012 * x + 1.19
    neurons = [x1, x2, x3, 0, 0, 0]

    for i in range(3, 6):
        s = 0
        for edge in edges:
            if edge[1] == i:
                # источник * стоимость ребра
                if edge[0] == -1:
                    s += edge[2]
                else:
                    s += neurons[edge[0]] * edge[2]
        if i == 5: neurons[i] = s
        else: neurons[i] = sigmoid(s)

    return neurons


def calculate_errors(error, edges):
    errors = [0, 0, 0, 0, 0, error]

    for i in [4,3]:
        for edge in edges:
            if edge[0] == i:
                # адресат * стоимость ребра
                errors[i] += errors[edge[1]] * edge[2]

    return errors


def correct_edges(edges, errors, neuron):
    for edge in edges:
        if edge[0] == -1:
            edge[2] += errors[edge[1]] * sigmoid_derivative(neuron[edge[1]]) * a
        else:
            edge[2] += errors[edge[1]] * sigmoid_derivative(neuron[edge[1]]) * neuron[edge[0]] * a

    return edges


def train(edges, x_data, y_data):
    x_arr = x_data
    y_arr = y_data
    for (i, x) in enumerate(x_arr):
        neurons = new_neurons(x, edges)
        error = y_arr[i] - neurons[5]
        errors = calculate_errors(error, edges)
        edges = correct_edges(edges, errors, neurons)

    return edges


if __name__ == "__main__":
    for i in range(1, 11):
        edges = generate_edges()
        random.shuffle(x_data)
        random.shuffle(y_data)
        vals = train(edges, x_data, y_data)
    for j in range(-30, 30):
        res = new_neurons(j, vals)
        print(j, res[5])


#Подсчет синопса
#def forward(data, i, j):
#    sum = 0
#    for x in data:
#        k = 0
#        sum += x * weights[i][j][k]
#        k += 1
#    sum += biases[i][j]
#    return sum


#def train_network(x_data, y_data):
#    i = 0
#    for data in x_data:
#        net_input = input(data)

        #first_layer
#        synapse_1 = forward(net_input,0,0) #Вычислила функцию S
#        neurons[0][0] = sigmoid(synapse_1) #Вычислила саму функцию F
#        synapse_2 = forward(net_input,0,1)
#        neurons[0][1] = sigmoid(synapse_2)
#        synapse_3 = forward(net_input,0,2)
#        neurons[0][2] = sigmoid(synapse_3)
#        first_layer = array([neurons[0][0], neurons[0][1], neurons[0][2]])

        #second_layer
#        synapse_4 = forward(first_layer,1,0)
#        neurons[1][0] = sigmoid(synapse_4)
#        synapse_5 = forward(first_layer,1,1)
#        neurons[1][1] = sigmoid(synapse_5)
#        second_layer = array([neurons[1][0], neurons[1][1]])

        #output
#        synapse_6 = forward(second_layer,2,0)
#        neurons[2] = sigmoid(synapse_6)

#        error = y_data[i] - neurons[2]
#        calculate_errors(error, weights)


 #       print('Получилось =', neurons[2])
 #       print('Ошибка =', error)

        #Далее надо идти обратно и изменять веса
        #i += 1
        #adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
        #self.synaptic_weights += adjustment


#train_network(x_data, y_data)
