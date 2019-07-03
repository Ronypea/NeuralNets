import random
import numpy as np
import math

a = 0.01

# задаём веса рёбрам
def generate_edges():
    random.seed()

    edges = [
        [0, 3, random.choice([-1, 1])],
        [0, 4, random.choice([-1, 1])],
        [1, 3, random.choice([-1, 1])],
        [1, 4, random.choice([-1, 1])],
        [2, 3, random.choice([-1, 1])],
        [2, 4, random.choice([-1, 1])],
        [3, 5, random.choice([-1, 1])],
        [4, 5, random.choice([-1, 1])],
    ]

    return edges


# функция активации
def activation_func(x):
    if x > 0:
        return x
    else:
        return 0


# функция активации
def activation_func_d(x):
    if x > 0:
        return 1
    else:
        return 0


def new_neurons(x, y, z, edges):
    vals = [x, y, z, 0, 0, 0]

    for i in range(3, 6):
        s = 0
        for edge in edges:
            if edge[1] == i:
                # источник * стоимость ребра
                if edge[0] == -1:
                    s += edge[2]
                else:
                    s += vals[edge[0]] * edge[2]
        vals[i] = activation_func(s)

    return vals


def calculate_errors(error, edges):
    errors = [0, 0, 0, 0, 0, error]

    for i in [4,3]:
        for edge in edges:
            if edge[0] == i:
                # адресат * стоимость ребра
                errors[i] += errors[edge[1]] * edge[2]

    return errors


def correct_edges(edges, errors, vals):
    for edge in edges:
        if edge[0] == -1:
            edge[2] += errors[edge[1]] * activation_func_d(vals[edge[1]]) * a
        else:
            edge[2] += errors[edge[1]] * activation_func_d(vals[edge[1]]) * vals[edge[0]] * a

    return edges

def calculate_res(x,y,z):
    return z or (y and not x)

def train(edges):
    for x in [0,1]:
        for y in [0,1]:
            for z in [0,1]:
                vals = new_neurons(x, y, z, edges)
                error = calculate_res(x, y, z) - vals[5]
                errors = calculate_errors(error, edges)
                edges = correct_edges(edges, errors, vals)

    return edges


if __name__ == "__main__":
    for i in range(1, 11):
        edges = generate_edges()
        vals = train(edges)
    sum1, sum2 = 0, 0
    for x in [0, 1]:
        for y in [0, 1]:
            for z in [0, 1]:
                res = new_neurons(x, y, z, vals)
                if res[5] >= 0.5: res[5] = 1
                else: res[5] = 0
                sum1 += 1
                if (calculate_res(x, y, z) != res[5]): sum2 +=1
                print(x, y, z, res[5])
                print ('Правильный ответ = ', calculate_res(x, y, z),
                        'Данный ответ =', res[5])
    print ('Ошибки/Все данные =', sum2 / sum1)


