import numpy as np
import math
import random

IL = 3
HL = 2
OL = 3

input_layer = np.random.rand(IL)

#input_layer = np.zeros(IL)
hidden_layer = np.zeros(HL)
output_layer = np.zeros(OL)

#actual_outputs = np.zeros(OL)
costs = np.zeros(OL)

actual_outputs = np.random.rand(OL)
#costs = np.random.rand(OL)

b_hl = np.random.rand(HL)
b_ol = np.random.rand(OL)
#b_hl = np.zeros(HL)
#b_ol = np.zeros(OL)

w_il_hl = np.random.random((HL,IL))
w_hl_ol = np.random.random((OL, HL))
#w_il_hl = np.zeros((HL, IL))
#w_hl_ol = np.zeros((OL, HL))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return np.maximum(0,x)

def der_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))

def ff_hidden():
    y = range(HL)
    for j in y:
        temp = b_hl[j]
        x = range(IL)
        for i in x:
            temp = temp + input_layer[i]*w_il_hl[j,i]
        hidden_layer[j]= sigmoid(temp)
    return True;
    
def ff_output():
    z = range(OL)
    for k in z:
        temp = b_ol[k]
        y = range(HL)
        for j in y:
            temp = temp + hidden_layer[j]*w_hl_ol[k,j]
        output_layer[k]= sigmoid(temp)
    return True;    

def calculate_cost():
    z = range(OL)
    for k in z:
        costs[k] = pow (output_layer[k] - actual_outputs[k], 2)
    return True;
    
ff_output()
calculate_cost()

print(output_layer)
print(actual_outputs)
print(costs)
