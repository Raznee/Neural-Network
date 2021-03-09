import numpy as np
import math
import random

IL = 2
HL = 2
OL = 2

LEARNING_RATE = 0.5

#input_layer = np.random.rand(IL)

input_layer = np.zeros(IL)
hidden_layer = np.zeros(HL)
output_layer = np.zeros(OL)

z_hl = np.zeros(HL)
z_ol = np.zeros(OL)

#target = np.zeros(OL)
cost = np.zeros(OL)

target = np.random.rand(OL)
#costs = np.random.rand(OL)

b_hl = np.random.rand(HL)
b_ol = np.random.rand(OL)
t_b_hl = np.zeros(HL)
t_b_ol = np.zeros(OL)

w_il_hl = np.random.random((HL,IL))
w_hl_ol = np.random.random((OL,HL))
t_w_il_hl = np.zeros((HL, IL))
t_w_hl_ol = np.zeros((OL, HL))

input_layer[0] = 0.05
input_layer[1] = 0.10

w_il_hl[(0,0)] = 0.15
w_il_hl[(0,1)] = 0.20
w_il_hl[(1,0)] = 0.25
w_il_hl[(1,1)] = 0.30

b_hl[0] = 0.35
b_hl[1] = 0.35

w_hl_ol[(0,0)] = 0.40
w_hl_ol[(0,1)] = 0.45
w_hl_ol[(1,0)] = 0.50
w_hl_ol[(1,1)] = 0.55

b_ol[0] = 0.60
b_ol[1] = 0.60

target[0] = 0.01
target[1] = 0.99

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
            z_hl[j] = temp
        hidden_layer[j]= sigmoid(temp)
    return True;
    
def ff_output():
    z = range(OL)
    for k in z:
        temp = b_ol[k]
        y = range(HL)
        for j in y:
            temp = temp + hidden_layer[j]*w_hl_ol[k,j]
            z_ol[k] = temp
        output_layer[k]= sigmoid(temp)
    return True;    

def calculate_cost():
    z = range(OL)
    for k in z:
        cost[k] = pow (output_layer[k] - target[k], 2) / 2
    return True

def feedforward():
    ff_hidden()
    ff_output()
    return True
    
def doh_C_per_doh_w_hl_ol(index1, index2):
    return doh_C_per_doh_a_ol(index1) * doh_a_ol_per_doh_z_ol(index1) * doh_z_ol_per_doh_w_hl_ol(index2)
    
def doh_C_per_doh_b_ol(index1, index2):
    return doh_C_per_doh_a_ol(index1) * doh_a_ol_per_doh_z_ol(index1) * doh_z_ol_per_doh_b_ol(index2)

def doh_C_tot_per_doh_w_il_hl(index1, index2):
    return doh_C_tot_per_doh_a_hl(index1) * doh_a_hl_per_doh_z_hl(index1) * doh_z_hl_per_doh_w_il_hl(index2)

def doh_C_tot_per_doh_b_hl(index1, index2):
    return doh_C_tot_per_doh_a_hl(index1) * doh_a_hl_per_doh_z_hl(index1) * doh_z_hl_per_doh_b_hl(index2)

def doh_z_hl_per_doh_b_hl(index):
    return 1

def doh_C_per_doh_a_ol(index):
    return (output_layer[index]-target[index])
    
def doh_a_ol_per_doh_z_ol(index):
    return der_sigmoid(z_ol[index])
    
def doh_z_ol_per_doh_w_hl_ol(index):
    return hidden_layer[index]
    
def doh_z_ol_per_doh_b_ol(index):
    return 1

def doh_C_per_doh_a_hl(index1, index2):
    return doh_C_per_doh_a_ol(index1) * doh_a_ol_per_doh_z_ol(index1)*doh_z_ol_per_doh_a_hl(index1, index2)

def doh_C_tot_per_doh_a_hl(index):
    temp = 0 
    a = range(OL)
    for b in a:
        temp = temp + doh_C_per_doh_a_hl(b, index)
    return temp

def doh_a_hl_per_doh_z_hl(index):
    return der_sigmoid(z_hl[index])
    
def doh_z_hl_per_doh_w_il_hl(index):
    return input_layer[index]
    
def doh_z_ol_per_doh_a_hl(index1, index2):
    return w_hl_ol[(index1,index2)]

def calculate_w_hl_ol(index1, index2):
    t_w_hl_ol[(index1,index2)] = w_hl_ol[(index1,index2)] - LEARNING_RATE*doh_C_per_doh_w_hl_ol(index1,index2)
    
def calculate_b_ol(index1, index2):
    t_b_ol[index2] = b_ol[index2] - LEARNING_RATE * doh_C_per_doh_b_ol(index1,index2)
    
def calculate_w_il_hl(index1, index2):
    t_w_il_hl[(index1,index2)] = w_il_hl[(index1,index2)] - LEARNING_RATE *doh_C_tot_per_doh_w_il_hl(index1,index2)

def calculate_b_hl(index1, index2):
    t_b_hl[index2] = b_hl[index2] - LEARNING_RATE * doh_C_tot_per_doh_b_hl(index1,index2)

def update_w_hl_ol(index1, index2):
    w_hl_ol[(index1,index2)] = t_w_hl_ol[(index1,index2)]
    
def update_b_ol(index):
    b_ol[index] = t_b_ol[index]
    
def update_w_il_hl(index1, index2):
    w_il_hl[(index1,index2)] = t_w_il_hl[(index1,index2)]
    
def update_b_hl(index):
    b_hl[index] = t_b_hl[index]

def backpropagate_ol():
    z = range(OL)
    for k in z:
        y = range(HL)
        for j in y:
            calculate_w_hl_ol(j, k)
        calculate_b_ol(j,k)
    return True
                
def backpropagate_hl():
    y = range(HL)
    for j in y:
        x = range(IL)
        for i in x:
            calculate_w_il_hl(i, j)
        calculate_b_hl(i,j)
    return True
    
def update():
    z = range(OL)
    for k in z:
        y = range(HL)
        for j in y:
            update_w_hl_ol(j, k)
        update_b_ol(k)
    y = range(HL)
    for j in y:
        x = range(IL)
        for i in x:
            update_w_il_hl(i, j)
        update_b_hl(j)
    return True
        
def backpropagate():
    backpropagate_ol()
    backpropagate_hl()
    return True

"""
print("input layer:")
print(input_layer)
print("hidden layer:")
print(hidden_layer)
print("output layer:")
print(output_layer)
print("target:")
print(target)
print(cost)
print(C0)
"""

print("Weights beetween HL and OL: ")
print(w_hl_ol)
print("Weights beetween IL and HL: ")
print(w_il_hl)
print("Biases of OL: ")
print(b_ol)
print("Biases of HL: ")
print(b_hl)

feedforward()    
calculate_cost()
backpropagate()
update()

print("New weights beetween HL and OL: ")
print(w_hl_ol)
print("New weights beetween IL and HL: ")
print(w_il_hl)
print("New biases of OL: ")
print(b_ol)
print("New biases of HL: ")
print(b_hl)
