import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def relu(x):
    return np.maximum(0,x)

def der_sigmoid(x):
  return sigmoid(x)*(1 - sigmoid(x))

IL =10
HL=5
OL=12

input_layer = np.zeros(IL)
hidden_layer = np.zeros(HL)
output_layer = np.zeros(OL)

actual_outputs = np.zeros(OL
costs = np.zeros(OL

b_hl = np.zeros(HL)
b_ol = np.zeros(OL)

w_il_hl = np.zeros((HL, IL))
w_hl_ol = np.zeros((OL, HL))
