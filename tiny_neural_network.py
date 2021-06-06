import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x)

training_inputs = np.array([[0, 0, 1],
						    [1, 1, 1],
						    [1, 0, 1],
						    [0, 1, 1],
						    [0, 1, 0]])

training_outputs = np.array([[0, 1, 1, 0, 0]]).T

#same random numbers
np.random.seed(2)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weights')
print(synaptic_weights)

for i in range(1):
	inputs = training_inputs
	outputs = sigmoid(np.dot(inputs, synaptic_weights))
	difference = training_outputs - outputs 
	adjustments = difference * sigmoid_derivative(outputs)
	print(inputs.T)
	print(adjustments)
	synaptic_weights += np.dot(inputs.T, adjustments)
	print(synaptic_weights)

print('Final synaptic weights')
print(synaptic_weights)

print('Write input values')
a = int(input('A = '))
b = int(input('B = '))
c = int(input('C = '))
new_input = np.array([[a, b, c]])
output = sigmoid(np.dot(new_input, synaptic_weights))

print('Result for [' + str(a) + ', ' + str(b) + ', ' + str(c) + ']')
print(output)

input()