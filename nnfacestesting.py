import nnfaces as nn
import numpy as np


# Initialization
training_matrix = nn.read_data_into_3d("data/facedata/facedatatest")
training_labels = nn.assign_labels("data/facedata/facedatatestlabels")

in_vector = nn.map_features(training_matrix[0])
hidden_layer = np.empty(51)  # Arbitrary number of neurons in the hidden layer = 50 + 1 (bias)
output_nodes_count = 2  # 2 face classifications

weights1 = np.load('nnfacesweights1.npy')
weights2 = np.load('nnfacesweights2.npy')

correct_guesses = 0
total_guesses = 0
i = 0

for image in training_matrix:
    a1 = nn.map_features(image)
    a2, a3 = nn.forward_propagation(a1, weights1, weights2)

    if a3.argmax() == training_labels[i].argmax():
        correct_guesses += 1
    total_guesses += 1
    i += 1

print(f'{round(correct_guesses/total_guesses, 4) * 100}%')
print(correct_guesses)
print(total_guesses)
