import nndigits as nn
import numpy as np

# Initialization
training_matrix = nn.read_data_into_3d("data/digitdata/testimages")
training_labels = nn.assign_labels("data/digitdata/testlabels")

in_vector = nn.map_features(training_matrix[0])
hidden_layer = np.empty(51)  # Arbitrary number of neurons in the hidden layer = 50 + 1 (bias)
output_nodes_count = 10  # 10 digit classifications

weights1 = np.load('nndigitsweights1.npy')
weights2 = np.load('nndigitsweights2.npy')

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

print(f'{round(correct_guesses/total_guesses, 2) * 100}%')
print(correct_guesses)
print(total_guesses)
