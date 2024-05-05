import numpy as np
import time

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


# Reads in the faces images from a given file such as 'trainingimages', and creates a 3D matrix where each 70x60
# 2D matrix is one of the 70x60 images representing a face (or not)
def read_data_into_3d(file_path):
    matrix_list = []
    current_matrix = []
    line_number = 0

    with open(file_path, 'r') as file:
        for line in file:
            line_number += 1
            line = line.rstrip('\n')

            if line == "":
                if current_matrix:
                    if len(current_matrix) == 70:
                        matrix_list.append(current_matrix)
                        current_matrix = []
                    else:
                        raise ValueError(f"Incomplete matrix at lines around {line_number - 70} to {line_number - 1}")
                continue

            if len(line) != 60:
                raise ValueError(
                    f"Line {line_number} does not contain exactly 60 characters (found {len(line)} characters)")
            current_matrix.append([char if char != '' else '' for char in line])

            if len(current_matrix) == 70:
                matrix_list.append(current_matrix)
                current_matrix = []

    if current_matrix:
        raise ValueError(f"Incomplete final matrix starting at line {line_number - len(current_matrix) + 1}")
    return np.array(matrix_list)


# Assigns respective labels for each image in a dictionary using one-hot-encoding,
# i.e., label = 1 is represented by the vector [0, 1].
# Each key of the dictionary corresponds to its respective matrix image in the overall 3D array of images.
def assign_labels(file_path):
    labels_dict = {}
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            stripped_line = line.strip()
            if stripped_line.isdigit():
                label = int(stripped_line)
                # Create one-hot encoded vector of size 2
                label_vector = [0] * 2
                label_vector[label] = 1
                labels_dict[index] = np.array(label_vector)
            else:
                raise ValueError("Non-integer value found in labels file")
    return labels_dict


# Flattens an n*m image into a vector, mapping the features as the vector elements.
# The raw pixels are used directly as the features, resulting in an (n*m)+1 vector.
# Starts at index 1, since index 0 is reserved for the bias.
def map_features(image_matrix):
    flattened_vector = [1]
    for row in image_matrix:
        for char in row:
            if char == ' ':  # Empty character
                flattened_vector.append(0.1)
            else:  # Non-empty character
                flattened_vector.append(10)
    return np.array(flattened_vector)


# To initialize weight matrices
def init_weights(dimx, dimy):
    weights = np.empty((dimx, dimy))
    for iy, ix in np.ndindex(weights.shape):
        weight = 0
        while weight == 0:
            weight = np.random.uniform(-1, 1)
        weights[iy, ix] = weight
    return weights


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Softmax activation function
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


# ReLU activation function
def ReLU(x):
    return np.maximum(x, 0)


def forward_propagation(layer1, weights_1_2, weights_2_out):
    z2 = np.dot(weights_1_2, layer1)
    a2 = sigmoid(z2)
    z3 = np.dot(weights_2_out, a2)
    a3 = softmax(z3)
    # testing prints
    i = 0
    if i == 1:
        print(f'layer1 is {layer1}')
        print(f'layer1 shape is {layer1.shape}')
        print(f'weights_1_2  is {weights_1_2}')
        print(f'weights_1_2 shape is {weights_1_2.shape}')
        print(f'z2 is {z2}')
        print(f'z2 shape is {z2.shape}')
        print(f'a2 is {a2}')
        print(f'a2 shape is {a2.shape}')
        print(f'weights_2_out shape is {weights_2_out.shape}')
        print(f'z3 is {z3}')
        print(f'z3 shape is {z3.shape}')
        print(f'a3 is {a2}')
        print(f'a3 shape is {a3.shape}')
        print(a3.sum())
    return a2, a3


def validate(training_matrix, training_labels, weights_1, weights_2):
    correct_guesses = 0
    total_guesses = 0
    i = 0

    for image in training_matrix:
        a1 = map_features(image)
        a2, a3 = forward_propagation(a1, weights_1, weights_2)

        if a3.argmax() == training_labels[i].argmax():
            correct_guesses += 1
        total_guesses += 1
        i += 1

    print(f'{round(correct_guesses / total_guesses, 4) * 100}%')
    print(correct_guesses)
    print(total_guesses)


def train():
    # Initialization
    training_matrix = read_data_into_3d("data/facedata/facedatatrain")
    training_labels = assign_labels("data/facedata/facedatatrainlabels")

    in_vector = map_features(training_matrix[0])
    hidden_layer = np.empty(51)  # Arbitrary number of neurons in the hidden layer = 50 + 1 (bias)
    output_nodes_count = 2  # 2 face classifications (face or no face)

    weights1 = init_weights(len(hidden_layer), len(in_vector))  # For weights mapping from input layer to hidden layer
    weights2 = init_weights(output_nodes_count,
                            len(hidden_layer))  # For weights mapping from hidden layer to output layer

    # Epoch looping
    start_time = time.time()
    iteration = 0

    try:
        while True:
            gradient1 = np.zeros(len(in_vector))
            gradient2 = np.zeros(len(hidden_layer))

            # print('gradient1 is', gradient1.shape)
            # print('gradient2 is', gradient2.shape)

            i = 0  # For quickly referencing between each image and its label
            for image in training_matrix:
                # Forward Propagation
                a1 = map_features(image)
                a2, a3 = forward_propagation(a1, weights1, weights2)
                # print(f'training_labels[i] is {training_labels[i]}')
                # print('a1 is', a1.shape)
                # print('a2 is', a2.shape)
                # print('a3 is', a3.shape)

                # Backward Propagation, error computing
                error3 = a3 - training_labels[i]
                error2 = np.dot(weights2.T, error3) * (a2 * (1 - a2))
                # print('error3 is', error3.shape)
                # print('error2 is', error2.shape)

                # Gradient computation
                gradient2 = gradient2 + np.dot(error3.reshape(2, 1), a2.reshape(1, 51))
                gradient1 = gradient1 + np.dot(error2.reshape(51, 1), a1.reshape(1, 4201))
                # print('gradient2 is', gradient2.shape)
                # print('gradient1 is', gradient1.shape)

                # print(a3)
                # print(training_labels[i])
                # exit(0)
                i += 1

            # Compute average regularized gradient
            # Using lambda = 0 for now (lambda is usually meant to prevent over-fitting)
            lmbd = 0.01
            D_1 = (gradient1 / (len(in_vector) - 1)) + (lmbd * weights1)
            D_2 = (gradient2 / (len(in_vector) - 1)) + (lmbd * weights2)

            # print('D_1 is', D_1.shape)
            # print('D_2 is', D_2.shape)
            # print('weights1 is', weights1.shape)
            # print('weights2 is', weights2.shape)

            # Update weights via gradient step
            learning_rate = 2  # Arbitrary, lower = less over-fitting but slower
            weights1 = weights1 - (learning_rate * D_1)
            weights2 = weights2 - (learning_rate * D_2)

            if iteration % 10 == 0:
                validate(training_matrix, training_labels, weights1, weights2)

            print(f'Finished epoch {iteration} at time {round(time.time() - start_time, 3)} seconds')
            iteration += 1

            elapsed_time = time.time() - start_time
            if elapsed_time > 600:
                print(f"Loop terminated: Exceeded {round(elapsed_time/60, 3)} minute(s).")
                np.save('nnfacesweights1', weights1)
                np.save('nnfacesweights2', weights2)
                print('Weights have been saved.')
                break

    except KeyboardInterrupt:
        print('Time spent:', round(time.time() - start_time, 3), 'seconds')
        np.save('nnfacesweights1', weights1)
        np.save('nnfacesweights2', weights2)
        print('Weights have been saved.')


def main():
    train()
