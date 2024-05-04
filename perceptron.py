import numpy as np
import time

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


# Reads in the digit images from a given file such as 'trainingimages', and creates a 3D matrix where each 28x28
# 2D matrix is one of the 28x28 images representing a digit
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
                    if len(current_matrix) == 28:
                        matrix_list.append(current_matrix)
                        current_matrix = []
                    else:
                        raise ValueError(f"Incomplete matrix at lines around {line_number - 28} to {line_number - 1}")
                continue

            if len(line) != 28:
                raise ValueError(
                    f"Line {line_number} does not contain exactly 28 characters (found {len(line)} characters)")
            current_matrix.append([char if char != '' else '' for char in line])

            if len(current_matrix) == 28:
                matrix_list.append(current_matrix)
                current_matrix = []

    if current_matrix:
        raise ValueError(f"Incomplete final matrix starting at line {line_number - len(current_matrix) + 1}")
    return np.array(matrix_list, dtype='<U1')

# Assigns respective labels for each image in a dictionary using one-hot-encoding,
# i.e., label = 9 is represented by the vector [0, 0, 0, 0, 0, 0, 0, 0, 0, 1].
# Each key of the dictionary corresponds to its respective matrix image in the overall 3D array of images.
def assign_labels(file_path):
    labels_dict = {}
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            stripped_line = line.strip()
            if stripped_line.isdigit():
                label = int(stripped_line)
                # Create one-hot encoded vector of size 10
                label_vector = [0] * 10
                label_vector[label] = 1
                labels_dict[index] = label_vector
            else:
                raise ValueError("Non-integer value found in labels file")
    if len(labels_dict) != 5000:
        raise ValueError("The number of labels does not match the expected count of 5000")
    return labels_dict


# Flattens an n*m image into a vector, mapping the features as the vector elements.
# The raw pixels are used directly as the features, resulting in an (n*m)+1 vector.
# Starts at index 1, since index 0 is reserved for the bias.
def map_features(image_matrix):
    flattened_vector = [1]
    for row in image_matrix:
        for char in row:
            if char == ' ':  # Empty character
                flattened_vector.append(-10)
            else:  # Non-empty character
                flattened_vector.append(10)
    return np.array(flattened_vector)


# To initialize weight matrices
def init_weights(dimx, dimy):
    weights = np.random.uniform(0.0000000001, 0.9999999999, size=(dimx, dimy))
    weights[0, 0] = 1
    return weights


def perceptron(features, labels, alpha=1.0, max_iterations=1000):
    num_samples, num_features = features.shape
    theta = np.random.randn(num_features) * 0.01
    iteration = 0
    
    while iteration < max_iterations:
        Delta = np.zeros(num_features)  # initialize delta for epoch
        for i in range(num_samples):
            if labels[i] * np.dot(features[i], theta) <= 0:
                Delta += labels[i] * features[i]  # Update delta if wrong
                print(str(iteration) + " is incorrect")
        if labels[i] * np.dot(features[i], theta) > 0:
            print(str(iteration) + " is correct")
        Delta /= num_samples  # find average update
        theta += alpha * Delta  # Update weights
        iteration += 1
    
    return theta


def calculate_accuracy(features, labels, weights):
    predictions = np.dot(features, weights)
    predicted_labels = np.sign(predictions)
    accuracy = (predicted_labels == labels).mean() * 100
    return accuracy

'''
def calculate_accuracy(features, labels, weights):
    correct_count = 0
    total_count = len(labels)
    prediction = 0

    for i in range(total_count):

        # find prediction, zip just pairs values together
        for feature, weight in zip(features[i], weights):
            prediction += feature * weight
        # make it 1 or -1
        prediction_label = -1 if prediction < 0 else 1
        # compare prediction and label
        if prediction_label == labels[i]:
            correct_count += 1

    accuracy = (correct_count / total_count) * 100
    return accuracy'''




training_matrix = read_data_into_3d("data/digitdata/trainingimages")
training_labels_dict = assign_labels("data/digitdata/traininglabels")

binary_labels = []
for label in training_labels_dict.values():
    # find index of the max value
    index = np.argmax(label)
    
    # Check if index is 0, assign -1, if not then assign 1
    if index == 0:
        binary_label = -1
    else:
        binary_label = 1
    
    binary_labels.append(binary_label)

features = np.array([map_features(image) for image in training_matrix])
labels = np.array(binary_labels)

# Perceptron training
weights = perceptron(features, labels, alpha=0.1, max_iterations=1000)

accuracy = calculate_accuracy(features, labels, weights)
print(f"Accuracy: {accuracy:.2f}%")

np.savetxt("perceptron_weights.txt", weights, fmt='%f')
print("weights saved")
