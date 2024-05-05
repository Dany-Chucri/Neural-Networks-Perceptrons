import numpy as np
import time

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


# Reads in the digit images from a given file such as 'trainingimages', and creates a 3D matrix where each 28x28
# 2D matrix is one of the 28x28 images representing a digit
def read_data_into_3d(file_path, input):

    if(input=="1"):
        height = 70
        weight = 60
    else:
        height = 28
        weight = 28


    matrix_list = []
    current_matrix = []
    line_number = 0

    with open(file_path, 'r') as file:
        for line in file:
            line_number += 1
            line = line.rstrip('\n')

            if line == "":
                if current_matrix:
                    if len(current_matrix) == height:
                        matrix_list.append(current_matrix)
                        current_matrix = []
                    else:
                        raise ValueError(f"Incomplete matrix at lines around {line_number - 28} to {line_number - 1}")
                continue

            if len(line) != weight:
                print(weight)
                print(len(line))
                raise ValueError(
                    f"Line {line_number} does not contain exactly 28 characters (found {len(line)} characters)")
            current_matrix.append([char if char != '' else '' for char in line])

            if len(current_matrix) == height:
                matrix_list.append(current_matrix)
                current_matrix = []

    if current_matrix:
        raise ValueError(f"Incomplete final matrix starting at line {line_number - len(current_matrix) + 1}")
    return np.array(matrix_list, dtype='<U1')

# Assigns respective labels for each image in a dictionary using one-hot-encoding,
# i.e., label = 9 is represented by the vector [0, 0, 0, 0, 0, 0, 0, 0, 0, 1].
# Each key of the dictionary corresponds to its respective matrix image in the overall 3D array of images.
def assign_labels(file_path, type):

    n = 5000
    if(type == 1):
        n=451
    elif(type == 2):
        n=301
    elif(type == 3):
        n=1000
    elif(type==4):
        n=150

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
    if len(labels_dict) != n:
        print(len(labels_dict))
        print(n)
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
        Delta /= num_samples  # find average update
        theta += alpha * Delta  # Update weights
        iteration += 1
    
    return theta

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
    return accuracy
'''

def calculate_accuracy(features, labels, weights):
    predictions = np.dot(features, weights)
    predicted_labels = np.sign(predictions)
    accuracy = (predicted_labels == labels).mean() * 100
    return accuracy

def convert_to_binary_labels(labels_dict):
    binary_labels = []
    for label in labels_dict.values():
        # find index of the max value
        index = np.argmax(label)
        # Check if index is 0, assign -1, if not then assign 1
        binary_label = -1 if index == 0 else 1
        binary_labels.append(binary_label)
    return binary_labels

user_input = input("Enter 0 for digit classification and 1 for facial recognition\n")

if(user_input == "0"):

    training_matrix = read_data_into_3d("data/digitdata/trainingimages", user_input)
    training_labels_dict = assign_labels("data/digitdata/traininglabels", 0)

    binary_labels = convert_to_binary_labels(training_labels_dict)

    features = np.array([map_features(image) for image in training_matrix])
    labels = np.array(binary_labels)

    # Perceptron training
    print("training...\n")
    weights = perceptron(features, labels, alpha=0.1, max_iterations=1000)

    testingInput = input("\nInput 0 for training set accuracy, 1 for validation set accuracy, or 2 for testing set accuracy\n")

    if(testingInput == "0"):
        accuracy = calculate_accuracy(features, labels, weights)
        print(f"\nPerceptron Digit Characterization Accuracy (training): {accuracy:.2f}%\n")
        np.savetxt("perceptron_digits_weights.txt", weights, fmt='%f')
        print("Weights Succesfully Saved")

    elif(testingInput == "1"):
        validation_matrix = read_data_into_3d("data/digitdata/validationimages", user_input)
        validation_labels_dict = assign_labels("data/digitdata/validationlabels", 3)

        val_binary_labels = convert_to_binary_labels(validation_labels_dict)

        val_features = np.array([map_features(image) for image in validation_matrix])
        val_labels = np.array(val_binary_labels)

        accuracy = calculate_accuracy(val_features, val_labels, weights)
        print(f"\nPerceptron Digit Characterization Accuracy (training): {accuracy:.2f}%\n")

        np.savetxt("perceptron_digit_weights.txt", weights, fmt='%f')
        print("Weights Succesfully Saved")

    elif(testingInput == "2"):
        test_matrix = read_data_into_3d("data/digitdata/testimages", user_input)
        test_labels_dict = assign_labels("data/digitdata/testlabels", 3)

        test_binary_labels = convert_to_binary_labels(test_labels_dict)

        test_features = np.array([map_features(image) for image in test_matrix])
        test_labels = np.array(test_binary_labels)

        accuracy = calculate_accuracy(test_features, test_labels, weights)
        print(f"\nPerceptron Digit Characterization Accuracy (training): {accuracy:.2f}%\n")

        np.savetxt("perceptron_digit_weights.txt", weights, fmt='%f')
        print("Weights Succesfully Saved")

    else: 
        print("Invalid Input")


    validation_matrix = read_data_into_3d("data/digitdata/validationimages", user_input)
    validation_labels_dict = assign_labels("data/digitdata/validationlabels", 3)

    val_binary_labels = convert_to_binary_labels(validation_labels_dict)

    val_features = np.array([map_features(image) for image in validation_matrix])
    val_labels = np.array(val_binary_labels)


    accuracy = calculate_accuracy(val_features, val_labels, weights)
    print(f"\nPerceptron Digit Classification Accuracy: {accuracy:.2f}%\n")

    np.savetxt("perceptron_digits_weights.txt", weights, fmt='%f')
    print("Weights Succesfully Saved")

elif(user_input == "1"):

    training_matrix = read_data_into_3d("data/facedata/facedatatrain", user_input)
    training_labels_dict = assign_labels("data/facedata/facedatatrainlabels", 1)

    binary_labels = convert_to_binary_labels(training_labels_dict)

    features = np.array([map_features(image) for image in training_matrix])
    labels = np.array(binary_labels)

    # Perceptron training
    weights = perceptron(features, labels, alpha=0.1, max_iterations=1000)

    testingInput = input("\nInput 0 for training set accuracy, 1 for validation set accuracy, or 2 for testing set\n")

    if(testingInput == "0"):
        accuracy = calculate_accuracy(features, labels, weights)
        print(f"\nPerceptron Facial Recognition Accuracy (training): {accuracy:.2f}%\n")
        np.savetxt("perceptron_faces_weights.txt", weights, fmt='%f')
        print("Weights Succesfully Saved")

    elif(testingInput == "1"):
        validation_matrix = read_data_into_3d("data/facedata/facedatavalidation", user_input)
        validation_labels_dict = assign_labels("data/facedata/facedatavalidationlabels", 2)

        val_binary_labels = convert_to_binary_labels(validation_labels_dict)

        val_features = np.array([map_features(image) for image in validation_matrix])
        val_labels = np.array(val_binary_labels)

        accuracy = calculate_accuracy(val_features, val_labels, weights)
        print(f"\nPerceptron Facial Recognition Accuracy (validation): {accuracy:.2f}%\n")

        np.savetxt("perceptron_faces_weights.txt", weights, fmt='%f')
        print("Weights Succesfully Saved")

    elif(testingInput == "2"):
        test_matrix = read_data_into_3d("data/facedata/facedatatest", user_input)
        test_labels_dict = assign_labels("data/facedata/facedatatestlabels", 4)

        test_binary_labels = convert_to_binary_labels(test_labels_dict)

        test_features = np.array([map_features(image) for image in test_matrix])
        test_labels = np.array(test_binary_labels)

        accuracy = calculate_accuracy(test_features, test_labels, weights)
        print(f"\nPerceptron Facial Recognition Accuracy (test): {accuracy:.2f}%\n")

        np.savetxt("perceptron_faces_weights.txt", weights, fmt='%f')
        print("Weights Succesfully Saved")

    else: 
        print("Invalid Input")
        
else:
    print("Invalid Input")




