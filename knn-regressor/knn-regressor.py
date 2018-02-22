import numpy as np
import csv
import random
import operator


def loadDataset(filename, split_ratio, training=[], test=[]):
    """
    :param filename: path to .csv dataset file
    :param split_ratio: ratio to split the dataset in training and test instances
    :param training: training output array
    :param test: test output array
    :return: None
    """
    with open(filename, 'r') as csvdata:
        data = csv.reader(csvdata)
        dataset = list(data)
        dataset.pop(0)
        for instance in range(len(dataset) - 1):
            for features in range(len(dataset[instance])):
                dataset[instance][features] = float(dataset[instance][features])
            if random.random() < split_ratio: #Split the dataset into training and test
                training.append(dataset[instance])
            else:
                test.append(dataset[instance])


def getDistance(instance1, instance2):
    """
    :param instance1: Float array
    :param instance2:  Floar array
    :return: Euclidean distance between instance1 and instance2
    """
    return np.linalg.norm(np.asarray(instance1) - np.asarray(instance2))


def getNeighbours(training, testInstance, k):
    """
    :param training: training input array. Array of array of floats
    :param testInstance: test input array. Array of floats
    :param k: k hyperparameter
    :return: k nearest neighbours to testInstance. Array of array of floats
    """
    distances = []
    for instance in training:
        dist = getDistance(testInstance, instance)
        distances.append((instance, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for instance in range(k):
        neighbours.append(distances[instance][0])
    return neighbours


def calculateValue(neighbours, k):
    """
    :param neighbours: k nearest neighbours to test instance
    :param k: k hyperparameter
    :return: calculated regeression value
    """
    sum = 0
    for neighbour in neighbours:
        sum += neighbour[-1]
    return sum/k


def getAccuracy(X, test, k):
    """
    :param X: Input training data. Array of array of floats
    :param test: Input test data. Array of array of floats
    :param k: k hyperparameter
    :return: None
    """
    predictions = []
    print('Train set: ' + repr(len(X)))
    print('Test set: ' + repr(len(test)))
    print('k: ' + repr(k))
    for testInstance in test:
        neighbours = getNeighbours(X, testInstance, k)
        prediction = calculateValue(neighbours, k)
        predictions.append(prediction)
    correct = 0
    for i in range(len(test)):
        # print('predicted: ' + repr(predictions[i]) + ' | actual: ' + repr(test[i][-1]))
        if (predictions[i] - 0.15) <= test[i][-1] <= (predictions[i] + 0.15):
            correct += 1
    print('Accuracy: ' + repr((correct / float(len(test))) * 100.0))
    return (correct / float(len(test))) * 100.0


def myknnregress(X, test, k):
    """
    :param X: Training dataset
    :param test: Test datainstance
    :param k: k hyperparamater
    :return: predicted regression value
    """
    print('Train set: ' + repr(len(X)))
    print('Test set: ' + repr(len(test)))
    print('k: ' + repr(k))
    neighbours = getNeighbours(X, test, k)
    prediction = calculateValue(neighbours, k)
    return prediction


def main():
    # Predicting the age of a person based on other features
    # Features are standardized. Age 18-36 <=> 0-1
    # Accuracy is tested with 0.15 of tolerance
    X = []  # Training set
    test = []  # Test set
    split_ratio = 0.66
    k = 12
    loadDataset('fertility_diagnosis.csv', split_ratio, X, test)
    print('Predicted value: ' + repr(myknnregress(X, test[0], k)))

    # Uncomment below line to test accuracy
    getAccuracy(X, test, k)

    # Uncomment below lines for testing accuracy for diff values of k
    # observations = {}
    # for i in range(1, len(X)):
    #     observations[i] = (getAccuracy(X, test, i))
    # for o in observations:
    #     print(repr(o) + ',' + repr(observations[o]))


main()
