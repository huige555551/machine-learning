import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

np.random.seed(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def create_data(m1: list, m2: list, c1: list, c2: list)-> tuple:
    train1 = np.random.multivariate_normal(m1, c1, 1500)
    test1 = np.random.multivariate_normal(m1, c1, 500)

    train2 = np.random.multivariate_normal(m2, c2, 1500)
    test2 = np.random.multivariate_normal(m2, c2, 500)

    # plt.plot(train1, train2, "1")
    # plt.title("Multivariate data distribution")
    # plt.grid(True)
    # plt.show()

    train_label1 = np.zeros(1500)
    test_label1 = np.zeros(500)

    train_label2 = np.ones(1500)
    test_label2 = np.ones(500)

    training_data = np.vstack([train1, train2])
    test_data = np.vstack([test1, test2])

    training_label = np.concatenate((train_label1, train_label2))
    test_label = np.concatenate((test_label1, test_label2))

    return training_data, training_label, test_data, test_label


def logistic_regression(data: list, labels: list, max_iter: int, learning_rate: float)-> list:
    weights = np.zeros(2)
    for i in range(max_iter):
        predictions = sigmoid(np.dot(data, weights))
        error = labels - predictions
        gradient = np.dot(data.T, error)
        weights += learning_rate * gradient
    return weights


def calc_metrics(test_data, test_label, weights):
    confusion_matrix = [{'correct': 0, 'incorrect': 0}, {'correct': 0, 'incorrect': 0}]
    acc = 0
    predictions = []
    for i in range(len(test_data)):
        prediction = round(sigmoid(np.dot(test_data[i], weights)))
        predictions.append(prediction)
        if test_label[i] == prediction:
            confusion_matrix[int(test_label[i])]['correct'] += 1
            acc += 1
        else:
            confusion_matrix[int(test_label[i])]['incorrect'] += 1
    return acc, confusion_matrix, predictions


def main():
    mean1 = [1, 0]
    mean2 = [0, 1.5]
    cov1 = [[1, 0.75], [0.75, 1]]
    cov2 = [[1, 0.75], [0.75, 1]]
    steps = 100
    learning_rate = 0.001
    training_data, training_label, test_data, test_label = create_data(mean1, mean2, cov1, cov2)
    weights = logistic_regression(training_data, training_label, steps, learning_rate)
    acc, confusion_matrix, predictions = calc_metrics(test_data, test_label, weights)
    fpr, tpr, th = metrics.roc_curve(test_label, predictions)
    auc = metrics.roc_auc_score(test_label, predictions)
    print('Accuracy: ' + repr(acc/len(test_data)*100) + '%')
    print('AUC: ' + repr(auc))
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


main()