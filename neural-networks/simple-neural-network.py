import numpy as np
from sklearn.datasets import fetch_mldata


np.random.seed(12)


def load_data()-> tuple:
    mnist = fetch_mldata("MNIST original")
    data, labels = mnist["data"], mnist["target"]
    test = data[60000:]
    test_labels = labels[60000:]
    data = data[:60000]
    labels = labels[:60000]
    f_train_data = []
    f_train_labels = []
    f_test_data = []
    f_test_labels = []
    for i in range(len(labels)):
        if labels[i] < 5:
            f_train_data.append(data[i])
            f_train_labels.append(labels[i])
    for i in range(len(test_labels)):
        if test_labels[i] < 5:
            f_test_data.append(test[i])
            f_test_labels.append(test_labels[i])
    return np.array(f_train_data), np.array(f_train_labels), np.array(f_test_data), np.array(f_test_labels)


def convert_to_one_hot(train: list)-> tuple:
    one_hot_train = []
    for i in range(len(train)):
        temp = np.zeros(5)
        temp[int(train[i])] = 1
        one_hot_train.append(temp)
    return np.array(one_hot_train)


def add_bias_unit(X, column=True):
    if column:
        bias_added = np.ones((X.shape[0], X.shape[1] + 1))
        bias_added[:, 1:] = X
    else:
        bias_added = np.ones((X.shape[0] + 1, X.shape[1]))
        bias_added[1:, :] = X
    return bias_added


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def dev_sigmoid(x):
    f = sigmoid(x)
    return f * (1 - f)


def init_weights(n_features, n_hidden, n_output):
    # w1 = np.random.uniform(-0.5, 0.5, size=(n_features+1)*n_hidden).reshape(n_hidden, n_features + 1)
    # w2 = np.random.uniform(-0.5, 0.5, size=(n_hidden+1)*n_output).reshape(n_output, n_hidden + 1)
    w1 = np.random.randn((n_features + 1) * n_hidden).reshape(n_hidden, n_features + 1)
    w2 = np.random.randn((n_hidden + 1) * n_output).reshape(n_output, n_hidden + 1)
    return w1, w2


def train_neural_network(data: np.ndarray, labels: np.ndarray, max_iter: int, learning_rate: float, hidden_units)-> object:
    # print("data", data.shape)
    w1, w2 = init_weights(len(data[0]), hidden_units, 5)
    # print("w1", w1.shape)
    # print("w2", w2.shape)
    for i in range(max_iter):
        mini_batches = np.array_split(range(data.shape[0]), 50)
        for batch in mini_batches:
            # forward
            a1 = add_bias_unit(data[batch])
            z2 = np.dot(w1, a1.T)
            a2 = sigmoid(z2)
            a2 = add_bias_unit(a2, column=False)
            z3 = np.dot(w2, a2)
            a3 = sigmoid(z3)

            # backward
            sigma3 = a3 - labels[batch, :].T
            z2 = add_bias_unit(z2, column=False)
            sigma2 = np.dot(w2.T, sigma3) * dev_sigmoid(z2)
            sigma2 = sigma2[1:, :]
            grad1 = np.dot(sigma2, a1)
            grad2 = np.dot(sigma3, a2.T)

            w1 -= learning_rate * grad1
            w2 -= learning_rate * grad2
    return w1, w2


def calc_metrics(test_data, test_label, w1, w2)-> tuple:
    acc = 0
    predictions = []
    confusion_matrix = np.zeros((5, 5))
    # accuracy
    a1 = add_bias_unit(test_data)
    z2 = np.dot(w1, a1.T)
    a2 = sigmoid(z2)
    a2 = add_bias_unit(a2, column=False)
    z3 = np.dot(w2, a2)
    y_pred = np.argmax(z3, axis=0)

    for i in range(len(test_data)):
        if test_label[i] == y_pred[i]:
            acc += 1
        confusion_matrix[y_pred[i]][int(test_label[i])] += 1
    # prediction
    precision = 0
    for i in range(5):
        precision += confusion_matrix[i][i] / sum(confusion_matrix[:,i])
    precision /= 5
    # recall
    recall = 0
    for i in range(5):
        recall += confusion_matrix[i][i] / sum(confusion_matrix[i])
    recall /= 5
    return acc, predictions, confusion_matrix, precision, recall


def main():
    # 10000 iterations takes too long to run
    steps = 100
    learning_rate = 0.001
    train_data, train_labels, test_data, test_labels = load_data()
    train_labels = convert_to_one_hot(train_labels)
    w1, w2 = train_neural_network(train_data, train_labels, steps, learning_rate, 50)
    acc, predictions, confusion_matrix, precision, recall = calc_metrics(test_data, test_labels, w1, w2)
    print("Confustion matrix: \n", confusion_matrix)
    print("Precsion: " + repr(round(precision, 4)))
    print("Recall: " + repr(round(recall, 4)))
    print("Accuracy: " + repr(round(acc/len(test_data), 4) * 100) + " %")


main()
