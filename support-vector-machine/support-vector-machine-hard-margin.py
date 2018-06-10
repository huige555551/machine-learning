import zipfile
from PIL import Image
import numpy as np
import cvxopt


def read():
    """
    Read faces from att_faces.zip
    :return: features split in two, and labels
    """
    dataset = {}
    with zipfile.ZipFile('att_faces.zip', 'r') as f:
        for i in range(1, 41):
            for j in range(1, 11):
                with f.open("s%d/%d.pgm" % (i, j), 'r') as r:
                    if i not in dataset:
                        dataset[i] = []
                    dataset[i].append(np.array(Image.open(r)).ravel())
    for i in dataset:
        dataset[i] = np.split(np.array(dataset[i]), 2)
    x = np.array(list(dataset.values()))
    y = np.array(list(dataset))
    return x, y


def convex_optimizer(x, y):
    """
    Quadratic problem solver
    :param x: training data
    :param y: one-vs-all labels
    :return: solution to quadratic problem
    """
    # optimization
    K = y[:, None] * x
    K = np.dot(K, K.T)
    P = cvxopt.matrix(K)
    q = -np.ones((200, 1))
    q = cvxopt.matrix(q)
    G = -np.eye(200)
    G = cvxopt.matrix(G)
    h = np.zeros(200)
    h = cvxopt.matrix(h)
    A = y.reshape(1, -1)
    A = cvxopt.matrix(A)
    b = np.zeros(1)
    b = cvxopt.matrix(b)
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas


def get_weights_bias(alphas, x, y):
    """
    :param alphas: Solution to quadratic problem
    :param x: training data
    :param y: one-vs-all labels
    :return: weights and bias
    """
    # weights
    w = np.sum(alphas * y[:, None] * x, axis=0)
    cond = (alphas > 0).reshape(-1)
    b = y[cond] - np.dot(x[cond], w)
    if len(b) == 0:
        bias = 0
    else:
        bias = b[0]
    return w, bias


def predict(w, bias, x, y, cv):
    """
    :param w: weights
    :param bias: Bias
    :param x: test data
    :param y: one-vs-all test labels
    :param cv: cross-validation index 0,1
    :return: accuracy
    """
    # prediction
    acc = 0
    for i in range(40):
        for j in range(5):
            preds = []
            for k in range(40):
                pred = np.dot(x[i][cv][j], w[k]) + bias[k]
                preds.append(pred)
            if y[i] == (np.argmax(preds) + 1):
                acc += 1
    return acc / 200 * 100


def support_vector_machine(x, y, complete_x, cv):
    """
    :param x: training data
    :param y: one-vs-all labels
    :param complete_x: complete training and test data
    :param cv: cross-validation index 0,1
    :return: accuracies of all 10 classifiers
    """
    ova_y = {}
    for i in range(len(y)):
        pre = -np.ones(i * 5)
        mid = np.ones(5)
        post = -np.ones((len(y) - i - 1) * 5)
        ova_y[i] = np.hstack((pre, mid, post))
    alphas = []
    for i in range(len(ova_y)):
        alphas.append(convex_optimizer(x=x, y=ova_y[i]))
    weights = []
    bias = []
    for i in range(len(ova_y)):
        tup = get_weights_bias(alphas=alphas[i], x=x, y=ova_y[i])
        weights.append(tup[0])
        bias.append(tup[1])
    acc = predict(w=weights, bias=bias, x=complete_x, y=y, cv=cv)
    return acc


def cross_validation_fit(x, y):
    """
    :param x: training data
    :param y: one-vs-all labels
    :return: cross validation accuracy
    """
    # Train 0 Test 1
    x1 = x[:, 0]
    x1 = x1.reshape([200, 10304])
    acc1 = support_vector_machine(x=x1, y=y, complete_x=x, cv=1)

    # Train 1 Test 0
    x2 = x[:, 1]
    x2 = x2.reshape([200, 10304])
    acc2 = support_vector_machine(x=x2, y=y, complete_x=x, cv=0)

    # cross-validation accuracy
    return (acc1 + acc2) / 2


def main():
    x, y = read()

    # cross-validation accuracy
    acc = cross_validation_fit(x, y)
    print("Accuracy")
    print("%s%%" % acc)


main()
