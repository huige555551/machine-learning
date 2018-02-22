import csv


def loadDataset(filename: str)->tuple:
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
        cleaned_dataset = []
        for instance in dataset:
            if instance[11] != '?':
                cleaned_dataset.append(instance)
        return cleaned_dataset[:4000], cleaned_dataset[4000:]


def split_data_by_classes(training: list)->dict:
    """
    Split data into poisonous and edible classes
    :param training: training set
    :return: class separated instances
    """
    seperated = {}
    for instance in training:
        if instance[0] not in seperated:
            seperated[instance[0]] = []
        seperated[instance[0]].append(instance)
    return seperated


def calculate_probabilities(separated: dict)-> tuple:
    """
    Calculating prob for each feature
    :param separated: class separated instances
    :return: dependent and independent probabilities for each feature
    """
    d_prob = {}
    for k, v in separated.items():
        d_prob[k] = []
        for instance in v:
            i = 0
            for feature in instance:
                if i >= len(d_prob[k]):
                    d_prob[k].append({})
                if feature not in d_prob[k][i]:
                    d_prob[k][i][feature] = 0
                d_prob[k][i][feature] += 1
                i += 1

    # feature probabilities given class
    for k, v in d_prob.items():
        class_count = v[0][k]
        for feature in v:
            for attribute in feature.keys():
                feature[attribute] /= class_count

    # independent feature probabilities
    i_prob = []
    for k, v in separated.items():
        for instance in v:
            i = 0
            for feature in instance:
                if i >= len(i_prob):
                    i_prob.append({})
                if feature not in i_prob[i]:
                    i_prob[i][feature] = 0
                i_prob[i][feature] += 1
                i += 1
    sum = 0
    for c in i_prob[0].keys():
        sum += i_prob[0][c]
    for item in i_prob:
        for c in item.keys():
            item[c] /= sum

    return d_prob, i_prob


def calculate_z(dependent_prob: list, independent_prob: list)->float:
    """
    Calculating Z scaling factor
    :param dependent_prob: class dependent feature prob
    :param independent_prob: class independent feature prob
    :return: scaling factor Z
    """
    Z = 0
    for c in dependent_prob.keys():
        dependent_prob[c].pop(0)
    for c in dependent_prob.keys():
        for f in dependent_prob[c]:
            for k in f.keys():
                Z += independent_prob[0][c]*f[k]
    return Z


def predict_class(independent_prob: list, dependent_prob: list, instance: list, Z: float)->str:
    """
    Predict label from prior prob
    :param independent_prob: class independent feature prob
    :param dependent_prob: class dependent feature prob
    :param instance: test instance
    :param Z: scaling factor
    :return: predicted label
    """
    product = {'e': 1, 'p': 1}
    for i, f in enumerate(instance):
        if f in dependent_prob['p'][i]:
            product['p'] *= dependent_prob['p'][i][f]
    for i, f in enumerate(instance):
        if f in dependent_prob['e'][i]:
            product['e'] *= dependent_prob['e'][i][f]
    product['e'] = independent_prob[0]['e']*product['e']/Z
    product['p'] = independent_prob[0]['p']*product['p']/Z
    if product['e'] >= product['p']:
        return 'p'
    else:
        return 'e'


def main():
    training, test = loadDataset('agaricus-lepiota.data.csv')
    separated = split_data_by_classes(training)
    dependent_prob, independent_prob = calculate_probabilities(separated)
    Z = calculate_z(dependent_prob, independent_prob)
    acc = 0
    confusion_matrix = {'p': {'correct':0, 'incorrect':0}, 'e': {'correct':0, 'incorrect':0}}
    for instance in test:
        result = instance.pop(0) # remove first feature from instance as it is class label
        prediction = predict_class(independent_prob, dependent_prob, instance, Z)
        # print('prediction: ' + prediction + ' | actual: ' + result)
        if result == prediction:
            confusion_matrix[result]['correct'] += 1
            acc += 1
        else:
            confusion_matrix[result]['incorrect'] += 1
    print('Accuracy: ' + repr(acc/len(test)*100) + '%')
    print(confusion_matrix)


main()
