import math


def read()-> dict:
    """
    Read from file
    :return: dictionary of entires in file
    """
    with open('reviews.txt', 'r') as f:
        x = f.read()
        return eval(x)


def getSimilarity(r1: str, r2: str, reviews: dict)-> float:
    """
    :param r1: Reviewer 1
    :param r2: Reviewer 2
    :param reviews: dictionary of entries from file
    :return: euclidean distance between reviewers r1 and r2 in reviews dict
    """
    similarity = 0.0
    if r1 in reviews and r2 in reviews:
        for key in reviews[r1]:
            if key in reviews[r2]:
                similarity += (reviews[r1][key] - reviews[r2][key])**2
    else:
        # Reviewers not present in file
        return 0.0
    return math.sqrt(similarity)


def getSimilarities(r1: str, reviews: dict)-> None:
    """
    :param r1: Reviewer 1
    :param reviews: dictionary of entries from file
    :return: euclidean distance between reviewer r1 and all other reviewers in reviews dict
    """
    if r1 in reviews:
        for key in reviews.keys():
            if r1 != key:
                print("%s \t\t %-20s %s" % (r1, key, getSimilarity(r1, key, reviews)))
    else:
        print("No such entry: %s" % r1)


def main():
    reviews = read()
    print(getSimilarity("Peter", "Trevor Chappell", reviews))
    getSimilarities("Peter", reviews)


main()
