from textblob import TextBlob
import zipfile


def read()-> tuple:
    """
    Read from files yelp100Reviews.zip, AFFIN-111.txt, positive.txt, negative.txt
    :return: list of reviews, dict of word scores from AFFIN-111.txt, dict of pos and neg words from positive.txt
    & negative.txt
    """
    reviews = []
    with zipfile.ZipFile('yelp100Reviews.zip', 'r') as f:
        for i in range(100):
            with f.open("yelp100Reviews/%s.txt" % (i+1), 'r') as r:
                reviews.append(r.read().decode('UTF-8'))

    affin = {}
    with open('AFFIN-111.txt', 'r') as f:
        for line in f:
            word = line.replace('\n','').split('\t')
            affin[word[0]] = int(word[1])

    pos_neg = {"positive": [],"negative": []}
    with open('positive.txt', 'r') as f:
        for line in f:
            if line[0] != ';':
                word = line.replace('\n','')
                pos_neg["positive"].append(word)
    with open('negative.txt', 'r') as f:
        for line in f:
            if line[0] != ';':
                word = line.replace('\n','')
                pos_neg["negative"].append(word)
    return reviews, affin, pos_neg


def textblob_sentiment(review: str)-> float:
    """
    :param review: Single review
    :return: positive or negative score for review
    """
    return TextBlob(review).sentiment.polarity


def affin_sentiment(review: str, affin: dict)-> int:
    """
    :param review: Single review
    :param affin: dict of word scores from AFFIN-111.txt
    :return: cumulative score of review
    """
    score = 0
    for word in review.replace('\n', ' ').split(' '):
        if word in affin:
            score += affin[word]
    return score


def positive_negative(review: str, pos_neg: dict)-> int:
    """
    :param review: Single review
    :param pos_neg: dict of pos and neg words from positive.txt
    :return: net score positive - negative
    """
    pos = 0
    neg = 0
    for word in review.replace('\n', ' ').split(' '):
        if word in pos_neg["positive"]:
            pos += 1
        elif word in pos_neg["negative"]:
            neg += 1
    return pos - neg


def main():
    reviews, affin, pos_neg = read()
    print("Review \t AFFIN-Sentiment \t Pos_Neg-Sentiment \t Textblob-Sentiment")
    for i in range(len(reviews)):
        print("%-5s \t %-15s \t %-15s \t %s" % (i+1, affin_sentiment(reviews[i], affin), positive_negative(reviews[i], pos_neg), round(textblob_sentiment(reviews[i]), 2)))


main()
