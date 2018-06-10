# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 22:06:24 2018

@author: Vaishali Goyal
"""
from textblob import TextBlob
import zipfile


def read()-> tuple:
#read from the yelp file to return list of reviews and dictionary of words 
   
    reviews = []
    with zipfile.ZipFile('yelp100Reviews.zip', 'r') as f:
        for i in range(100):
            with f.open("yelp100Reviews/%s.txt" % (i+1), 'r') as r:
                reviews.append(r.read().decode('UTF-8'))
            
    #method 1:        
    affin = {}
    with open('AFFIN-111.txt', 'r') as f:
        for line in f:
            word = line.replace('\n','').split('\t')
            affin[word[0]] = int(word[1])
    return reviews, affin

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

def textblob_sentiment(review: str)-> float:
    """
    :param review: Single review
    :return: positive or negative score for review
    """
    return TextBlob(review).sentiment.polarity


def main():
    reviews, affin = read()
    print("Review \t AFFIN-Sentiment \t Textblob-Sentiment")
    for i in range(len(reviews)):
        print("%-5s \t %-15s \t %-15s" % (i+1, affin_sentiment(reviews[i], affin), round(textblob_sentiment(reviews[i]), 2)))


main()