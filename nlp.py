#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 03:36:04 2018

@author: ogrenci
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Airline-Sentiment.csv')
x = dataset.iloc[:, 1].values
y = dataset.iloc[:, 0].values

import re
from autocorrect import spell
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()
stopwords = stopwords.words('english') + ['lt']
corpus = []

for i in range(0, len(x)):
    print(i)
    comment = x[i]
    comment = re.sub('[^a-zA-Z]', ' ', comment)
    comment = comment.lower()
    comment = word_tokenize(comment)
    comment = [stemmer.stem(word) for word in comment if not word in stopwords]
    comment = ' '.join(comment)
    corpus.append(comment)

from sklearn.feature_extraction.text import CountVectorizer
cnt_vect = CountVectorizer(ngram_range=(1,2))
x_cnts = cnt_vect.fit_transform(corpus).toarray()

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_cnts, y, test_size=0.2)

# Naive Bayes 
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier()
model = classifier.fit(x_train, y_train)

# Predict Class
y_pred = classifier.predict(x_test)

# Accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

features = cnt_vect.get_feature_names()
df = pd.DataFrame(x_cnts, columns=features)

highest_frequency = df.max()


def viterbi_segment(text):
    probs, lasts = [1.0], [0]
    for i in range(1, len(text) + 1):
        prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
                        for j in range(max(0, i - max_word_length), i))
        probs.append(prob_k)
        lasts.append(k)
    words = []
    i = len(text)
    while 0 < i:
        words.append(text[lasts[i]:i])
        i = lasts[i]
    words.reverse()
    return words, probs[-1]

def word_prob(word): return dictionary[word] / total
def words(text): return re.findall('[a-z]+', text.lower()) 

dictionary = Counter(words(open('words.txt').read()))

max_word_length = max(map(len, dictionary))

total = float(sum(dictionary.values()))

