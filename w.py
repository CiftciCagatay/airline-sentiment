#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 03:29:38 2018

@author: ogrenci
"""

import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
from autocorrect import spell
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split

# Import Dataset
dataset = pd.read_csv('Airline-Sentiment.csv')
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 0].values

# Vars.
stemmer = PorterStemmer()
corpus = []
# Regex string to remove hashtags, punc. etc.
regex = '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)'
# English dictionary file for splitting words
dictionary = Counter(words(open('words.txt').read()))
max_word_length = max(map(len, dictionary))
total = float(sum(dictionary.values()))

# Process all the comments in dataset
for i in range(0, len(X)):
    corpus.append(process_comment(X[i]))
    print(i)

# Vectorize each comment by freq. 'customer' 'customer service'
tfidf_vect = TfidfVectorizer(ngram_range=(1,2))
X_tfidf = tfidf_vect.fit_transform(raw_documents=X).toarray()

# Split test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2)

# Create and train classifier
clf = MultinomialNB()
clf = clf.fit(X_train, y_train)

# Predict 
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)

features = tfidf_vect.get_feature_names()
df = pd.DataFrame(X_tfidf, columns=features)

highest_frequency = df.max()

def process_comment(comment):
    # Remove punc. hashtags etc.
    comment = re.sub(regex, ' ', comment)

    comment = comment.lower()
    comment = word_tokenize(comment)
    
    # Split words
    new = []
    for i in range(len(comment)):
        for word in viterbi_segment(comment[i])[0]:
            new.append(word)
            
    comment = new
    
    # Remove stopwords and stem, spellcheck every word in comments
    comment = [stemmer.stem(spell(word)) for word in comment if not word in stopwords.words('english')]
    
    # Normalize comment
    comment = ' '.join(comment)
    return comment

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



