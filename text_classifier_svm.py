#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 06:05:21 2018

@author: ogrenci
"""


# Import dataset
import pandas as pd
dataset = pd.read_csv('Airline-Sentiment.csv')
x = dataset.iloc[:, 1].values
y = dataset.iloc[:, 0].values

# Split test and train data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.8)

# Tokenize text data 
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(raw_documents=x_train)

# Use frequency rather than occurence
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

# Train classifier
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier().fit(x_train_tfidf, y_train)

# Test
x_test_counts = count_vect.transform(x_test)
x_test_tfidf = tfidf_transformer.transform(x_test_counts)
y_pred = clf.predict(x_test_tfidf)

# Calculate accuracy
import numpy as np
np.mean(y_pred == y_test)