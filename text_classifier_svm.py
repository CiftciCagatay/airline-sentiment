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

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderY = LabelEncoder()
y = labelEncoderY.fit_transform(y)

# Split test and train data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1,1))),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, random_state=42)),
])

text_clf.fit(x_train, y_train)

# Test
y_pred = text_clf.predict(x_test)

# Calculate accuracy
import numpy as np
np.mean(y_pred == y_test)

from sklearn.model_selection import GridSearchCV
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(x_train, y_train)

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
