# -*- coding: utf-8 -*-

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split

from sklearn import preprocessing
Encode = preprocessing.LabelEncoder()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import SGDClassifier

import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

chatbot_df = pd.read_csv('Airline-Sentiment.csv')
x = chatbot_df.iloc[:, 1].values
y = chatbot_df.iloc[:, 0].values
x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=1, test_size=0.2)

result_cols = ["Classifier", "Accuracy"]
result_frame = pd.DataFrame(columns=result_cols)

# Any results you write to the current directory are saved as output.

classifiers = [SGDClassifier(
        n_iter = np.ceil(10**6 / len(x_train)),
        alpha=1e-05,
        penalty='elasticnet'
        )]
"""    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    MultinomialNB()"""

# hinge - l2 - SGDClassifier accuracy = 64.1067538126%
# hinge - l1 - SGDClassifier accuracy = 63.3986928105%
# squared_hinge - l2 - SGDClassifier accuracy = 45.8061002179%

for clf in classifiers:
    name = clf.__class__.__name__
    text_clf = Pipeline([
            ('vect', CountVectorizer(max_df=1.0, ngram_range=(1,2))), 
            ('tfidf', TfidfTransformer(norm='l2')),
            ('clf', clf),])

    from sklearn.model_selection import GridSearchCV
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        #'clf__n_iter': (10, 50, 80),
    }
    
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(x_train, y_train)
    
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    
    """
    text_clf.fit(x_train, y_train)
    
    predicted = text_clf.predict(x_test)
    acc = metrics.accuracy_score(y_test,predicted)
    print (name+' accuracy = '+str(acc*100)+'%')
    acc_field = pd.DataFrame([[name, acc*100]], columns=result_cols)
    result_frame = result_frame.append(acc_field)
        """
    
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=result_frame, color="r")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()
    
