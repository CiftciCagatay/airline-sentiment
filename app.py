# mongo.py

import json
from flask import Flask
from flask import jsonify
from flask import request
from flask_pymongo import PyMongo
from watson_developer_cloud import LanguageTranslatorV3, NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
    import Features, SentimentOptions
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config['MONGO_DBNAME'] = 'restdb'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/restdb'

mongo = PyMongo(app)

language_translator = LanguageTranslatorV3(
    version='2018-05-01',
    username='4b671883-de20-48ab-8cd1-071c8feee771',
    password='1SNJMiNm8Jx2',
    url='https://gateway.watsonplatform.net/language-translator/api'
)

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2018-03-16',
    username='b24cd55b-7fd8-4e9f-bb51-df0f5a9b7528',
    password='HYJiZjyunBZ6'
)


@app.route('/', methods=['GET'])
def get_all_tweets():
    tweet = mongo.db.tweets
    output = []
    for s in tweet.find():
        output.append(
            {'tweetid': s['tweetid'], 'text': s['text'], 'text_en': s['text_en'], 'username': s['username'], 'language': s['language'], 'category': s['category'], 'sentiment': s['sentiment']})
    return jsonify({'result': output})


@app.route('/star/', methods=['GET'])
def get_one_star(name):
    star = mongo.db.stars
    s = star.find_one({'name': name})
    if s:
        output = {'name': s['name'], 'distance': s['distance']}
    else:
        output = "No such name"
    return jsonify({'result': output})


@app.route('/', methods=['POST'])
def add_star():
    tweet = mongo.db.tweets
    text = request.json['text']
    username = request.json['username']
    tweetid = request.json['tweetid']
    # category = request.json['category']

    language = language_translator.identify(
        text).get_result()['languages'][0]['language']

    text_en = ''

    if language == 'en':
        text_en = text
    else:
        text_en = language_translator.translate(
            text=text,
            source=language,
            target='en').get_result()['translations'][0]['translation']

    sentiment = natural_language_understanding.analyze(
        text=text_en,
        features=Features(
            sentiment=SentimentOptions(
            ))).get_result()['sentiment']['document']['label']

    if sentiment == 'negative':
        ### Modele text_en i ver
        ### category = model.predict(text_en)
    else:
        ### nothing


    new_tweet = {'text': text, 'tweetid': tweetid,
                 'text_en': text_en, 'username': username, 'language': language, 'category': category, 'sentiment': sentiment}

    _id = tweet.insert_one(new_tweet)

    output = {'tweetid': new_tweet['tweetid'], 'text_en': new_tweet['text_en'],
              'text': new_tweet['text'], 'language': new_tweet['language'], 'user': new_tweet['username'], 'sentiment': new_tweet['sentiment'], 'category': new_tweet['category']}

    return jsonify({'result': output})


if __name__ == '__main__':
    app.run(debug=True)
