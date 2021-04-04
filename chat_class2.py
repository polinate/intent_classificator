import pandas as pd
import numpy as np
import os
import sys
import string
import csv
from pymystem3 import Mystem
import pickle
import json
import random
from flask import Flask, request, abort, jsonify




all_punctuations = string.punctuation + '‘’,:”][],' 

m = Mystem()

def lemm(row):
    l = m.lemmatize(row)
    l.remove('\n')
    
    return ', '.join(l)


def punc_remover(raw_text):
    no_punct = "".join([i for i in raw_text if i not in all_punctuations])
    return no_punct

#lemmer = nltk.stem.WordNetLemmatizer()
def lem(words):
    return " ".join([lemm(word) for word in words.split()])

def text_cleaner(raw):
    cleaned_text = punc_remover(raw.lower())
    return cleaned_text



def predict( statement): 
    statement = text_cleaner(statement)
    statement = lem(statement)

    tfidf_vectorizer = pickle.load(open('tfidftransformer.pkl', "rb"))
    tfidf_train1 = tfidf_vectorizer.transform([statement])
    #svmClassifier = joblib.load('pac_chatbot_classifier.pkl')
    loaded_model = pickle.load(open('finalized_model_bnb.sav', 'rb'))
    pred = loaded_model.predict(tfidf_train1)
    return pred[0]
    



def get_response(statement):
    intents = json.loads(open('intents.json',encoding='utf-8').read())
    list_of_intents = intents['intents']
    tag = predict(statement)
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


flask_app = Flask(__name__)

@flask_app.route('/predict', methods=['POST'])
def create_answer():
    message = request.args['message']
    #message=[message]
    res = get_response(message)
    result = {
    "error" : "0",
    "message" : "success",
    "answer" : res}
    return flask_app.response_class(response=json.dumps(result), mimetype='application/json')
     
  
  
if __name__ == "__main__":
   flask_app.run(host='0.0.0.0', port=5000, debug=True)
