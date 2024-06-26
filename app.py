from fastapi import FastAPI
from joblib import load
from datetime import datetime
from mangum import Mangum
import boto3
import uuid

from fastapi.responses import JSONResponse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.cluster import KMeans , AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from nltk.tokenize import word_tokenize
from string import punctuation
from unidecode import unidecode
from contractions import fix
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC
import re
import nltk
import pickle
from nltk.corpus import stopwords
import neptune
import yaml

nltk.data.path.append("nltkdata")

class_labels = {'politics':0, 'sport':1, 'tech':2, 'entertainment':3, 'business':4}

# Data Preprocessing
stopwords_list = stopwords.words('english')
len(stopwords_list)

def preprocess_data(text):
    text = text.lower()
    text = text.replace("\n"," ").replace("\t"," ")
    text = re.sub("\s+"," ",text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # tokens
    tokens = word_tokenize(text)
    
    data = [i for i in tokens if i not in punctuation]
    data = [i for i in data if i not in stopwords_list]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    final_text = []
    for i in data:
        word = lemmatizer.lemmatize(i)
        final_text.append(word)
        
    return " ".join(final_text)

vectorizer = load('assets/TF-IDF_v1.0.0.joblib')

def final_preprocessing(text):
    lemmatized = preprocess_data(text)
    vector = vectorizer.transform([lemmatized])
    return vector

dynamodb = boto3.resource('dynamodb') 
app = FastAPI()
handler=Mangum(app)

# Load your model
model = load('assets/Logistic Regression_v1.0.0.joblib')
classes = ['politics', 'sport', 'tech', 'entertainment', 'business']

uncertainty_threshold = 0.5

def upload_data_to_dynamodb(data, table_name):
    table = dynamodb.Table(table_name)
    response = table.put_item(
        Item={ 
            'id': str(uuid.uuid4()),
            'timestamp': data['timestamp'], 
            'text': data['text'], 
            'prediction': data['prediction'], 
            'confidence': str(data['confidence']), 
        } 
    )
    return response

@app.get("/")
def home():
    return "Welcome to BBC News Text Classification API"

@app.get("/predict")
async def predict(text: str):
    vector = final_preprocessing(text)
    probs = model.predict_proba(vector)[0]  # Get prediction probabilities
    pred_index = probs.argmax()

    prediction = classes[pred_index]
    confidence = probs[pred_index]

    is_uncertain = confidence < uncertainty_threshold
    is_uncertain = bool(is_uncertain)  # Explicit conversion to Python bool

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = {
        "text": text,
        "prediction": prediction,
        "confidence": float(confidence),  # Convert to Python float for JSON serialization
        "is_uncertain": is_uncertain,
        "timestamp": timestamp  # Include the formatted timestamp in the response
    }
    
    response = upload_data_to_dynamodb(result, 'uncertainty_textcls')
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
