import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.cluster import KMeans , AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from nltk.tokenize import word_tokenize
from string import punctuation
# from unidecode import unidecode
# from contractions import fix
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

import os
import neptune

from dotenv import load_dotenv
import joblib
import yaml

class_labels = {'politics':0, 'sport':1, 'tech':2, 'entertainment':3, 'business':4}

## Data Preprocessing
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

vectorizer = joblib.load('assets/TF-IDF_v1.0.0.joblib')

def final_preprocessing(text):
    lemmatized = preprocess_data(text)
    vector = vectorizer.transform([lemmatized])
    return vector

if __name__ == '__main__':
    print(final_preprocessing("""Microsoft seeking spyware trojan Microsoft is investigating a trojan program that attempts to switch off the firm's anti-spyware software. The spyware tool was only released by Microsoft in the last few weeks and has been downloaded by six million people. Stephen Toulouse, a security manager at Microsoft, said the malicious program was called Bankash-A Trojan and was being sent as an e-mail attachment. Microsoft said it did not believe the program was widespread and recommended users to use an anti-virus program. The program attempts to disable or delete Microsoft's anti-spyware tool and suppress warning messages given to users. It may also try to steal online banking passwords or other personal information by tracking users' keystrokes. Microsoft said in a statement it is investigating what it called a criminal attack on its software. Earlier this week, Microsoft said it would buy anti-virus software maker Sybari Software to improve its security in its Windows and e-mail software. Microsoft has said it plans to offer its own paid-for anti-virus software but it has not yet set a date for its release. The anti-spyware program being targeted is currently only in beta form and aims to help users find and remove spyware - programs which monitor internet use, causes advert pop-ups and slow a PC's performance."""))
