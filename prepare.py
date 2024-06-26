import os
import nltk
if not os.path.exists('nltkdata'):
  os.mkdir('nltkdata')
  nltk.download('wordnet',download_dir='nltkdata')
  nltk.download('stopwords',download_dir='nltkdata')
  nltk.download('punkt',download_dir='nltkdata')
  nltk.download('omw-1.4',download_dir='nltkdata')