import math
import string
from json.decoder import NaN

import numpy
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report


def clean_up_text(row):
    txt = row['text']

    txt = txt.lower()
    txt = txt.translate(txt.maketrans('', '', string.punctuation))

    txt_arr = txt.split(' ')
    stop_words = stopwords.words('english')
    txt_arr = [word for word in txt_arr if not word in stop_words]

    txt = ' '.join(txt_arr)

    return txt

emotions = [
    "Angry",
    "Happy",
    "Sad",
    "Neutral",
    "Surprise",
    "Worried"
]

transcript = "../data/label-transcription.csv"
open_vocab_labels = "../data/final-openset-english.csv"
discrete_labels = "../data/label-disdim.csv"

### Reading data into dataframes and combining into on dataframe
transcript = pd.read_csv(transcript, usecols=[0, 2])
labeled_data = pd.read_csv(discrete_labels)

# Rename for comprehension; drop NaN / empty values
transcript.rename(columns={'english': 'text'}, inplace=True)
transcript.drop(transcript[transcript['text'].isna()].index, inplace=True)

# Merge the discrete emotion labeled data with our transcript by the sample name
data = labeled_data.merge(transcript, left_on="name", right_on="name", how="inner")

print(data.head(10).to_string())
print(data.shape)
print()

### Preprocessing
# Strip punctuation and make line lower case. Remove stop words.
data['cleaned_text'] = data.apply(clean_up_text, axis=1)

### Token features: looking to get Bag of Words and TF-IDF as features
# Grab the number of tokens per line in data and then sum for total tokens

# print(data.head().to_string())
le = LabelEncoder()
data['emotion_code'] = le.fit_transform(data['discrete'])

# Name English Discrete Valence
x = data.drop(['discrete', 'name', 'text', 'valence'], axis=1)
y = data[['emotion_code']]

# print(x)
print(x.shape)
print(x.columns.values)

# print(y)
# print(y.shape)

# split data for training sets and test sets
# we have two sets per stage because we need a set of features to guide us to the target value
x_train, x_test, y_train, y_test = train_test_split(data['cleaned_text'], data['emotion_code'], test_size=0.2, random_state=42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
x_train = tfidf_vectorizer.fit_transform(x_train)

x_test = tfidf_vectorizer.transform(x_test)

print(data.shape)
print(x_train.toarray())

# Random Forest Classifier
rand_forest = RandomForestClassifier(random_state=1, n_estimators=100)


rand_forest = rand_forest.fit(x_train, y_train)
y_pred_rfc = rand_forest.predict(x_test)

print(classification_report(y_test, y_pred_rfc, zero_division=numpy.nan))