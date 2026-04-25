
from json.decoder import NaN
from keras.src.layers import LSTM, Dropout
from sklearn.model_selection import train_test_split

from Utils import util
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

transcript = "../data/text_training/label-transcription.csv"
open_vocab_labels = "../data/text_training/final-openset-english.csv"
discrete_labels = "../data/text_training/label-disdim.csv"

### Reading data into dataframes and combining into on dataframe
transcript = pd.read_csv(transcript, usecols=[0, 2])
labeled_data = pd.read_csv(discrete_labels)

# Rename for comprehension; drop NaN / empty values
transcript.rename(columns={'english': 'text'}, inplace=True)
transcript.drop(transcript[transcript['text'].isna()].index, inplace=True)

# Merge the discrete emotion labeled data with our transcript by the sample name
data = labeled_data.merge(transcript, left_on="name", right_on="name", how="inner")

### Preprocessing
# Strip punctuation and make line lower case. Remove stop words.
data['cleaned_text'] = data.apply(util.clean_up_text, axis=1)

vectorizer = CountVectorizer()
count = vectorizer.fit_transform(data['text'])
vocab = vectorizer.get_feature_names_out()
vocab_size = len(vocab)

# print(data.head().to_string())
le = LabelEncoder()
data['emotion_code'] = le.fit_transform(data['discrete'])

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(data['text'])

sequences = tokenizer.texts_to_sequences(data['text'])

max_length = 20
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = data['emotion_code'].values

embedding_dim = 64

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(64, return_sequences=False, activation='tanh'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', 'Precision', 'Recall']
)

model.summary()

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

model.fit(X, y, epochs=100, batch_size=32, verbose=1, validation_split=0.2)
