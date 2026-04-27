from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd
from Utils import util

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

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
