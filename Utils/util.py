
import string
from nltk.corpus import stopwords

def clean_up_text(row):
    txt = row['text']

    txt = txt.lower()
    txt = txt.translate(txt.maketrans('', '', string.punctuation))

    txt_arr = txt.split(' ')
    stop_words = stopwords.words('english')
    txt_arr = [word for word in txt_arr if not word in stop_words]

    txt = ' '.join(txt_arr)

    return txt
