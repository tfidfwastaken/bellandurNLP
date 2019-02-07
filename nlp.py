import re
import numpy as np
import pandas as pd
from pprint import pprint

import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#import
df = pd.read_json('reps.json')

data = df.text.values.tolist()

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

# tokenizing
def tokenize(texts):
    for text in texts:
        yield (gensim.utils.simple_preprocess(str(text), deacc=True))

tokens = list(tokenize(data))

print(tokens[:15])
