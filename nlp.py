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

# Building trigram and bigram models
bigram = gensim.models.Phrases(tokens, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[tokens], threshold=100)

# Phraser makes the trained model work faster because it uses only what's needed
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# removing stopwords
def remove_stopwords(tokens):
    return [[word for word in simple_preprocess(str(token)) if word not in stop_words] for token in tokens]

# make bigrams and trigrams
def make_bigrams(tokens):
    return [bigram_mod[token] for token in tokens]

def make_bigrams(tokens):
    return [trigram_mod[bigram_mod[token]] for token in tokens]

# lemmatize the tokens
def lemmatize(tokens, pos_tags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    token_output = []
    for token in tokens:
        doc = nlp(" ".join(token))
        token_output.append([token.lemma_ for token in doc if token.pos_ in pos_tags])
    return token_output

tokens_nostop = remove_stopwords(tokens)
bigrams = make_bigrams(tokens_nostop)
# later
# trigrams = make_trigrams(tokens_nostop)
nlp = spacy.load('en', disable=['ner', 'parser'])

data_lemmatized = lemmatize(bigrams)

print(bigrams[:1])
