import re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
stop_words = stopwords.words('english')


data = [] # We'll shove some data here soon


# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

