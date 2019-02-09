#just suing topic modelling on a custom dataset
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset=’train’, shuffle = True)
newsgroups_test = fetch_20newsgroups(subset=’test’, shuffle = True)

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

dictionary = gensim.corpora.Dictionary(processed_docs)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 8, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 2)
 
#storing the result
import pandas as pd

mixture = [dict(lda_model[x]) for x in corpus1]
pd.DataFrame(mixture).to_csv("topic_mixture.csv")

top_words_per_topic = []
for t in range(lda_model.num_topics):
    top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t, topn = 5)])

pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("top_words.csv")

#where the actual analysis/graphing begins
import sklearn
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

import numpy as np

from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

fig = tools.make_subplots(rows=2, cols=3,
                          print_grid=False,
                          specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}],
                                 [ {'is_3d': True, 'rowspan':1}, None, None]])
scene = dict(
    camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=2.5, y=0.1, z=0.1)
    ),
    xaxis=dict(
        range=[-1, 4],
        title='Petal width',
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)',
        showticklabels=False, ticks=''
    ),
    yaxis=dict(
        range=[4, 8],
        title='Sepal length',
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)',
        showticklabels=False, ticks=''
    ),
    zaxis=dict(
        range=[1,8],
        title='Petal length',
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)',
        showticklabels=False, ticks=''
    )
)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target

estimators = {'k_means_iris_3': KMeans(n_clusters=3),
              'k_means_iris_8': KMeans(n_clusters=8),
              'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,
                                              init='random')}
fignum = 1
for name, est in estimators.items():
    est.fit(X)
    labels = est.labels_

    trace = go.Scatter3d(x=X[:, 3], y=X[:, 0], z=X[:, 2],
                         showlegend=False,
                         mode='markers',
                         marker=dict(
                                color=labels.astype(np.float),
                                line=dict(color='black', width=1)
        ))
    fig.append_trace(trace, 1, fignum)
    
    fignum = fignum + 1

y = np.choose(y, [1, 2, 0]).astype(np.float)

trace1 = go.Scatter3d(x=X[:, 3], y=X[:, 0], z=X[:, 2],
                      showlegend=False,
                      mode='markers',
                      marker=dict(
                            color=y,
                            line=dict(color='black', width=1)))
fig.append_trace(trace1, 2, 1)

fig['layout'].update(height=900, width=900,
                     margin=dict(l=10,r=10))

fig['layout']['scene1'].update(scene)
fig['layout']['scene2'].update(scene)
fig['layout']['scene3'].update(scene)
fig['layout']['scene4'].update(scene)
fig['layout']['scene5'].update(scene)

py.iplot(fig)