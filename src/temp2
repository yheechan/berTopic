def get_data(path, y):

	texts = []
	labels = []

	with open(path, encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			texts.append(line)
			labels.append(y)

	return texts, labels

# load data
paths = ['../data/train.negative.csv',
			'../data/train.non-negative.csv',
			'../data/test.negative.csv',
			'../data/test.non-negative.csv']

# get data as list of dict with text and label
train_neg_x_ls, train_neg_y_ls = get_data(paths[0], 1)
train_non_x_ls, train_non_y_ls = get_data(paths[1], 0)
test_neg_x_ls, test_neg_y_ls = get_data(paths[2], 1)
test_non_x_ls, test_non_y_ls = get_data(paths[3], 0)

train_data = train_neg_x_ls + train_non_x_ls
test_data = test_neg_x_ls + test_non_x_ls

print('train data length: ', len(train_data))


'''
from sklearn.datasets import fetch_20newsgroups
train_data = fetch_20newsgroups(subset='all')['data']
print('train data length2: ', len(train_data))
'''

'''
# process data
import text_process as tp
import spacy
from nltk.corpus import stopwords

# train_data = [tp.remove_punctuation(sentence) for sentence in train_data]
# train_data = [sentence.lower() for sentence in train_data]
train_data = [sentence.strip().split() for sentence in train_data]

train_data = [' '.join(sentence) for sentence in train_data]

nlp = spacy.load('en_core_web_sm')
train_data = [[token.lemma_ for token in nlp(sentence)] for sentence in train_data]

stop_words = set(stopwords.words('english'))
train_data = [ [word for word in sentence if not word in stop_words and word != ' '] for sentence in train_data]

train_data = [' '.join(sentence) for sentence in train_data]
print(len(train_data))

'''





# BRING MODEL & ENCODE DATA to EMBEDDING VALUE
'''
distilbert gives good balance between speed and performance
supports multi-lingual models
'''
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(train_data, show_progress_bar=True)






# REDUCE EMBEDDING DIMENSIONS using UMAP
import umap
umap_embeddings = umap.UMAP(n_neighbors=10,
							n_components=5,
							metric='cosine').fit_transform(embeddings)
print('\ndone with reducing dimension')






# CLUSTER EMBEDDINGS using HDBSCAN
'''
does not force data points meaning knows outliers
'''
import hdbscan
cluster = hdbscan.HDBSCAN(min_cluster_size=5,
							metric='euclidean',
							cluster_selection_method='eom').fit(umap_embeddings)

print('\ndone with clustering')







# PLOT CLUSTERS
import matplotlib.pyplot as plt
import pandas as pd

# Prepare data
umap_data = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
result = pd.DataFrame(umap_data, columns=['x', 'y'])
result['labels'] = cluster.labels_

# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
plt.colorbar()
plt.show()

print('\ndone with plot')






#c-TF-IDF
docs_df = pd.DataFrame(train_data, columns=["Doc"])
docs_df['Topic'] = cluster.labels_
docs_df['Doc_ID'] = range(len(docs_df))
docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count
  
tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(train_data))


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)

print(top_n_words[0][:10])
print()
print(top_n_words[2][:10])
print()
print(top_n_words[3][:10])
print()
print(top_n_words[4][:10])
print()
'''
print(top_n_words[5][:10])
print()
'''
