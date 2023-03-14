from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
docs[:5]

print('total num doc: ', len(docs))

model = BERTopic()
topics, probabilities = model.fit_transform(docs)

print(model.get_topic(5))


model.visualize_topics()


