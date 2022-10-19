import pickle
import re
from time import time

import numpy as np
import pandas as pd
import requests
import spacy
from gensim.models import KeyedVectors, Word2Vec
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 获取训练数据
url = "http://localhost:4000/api/paper"
papers = requests.get(url).json()[:15000]
corpus = []
label = []
for data in papers:
    text = data["title"].lower() + ". " + data["abstract"].lower()
    result = re.findall(r"[(.*?)]", text, re.M)
    for r in result:
        if "et" in r:
            text = text.replace(f" [{r}]", "")
    corpus.append(text)
    label.append(data["topo_label"])
# corpus = np.array(corpus)
# corpus_df = pd.DataFrame({"text": corpus, "label": label})
X_train, X_test, y_train, y_test = train_test_split(corpus, label, test_size=0.3)

nlp = spacy.load("en_core_web_lg")


def spacy_tokenizer(sample):
    doc = nlp(str(sample))
    tokens = [word.lemma_ for word in doc if not word.is_stop if word.is_alpha if len(word.text) > 1]
    return tokens


class TfidfWord2Vec(object):
    def __init__(self, dictionary, wv):
        self.dictionary = dictionary
        self.wv = wv

    def fit(self, X, y):
        return self

    # def transform(self, X):
    #     return np.array(
    #         [
    #             np.mean(
    #                 [
    #                     self.wv.get_vector(w) * self.dictionary[w] if w in self.dictionary else self.wv.get_vector(w)
    #                     for w in spacy_tokenizer(text)
    #                     if w in self.wv
    #                 ],
    #                 axis=0,
    #             )
    #             for text in X
    #         ]
    #     )
    def transform(self, X):
        return np.array(
            [np.mean([self.wv.get_vector(w) for w in spacy_tokenizer(text) if w in self.wv], axis=0,) for text in X]
        )


t0 = time()
print("开始")
TfidfVec = pickle.load(open("model/Tfidf_model_min5.pkl", "rb"))
wv = KeyedVectors.load("model/word2vec_sg_500.wv", mmap="r")
dictionary = dict(zip(TfidfVec.get_feature_names_out(), list(TfidfVec.idf_)))
# model = Word2Vec.load("./model/gensim-model-wa7o86qv")
# result = TfidfVec.transform(X_train)
# raw_tokens = [spacy_tokenizer(sample) for sample in corpus]


# classifier = LogisticRegression(solver="liblinear")
with open("model/logistic_sg_500.pkl", "rb") as f:
    classifier = pickle.load(f)

pipe = Pipeline([("word2vec", TfidfWord2Vec(dictionary, wv)), ("classifier", classifier)])
# pipe.fit(X_train, y_train)
# with open("model/logistic_sg_500.pkl", "wb") as f:
#     pickle.dump(classifier, f)

predicted = pipe.predict(X_test[:100])


# print("Accuracy:", metrics.accuracy_score(y_test, predicted))
# print("Precision:", metrics.precision_score(y_test, predicted))
# print("Recall:", metrics.recall_score(y_test, predicted))
# print("结束，总计用时：", time() - t0)
result = []
for i, x in enumerate(X_test[:100]):
    if predicted[i] + y_test[i] == 1:
        result.append((x, predicted[i], y_test[i]))
print(result)
