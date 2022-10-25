import itertools
import pickle
import re
import tempfile
from time import time

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import spacy
from gensim.models import KeyedVectors, Word2Vec
from scipy import spatial
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 获取训练数据
url = "http://localhost:4000/api/paper"
papers = requests.get(url).json()[:15000]
corpus = []
label = []
for data in papers:
    text = data["title"] + ". " + data["abstract"]
    result = re.findall(r"[(.*?)]", text, re.M)
    for r in result:
        if "et" in r:
            text = text.replace(f" [{r}]", "")
    corpus.append(text)
    label.append(data["topo_label"])
X_train, X_test, y_train, y_test = train_test_split(corpus, label, test_size=0.3)

nlp = spacy.load("en_core_sci_lg")


def spacy_tokenizer(sample):
    doc = nlp(str(sample))
    tokens = [word.lemma_.lower() for word in doc if not word.is_stop if word.is_alpha if len(word.text) > 1]
    return tokens


def plot_confusion_matrix(cm, classes, title="Confusion matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def plot_LSA(test_data, test_labels, plot=True):
    lsa = TSNE(n_components=2, verbose=1, n_iter=500)
    lsa_scores = lsa.fit_transform(test_data)
    colors = ["orange", "blue"]
    if plot:
        plt.scatter(
            lsa_scores[:, 0],
            lsa_scores[:, 1],
            s=8,
            alpha=0.8,
            c=test_labels,
            cmap=matplotlib.colors.ListedColormap(colors),
        )
        red_patch = mpatches.Patch(color="orange", label="irrelevance")
        green_patch = mpatches.Patch(color="blue", label="topo")
        plt.legend(handles=[red_patch, green_patch], prop={"size": 30})


# t0 = time()
# print("开始CV")
# countVec = CountVectorizer(tokenizer=spacy_tokenizer, min_df=5)
# X_train_cv = countVec.fit_transform(X_train)
# X_test_cv = countVec.transform(X_test)

# TfidfVec = TfidfVectorizer(tokenizer=spacy_tokenizer, min_df=5)
# pickle.dump(TfidfVec, open("model/Tfidf_model.pkl", "wb"))
# TfidfVec = pickle.load(open("model/Tfidf_model.pkl", "rb"))
# dictionary = dict(zip(TfidfVec.get_feature_names_out(), list(TfidfVec.idf_)))


t0 = time()
print("开始分词")
raw_tokens = [spacy_tokenizer(sample) for sample in corpus]
print("结束分词，总计用时：", time() - t0)
t0 = time()
print("开始w2v模型")
model = Word2Vec(raw_tokens, min_count=5, vector_size=300, workers=4, sg=1, hs=1, window=10)

print(len(model.wv.index_to_key))
print("结束w2v模型，总计用时：", time() - t0)
# with tempfile.NamedTemporaryFile(prefix="gensim-model-", dir="D:\database\python\spider\model", delete=False) as tmp:
#     temporary_filepath = tmp.name
#     model.save(temporary_filepath)
# model = Word2Vec.load("./model/gensim-model-wa7o86qv")
# model.wv.save("model/word2vec.kv")
# wv = KeyedVectors.load("model/word2vec_win10.kv", mmap="r")

X_train_w2v = np.array(
    [np.mean([model.wv.get_vector(w) for w in spacy_tokenizer(text) if w in model.wv], axis=0) for text in X_train]
)
X_test_w2v = np.array(
    [np.mean([model.wv.get_vector(w) for w in spacy_tokenizer(text) if w in model.wv], axis=0) for text in X_test]
)
corpus_w2v = np.array(
    [np.mean([model.wv.get_vector(w) for w in spacy_tokenizer(text) if w in model.wv], axis=0) for text in corpus]
)
target_text = "Effect of magnetic impurity scattering on transport in topological insulators. Charge transport in topological insulators is primarily characterized by so-called topologically projected helical edge states, where charge carriers are correlated in spin and momentum. In principle, dissipationless current can be carried by these edge states as backscattering from impurities and defects is suppressed as long as time-reversal symmetry is not broken. However, applied magnetic fields or underlying nuclear spin defects in the substrate can break this time-reversal symmetry. In particular, magnetic impurities lead to backscattering by spin-flip processes. We have investigated the effects of pointwise magnetic impurities on the transport properties of helical edge states in the Bernevig, Hughes, and Zhang model using the nonequilibrium Green's function formalism and compared the results to a semianalytic approach. Using these techniques we study the influence of impurity strength and spin impurity polarization. We observe a secondary effect of defect-defect interaction that depends on the underlying material parameters which introduces a nonmonotonic response of the conductance to defect density. This in turn suggests a qualitative difference in magnetotransport signatures in the dilute and high-density spin impurity limits."
target_w2v = np.mean([model.wv.get_vector(w) for w in spacy_tokenizer(target_text) if w in model.wv], axis=0)
result = {}
for i, doc_w2v in enumerate(corpus_w2v):
    sim = 1 - spatial.distance.cosine(doc_w2v, target_w2v)
    result[papers[i]["title"]] = sim
list = sorted(result.items(), key=lambda x: x[1], reverse=True)
print(list[:10])

# # 降维可视化展示
# fig = plt.figure(figsize=(12, 12))
# plot_LSA(X_train_w2v, y_train)


# t0 = time()
# print("开始LR模型")
# # classifier = MultinomialNB()
# classifier = LogisticRegression()
# classifier.fit(X_train_w2v, y_train)
# predicted = classifier.predict(X_test_w2v)
# # with open("model/logistic.pkl", "rb") as f:
# #     classifier = pickle.load(f)
# # with open("model/logistic.pkl", "wb") as f:
# #     pickle.dump(classifier, f)

# # 结果展示
# cnf_matrix = metrics.confusion_matrix(y_test, predicted)
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=[0, 1])
# plt.show()
# print("Accuracy:", metrics.accuracy_score(y_test, predicted))
# print("Precision:", metrics.precision_score(y_test, predicted))
# print("Recall:", metrics.recall_score(y_test, predicted))
# print("结束，总计用时：", time() - t0)
