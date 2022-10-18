import pickle
import re
import string
from time import time

import numpy as np
import pandas as pd
import requests
import spacy
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# from spacy.tokenizer import Tokenizer
# from spacy.util import compile_infix_regex

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

# 初始化符号集和模型
punctuation = string.punctuation + "′‘’“”¯∼≃≈≠±≤≥≲×∘"
nlp = spacy.load("en_core_web_lg")

# 自定义规则，设置连字符连接的两个词看作一个
# inf = list(nlp.Defaults.infixes)
# inf = [x for x in inf if "-|–" not in x]
# infix_re = compile_infix_regex(tuple(inf))
# nlp.tokenizer = Tokenizer(
#     nlp.vocab,
#     prefix_search=nlp.tokenizer.prefix_search,
#     suffix_search=nlp.tokenizer.suffix_search,
#     infix_finditer=infix_re.finditer,
#     token_match=nlp.tokenizer.token_match,
#     rules=nlp.Defaults.tokenizer_exceptions,
# )


def spacy_tokenizer(sample):
    doc = nlp(str(sample))
    # tokens = [word.lemma_ for word in doc if not word.is_stop if word.pos_ != "SPACE"]
    # tokens = [token for token in tokens if not has_symbol(token)]
    # tokens = [token for token in tokens if not token[0].isdigit() if len(token) > 1]
    tokens = [word.lemma_ for word in doc if not word.is_stop if word.is_alpha if len(word.text) > 1]
    return tokens


def has_symbol(token):
    if token[0] in "-−" or token[-1] in "-−":
        return True
    punc_set = set(punctuation)
    punc_set.remove("-")
    token_set = set(token)
    if len(punc_set & token_set) > 0:
        return True
    else:
        return False


t0 = time()
print("开始TFIDF模型")
TfidfVec = TfidfVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1), decode_error="replace", max_df=0.8, min_df=3)
TfidfVec.fit(corpus)
vocs = TfidfVec.get_feature_names_out()
print(len(vocs))
print("结束TFIDF模型，总计用时：", time() - t0)
pickle.dump(TfidfVec, open("model/Tfidf_model_min3_2.pkl", "wb"))
t0 = time()
print("开始w2v模型")
raw_tokens = [spacy_tokenizer(sample) for sample in corpus]
model = Word2Vec(raw_tokens, min_count=3, vector_size=500, workers=4)
model.wv.save("model/word2vec_15000_2.wv")
print(len(model.wv.index_to_key))
print("结束w2v模型，总计用时：", time() - t0)
# with tempfile.NamedTemporaryFile(prefix="gensim-model-",dir="", delete=False) as tmp:
#     temporary_filepath = tmp.name
#     model.save(temporary_filepath)

# Tfidf_model = pickle.load(open("model/Tfidf_model.pkl", "rb"))
# vocs = Tfidf_model.get_feature_names_out()
# print(len(vocs), vocs[:100])
