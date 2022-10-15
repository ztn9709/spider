import nltk
import requests
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import MWETokenizer, RegexpTokenizer
from nltk.util import ngrams

url = "http://localhost:4000/api/paper"
tokens_bigram = []
tokens_trigram = []
tokens = []


def get_tokens(data):
    abs = data["abstract"]
    title = data["title"]
    # raw_text = title.lower() + ". " + abs.lower()
    raw_text = abs.lower()
    # 定义切词的pattern
    tokenizer = RegexpTokenizer(r"\w+(?:[-.]\w+)?")
    words = tokenizer.tokenize(raw_text)
    stopwords_list = set(stopwords.words("english"))
    words = [w for w in words if w not in stopwords_list]
    lem = WordNetLemmatizer()
    tags = nltk.pos_tag(words)
    for i, tag in enumerate(tags):
        if "NN" in tag[1]:
            words[i] = lem.lemmatize(words[i], "n")
        if "VB" in tag[1]:
            words[i] = lem.lemmatize(words[i], "v")
    tokens.extend(words)

    biwords = ["_".join(w) for w in ngrams(words, 2)]
    tokens_bigram.extend(biwords)
    triwords = ["_".join(w) for w in ngrams(words, 3)]
    tokens_trigram.extend(triwords)


if __name__ == "__main__":
    r = requests.get(url).json()
    for data in r:
        get_tokens(data)
    uni_voc = []
    dist_tri = FreqDist(tokens_trigram)
    for i in dist_tri:
        if dist_tri[i] >= 10:
            uni_voc.append((i.split("_")[0], i.split("_")[1], i.split("_")[2]))
    mwe_tokenizer = MWETokenizer(uni_voc)
    mwe_tokens = mwe_tokenizer.tokenize(tokens)
    dist_bi = FreqDist(tokens_bigram)
    for i in dist_bi:
        if dist_bi[i] >= 10:
            uni_voc.append((i.split("_")[0], i.split("_")[1]))
    mwe_tokenizer = MWETokenizer(uni_voc)
    mwe_tokens = mwe_tokenizer.tokenize(mwe_tokens)
    count = FreqDist(mwe_tokens)
    for c in count:
        if ("_" in c) & (count[c] > 30):
            print(c, count[c])
    # for i in dist_bi:
    #     if "state" in i:
    #         print(i + ":" + str(dist_bi[i]))

    # get_tokens(r[0])
