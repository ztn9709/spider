import pytextrank
import requests
import spacy

url = "http://localhost:4000/api/paper"


# @spacy.registry.misc("nouns_scrubber")
# def nouns_scrubber():
#     def scrubber_func(span: Span) -> str:
#         for token in span:
#             if token.pos_ == "NOUN":
#                 token.text = token.lemma_
#         return span.text

#     return scrubber_func


nlp = spacy.load("en_core_web_lg")  # 导入模块en_core_web_lg
# add PyTextRank to the spaCy pipeline
nlp.add_pipe("positionrank")


def extract_keywords(text):
    doc = nlp(text)
    keywords = []
    rank = []
    # examine the top-ranked phrases in the document
    for phrase in doc._.phrases:
        if len(phrase.text.split(" ")) > 1 and len(keywords) < 5:
            keywords.append(phrase.text)
            rank.append(phrase.rank)
    return keywords, rank


if __name__ == "__main__":
    r = requests.get(url).json()
    stat_weight = {}
    for data in r:
        text = data["title"].lower() + ". " + data["abstract"].lower()
        result = extract_keywords(text)
        data["keywords"] = result[0]
        # requests.get(
        #     "http://localhost:4000/api/paper/update",
        #     params={"DOI": data["DOI"], "keywords": result[0]},
        # )
        for i, w in enumerate(result[0]):
            if w in stat_weight:
                stat_weight[w] += result[1][i]
            else:
                stat_weight[w] = result[1][i]
    weight_list = sorted(stat_weight.items(), key=lambda x: x[1], reverse=True)
    print(weight_list[:20])
