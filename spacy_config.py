import re
import string
from time import time

import spacy

# from spacy.tokenizer import Tokenizer
# from spacy.util import compile_infix_regex

# 初始化符号集和模型
punctuation = string.punctuation + "′‘’“”¯∼≃≈≠±≤≥≲×∘"
nlp = spacy.load("en_core_web_sm")

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
