import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests
import spacy
from nltk import FreqDist
from spacy import displacy

nlp = spacy.load("en_core_sci_lg")
# text = "A central property of Chern insulators is the robustness of the topological phase and edge states to impurities in the system.Despite this, the Chern number cannot be straightforwardly calculated in the presence of disorder. Recently, work has been done to propose several local analogs of the Chern number, called local markers, that can be used to characterize disordered systems. However, it was unclear whether the proposed markers represented a physically measurable property of the system. Here we propose a local marker starting from a physical argument, as a local cross conductivity measured in the bulk of the system. We find the explicit form of the marker for a noninteracting system of electrons on the lattice and show that it corresponds to existing expressions for the Chern number. Examples are calculated for a variety of disordered and amorphous systems, showing that it is precisely quantized to the Chern number and robust against disorder."
# doc = nlp(text)
# displacy.serve(doc, style="ent")
# rows = []
# cols = ("text", "lemma", "POS", "DEP", "is_stopword")
# for t in doc:
#     row = [t.text, t.lemma_, t.pos_, t.dep_, t.is_stop]
#     rows.append(row)
# df = pd.DataFrame(rows, columns=cols)
# print(df)
url = "http://localhost:4000/api/paper"
papers = requests.get(url).json()[:1000]
corpus = [p["title"] for p in papers]
docs = list(nlp.pipe(corpus))
plt.rcParams.update({"figure.figsize": (16, 12)})
G = nx.Graph()
# G.add_node("topological insulators")
# edges = []
# for doc in docs:
#     if "topological insulators" in [ent.text for ent in doc.ents]:
#         for ent in doc.ents:
#             if ent.text != "topological insulators":
#                 edges.append(("topological insulators", ent.text))
# count = FreqDist(edges)
# edges = list(count.keys())[:15]
# G.add_edges_from(edges)
# print(nodes)

for doc in docs:
    nodes = []
    edges = []
    for i, ent in enumerate(doc.ents):
        nodes.append(ent)
        if i >= len(doc.ents) - 1:
            break
        for ent2 in doc.ents[i + 1 :]:
            edges.append((ent, ent2))
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
# print(nodes)
nx.draw_networkx(G, with_labels=False, node_size=5)
plt.show()
