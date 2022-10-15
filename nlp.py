import requests
import spacy

url = "http://localhost:4000/api/paper"
r = requests.get(url).json()
for data in r:
    text = data["title"].lower() + ". " + data["abstract"].lower()

nlp = spacy.load("en_core_web_sm")
text = "I He She Quantized bulk conductivity as a local Chern marker. A central property of Chern insulators is the robustness of the topological phase and edge states to impurities in the system. Despite this, the Chern number cannot be straightforwardly calculated in the presence of disorder. Recently, work has been done to propose several local analogs of the Chern number, called local markers, that can be used to characterize disordered systems. However, it was unclear whether the proposed markers represented a physically measurable property of the system. Here we propose a local marker starting from a physical argument, as a local cross conductivity measured in the bulk of the system. We find the explicit form of the marker for a noninteracting system of electrons on the lattice and show that it corresponds to existing expressions for the Chern number. Examples are calculated for a variety of disordered and amorphous systems, showing that it is precisely quantized to the Chern number and robust against disorder."
doc = nlp(text)

tokens = [word.lemma_.lower().strip() for word in doc if not word.is_stop if word.pos_ != 'PUNCT']

for word in doc[:10]:
    print(word.vector)
